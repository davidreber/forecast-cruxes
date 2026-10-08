"""The CPU half of the end to end path, on a question the paper never saw.

The GPU stage writes a cache under a key built from the premise file. The CPU
stage rebuilds the same key, finds the same file and writes the report. These
tests do the second half with a cache built from the committed pinned_scores
fixture, so the plumbing between the two stages is checked without a GPU.

pinned_scores.json replaces the old reranker_pinned_scores.json (side_a/side_b
vocabulary); cruxes.report replaces cruxes.set_inclusion_report and takes
--out-dir rather than --out, writing a timestamped file rather than one the
caller names.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cruxes.matching import DEFAULT_MATCHING_CRITERION
from cruxes.premises import PremiseSet, select_premises
from cruxes.reranker.model import DEFAULT_BATCH_SIZE, RERANKER_MODEL_NAME, RERANKER_REVISION
from cruxes.reranker.prompt import DEFAULT_RERANKER_PROMPT
from cruxes.report import main as report_main
from cruxes.scorecache import build_score_cache_key, write_score_cache
from cruxes.scoring import SCORE_KEYS

FIXTURE = Path(__file__).parent / "fixtures" / "pinned_scores.json"


@pytest.fixture(scope="module")
def recorded():
    with open(FIXTURE) as handle:
        return json.load(handle)


def write_premise_file(path: Path, recorded: dict) -> PremiseSet:
    premise_set = PremiseSet("fixture-question", recorded["introspected"], recorded["extracted"])
    path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "question_id": premise_set.question_id,
                        "introspected": list(premise_set.introspected),
                        "extracted": list(premise_set.extracted),
                    }
                ]
            }
        )
    )
    return premise_set


def cache_from_fixture(cache_dir: Path, premise_set: PremiseSet, recorded: dict, count: int, seed: int = 42):
    """Write the fixture's scores as a cache for this question.

    The selection permutes both pools, so the fixture's rows have to be
    permuted the same way. Doing that rather than pretending the order is the
    identity is what makes this a test of the correspondence between the
    selected premises and the matrix rows.
    """
    selected_introspected, selected_extracted = select_premises(premise_set, count, seed)
    n_introspected = len(recorded["introspected"])
    order = [recorded["introspected"].index(text) for text in selected_introspected]
    order += [n_introspected + recorded["extracted"].index(text) for text in selected_extracted]
    index = np.ix_(order, order)

    key = build_score_cache_key(
        selected_introspected + selected_extracted,
        criterion=DEFAULT_MATCHING_CRITERION,
        prompt=DEFAULT_RERANKER_PROMPT,
        model_name=RERANKER_MODEL_NAME,
        model_revision=RERANKER_REVISION,
        batch_size=DEFAULT_BATCH_SIZE,
    )
    matrices = {name: np.array(recorded[name])[index] for name in SCORE_KEYS}
    pools = {"introspected": len(selected_introspected), "extracted": len(selected_extracted)}
    return key, write_score_cache(cache_dir, key, matrices, environment={"note": "test"}, pools=pools)


def test_the_report_is_built_from_the_cache_the_key_points_at(tmp_path, recorded):
    premises = tmp_path / "questions.json"
    premise_set = write_premise_file(premises, recorded)
    cache_dir = tmp_path / "caches"
    cache_from_fixture(cache_dir, premise_set, recorded, count=3)

    out_dir = tmp_path / "out"
    status = report_main(
        [
            "--premises", str(premises),
            "--cache-dir", str(cache_dir),
            "--out-dir", str(out_dir),
            "--max-contrasts-per-side", "3",
            "--permutations", "50",
        ]
    )
    assert status == 0

    written = list(out_dir.glob("set_inclusion_*.json"))
    assert len(written) == 1
    report = json.loads(written[0].read_text())
    assert set(report["per_contrasts_per_side"]) == {"1", "2", "3"}
    entry = report["per_contrasts_per_side"]["3"]
    assert entry["summary"]["n_questions"] == 1
    venn = entry["per_question"]["fixture-question"]
    assert venn["shared"] + venn["introspected_only"] + venn["extracted_only"] == venn["total"]
    assert 0.0 <= venn["shared_frac"] <= 1.0
    assert "permutation" in venn
    assert report["metadata"]["model_revision"] == RERANKER_REVISION
    assert "legacy_constant_seed" not in report["metadata"]


def test_a_missing_cache_stops_the_report_and_names_the_question(tmp_path, recorded, capsys):
    premises = tmp_path / "questions.json"
    write_premise_file(premises, recorded)
    out_dir = tmp_path / "out"
    status = report_main(
        [
            "--premises", str(premises),
            "--cache-dir", str(tmp_path / "empty"),
            "--out-dir", str(out_dir),
            "--max-contrasts-per-side", "3",
            "--no-permutation",
        ]
    )
    assert status == 1
    assert "fixture-question" in capsys.readouterr().out
    assert not out_dir.exists() or not list(out_dir.glob("*.json"))


def test_a_changed_premise_makes_the_cache_unfindable(tmp_path, recorded):
    """Editing a premise must not silently reuse scores of the old one."""
    premises = tmp_path / "questions.json"
    premise_set = write_premise_file(premises, recorded)
    cache_dir = tmp_path / "caches"
    cache_from_fixture(cache_dir, premise_set, recorded, count=3)

    edited = json.loads(premises.read_text())
    edited["questions"][0]["introspected"][0] += " (reworded)"
    premises.write_text(json.dumps(edited))

    status = report_main(
        [
            "--premises", str(premises),
            "--cache-dir", str(cache_dir),
            "--out-dir", str(tmp_path / "out"),
            "--max-contrasts-per-side", "3",
            "--no-permutation",
        ]
    )
    assert status == 1


def test_the_report_writes_only_the_set_inclusion_file(tmp_path, recorded):
    """The baselines are procedures outside the package, not report output."""
    premises = tmp_path / "questions.json"
    premise_set = write_premise_file(premises, recorded)
    cache_dir = tmp_path / "caches"
    cache_from_fixture(cache_dir, premise_set, recorded, count=3)

    out_dir = tmp_path / "out"
    status = report_main(
        [
            "--premises", str(premises),
            "--cache-dir", str(cache_dir),
            "--out-dir", str(out_dir),
            "--max-contrasts-per-side", "3",
            "--permutations", "20",
        ]
    )
    assert status == 0
    assert [p.name.split("_2")[0] for p in out_dir.iterdir()] == ["set_inclusion"]


def test_no_latest_file_or_symlink_is_ever_written(tmp_path, recorded):
    premises = tmp_path / "questions.json"
    premise_set = write_premise_file(premises, recorded)
    cache_dir = tmp_path / "caches"
    cache_from_fixture(cache_dir, premise_set, recorded, count=3)

    out_dir = tmp_path / "out"
    report_main(
        [
            "--premises", str(premises),
            "--cache-dir", str(cache_dir),
            "--out-dir", str(out_dir),
            "--max-contrasts-per-side", "3",
            "--no-permutation",
        ]
    )
    for path in out_dir.iterdir():
        assert "latest" not in path.name
        assert not path.is_symlink()


def test_the_shipped_example_file_reads():
    """The file a reader is told to run the package on.

    One question's extracted pool repeats its first premise twice, once
    exactly and once with a double space, so the worked example shows the
    count of repeats removed before selection.
    """
    from cruxes.dedupe import normalize_premise_text
    from cruxes.premises import dedupe_premise_set, load_premise_sets

    example = Path(__file__).parents[1] / "examples" / "premises_example.json"
    sets = {s.question_id: s for s in load_premise_sets(example)}
    assert {q: (len(s.introspected), len(s.extracted)) for q, s in sets.items()} == {
        "rail-tunnel-opens-2027": (6, 6),
        "drug-approval-by-june": (6, 8),
        "election-turnout-above-sixty": (6, 6),
    }

    repeated = sets["drug-approval-by-june"]
    _, counts = dedupe_premise_set(repeated)
    assert counts == {
        "introspected": {"given": 6, "kept": 6, "removed": 0},
        "extracted": {"given": 8, "kept": 6, "removed": 2},
    }
    introspected, extracted = select_premises(repeated, 25)
    assert len(introspected) == 6
    assert len(extracted) == len({normalize_premise_text(text) for text in extracted}) == 6
