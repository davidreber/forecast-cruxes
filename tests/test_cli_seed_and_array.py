"""Two promises of the command line, checked.

``--seed`` is the base of every derived seed, the permutation null's included,
and the report says which base it used. ``--array-task-id`` and
``--num-array-tasks`` go together, partition the questions without overlap,
and an id out of range is refused before anything is planned.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cruxes.reranker.score import main as score_main
from cruxes.report import main as report_main

from test_end_to_end_cpu import FIXTURE, cache_from_fixture, write_premise_file


@pytest.fixture(scope="module")
def recorded():
    return json.loads(FIXTURE.read_text())


def _report(tmp_path: Path, recorded: dict, seed: int, name: str) -> dict:
    premises = tmp_path / f"questions_{name}.json"
    premise_set = write_premise_file(premises, recorded)
    cache_dir = tmp_path / f"caches_{name}"
    cache_from_fixture(cache_dir, premise_set, recorded, count=3, seed=seed)
    out_dir = tmp_path / f"out_{name}"
    status = report_main(
        [
            "--premises", str(premises), "--cache-dir", str(cache_dir), "--out-dir", str(out_dir),
            "--max-contrasts-per-side", "3", "--seed", str(seed), "--permutations", "200",
        ]
    )
    assert status == 0
    return json.loads(next(out_dir.glob("set_inclusion_*.json")).read_text())


def test_the_seed_reaches_the_permutation_null_and_the_metadata(tmp_path, recorded):
    """Two bases that select the same premises still draw different nulls."""
    # The fixture has 3+3 premises and the selection takes all of them, so the
    # cache key is the same under any base seed and only the null can move.
    first = _report(tmp_path, recorded, 42, "a")
    second = _report(tmp_path, recorded, 7, "b")
    assert first["metadata"]["seed"] == 42 and second["metadata"]["seed"] == 7
    assert "base seed 42" in first["metadata"]["permutation_seeding"]
    assert "base seed 7" in second["metadata"]["permutation_seeding"]
    nulls = [
        r["per_contrasts_per_side"]["3"]["per_question"]["fixture-question"]["permutation"]["permutation_shared_mean"]
        for r in (first, second)
    ]
    assert nulls[0] != nulls[1]
    again = _report(tmp_path, recorded, 7, "c")
    assert again["per_contrasts_per_side"]["3"]["permutation"] == second["per_contrasts_per_side"]["3"]["permutation"]


QUESTIONS = {
    "questions": [
        {"question_id": f"q{i}", "introspected": [f"i{i}a", f"i{i}b"], "extracted": [f"e{i}a", f"e{i}b"]}
        for i in range(3)
    ]
}


def _dry_run(tmp_path: Path, capsys, extra: list) -> str:
    premises = tmp_path / "questions.json"
    premises.write_text(json.dumps(QUESTIONS))
    status = score_main(["--premises", str(premises), "--cache-dir", str(tmp_path / "c"), "--dry-run"] + extra)
    assert status == 0
    return capsys.readouterr().out


def test_array_tasks_partition_the_questions_without_overlap(tmp_path, capsys):
    seen = []
    for task in range(2):
        out = _dry_run(tmp_path, capsys, ["--array-task-id", str(task), "--num-array-tasks", "2"])
        seen.append({line.split(":")[0].strip() for line in out.splitlines() if line.startswith("  q")})
    assert seen[0] == {"q0", "q2"} and seen[1] == {"q1"}


def test_one_array_flag_alone_is_refused(tmp_path):
    premises = tmp_path / "questions.json"
    premises.write_text(json.dumps(QUESTIONS))
    base = ["--premises", str(premises), "--cache-dir", str(tmp_path / "c"), "--dry-run"]
    with pytest.raises(SystemExit):
        score_main(base + ["--array-task-id", "0"])
    with pytest.raises(SystemExit):
        score_main(base + ["--num-array-tasks", "2"])
    with pytest.raises(SystemExit):
        score_main(base + ["--array-task-id", "2", "--num-array-tasks", "2"])
