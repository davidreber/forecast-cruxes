"""Repeated premise texts are removed from each pool before selection.

The rule: two premises of one pool are repeats when
their texts are equal after collapsing whitespace runs to one space, case
sensitive, with no fuzzy matching. The first occurrence is kept verbatim in
its place. select_premises applies the rule by default, plan() records the
counts per question, cruxes-score prints them and cruxes-report writes them
into the report's metadata.

The report test reuses the pattern of test_end_to_end_cpu.py: a cache built
from the pinned_scores fixture for the selection the report will recompute,
here the selection from the deduplicated pools.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from cruxes.dedupe import DEDUPE_RULE, dedupe_premises, normalize_premise_text
from cruxes.matching import DEFAULT_MATCHING_CRITERION
from cruxes.premises import PremiseSet, dedupe_premise_set, select_premises
from cruxes.report import build_parser as cpu_parser
from cruxes.report import main as report_main
from cruxes.reranker.model import DEFAULT_BATCH_SIZE, RERANKER_MODEL_NAME, RERANKER_REVISION
from cruxes.reranker.prompt import DEFAULT_RERANKER_PROMPT
from cruxes.reranker.score import main as score_main
from cruxes.runplan import deduplication_summary, plan, settings_from_args
from cruxes.scorecache import build_score_cache_key, write_score_cache
from cruxes.scoring import SCORE_KEYS

FIXTURE = Path(__file__).parent / "fixtures" / "pinned_scores.json"

#: Three questions. alpha repeats one introspected text, beta repeats one
#: extracted text twice (by whitespace only), gamma repeats in both pools.
QUESTIONS = {
    "questions": [
        {
            "question_id": "gamma",
            "introspected": ["e", "e", "f", "e"],
            "extracted": ["u", "v", "u"],
        },
        {
            "question_id": "alpha",
            "introspected": ["a", "a", "b"],
            "extracted": ["x", "y"],
        },
        {
            "question_id": "beta",
            "introspected": ["c", "d"],
            "extracted": ["z", "z ", "w", "  z"],
        },
    ]
}

EXPECTED_PER_QUESTION = {
    "alpha": {
        "introspected": {"given": 3, "kept": 2, "removed": 1},
        "extracted": {"given": 2, "kept": 2, "removed": 0},
    },
    "beta": {
        "introspected": {"given": 2, "kept": 2, "removed": 0},
        "extracted": {"given": 4, "kept": 2, "removed": 2},
    },
    "gamma": {
        "introspected": {"given": 4, "kept": 2, "removed": 2},
        "extracted": {"given": 3, "kept": 2, "removed": 1},
    },
}


def write_premises(tmp_path: Path, data=None) -> Path:
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(data if data is not None else QUESTIONS))
    return path


def default_settings(tmp_path: Path, premises: Path):
    args = cpu_parser().parse_args(
        ["--premises", str(premises), "--cache-dir", str(tmp_path), "--out-dir", str(tmp_path)]
    )
    return settings_from_args(args)


# The rule.


def test_normalize_collapses_whitespace_and_keeps_case():
    assert normalize_premise_text("Will it rain?") == "Will it rain?"
    assert normalize_premise_text("Will  it   rain?") == "Will it rain?"
    assert normalize_premise_text("  Will it rain?\n") == "Will it rain?"
    assert normalize_premise_text("Will\tit\nrain?") == "Will it rain?"
    assert normalize_premise_text("Will it Rain?") == "Will it Rain?"
    assert normalize_premise_text("   ") == ""


def test_whitespace_runs_leading_and_trailing_whitespace_tabs_and_newlines_make_repeats():
    pool = ["Will it rain?", "Will  it rain?", "  Will it rain? ", "Will\tit\nrain?"]
    kept, dropped = dedupe_premises(pool)
    assert kept == ["Will it rain?"]
    assert dropped == [1, 2, 3]


def test_case_differences_are_not_repeats():
    """No case folding: "A b" and "a b" are two premises."""
    kept, dropped = dedupe_premises(["A b", "a b", "A B"])
    assert kept == ["A b", "a b", "A B"]
    assert dropped == []


def test_near_duplicates_are_not_repeats():
    """No fuzzy matching: punctuation or one changed word keeps both."""
    pool = ["Will it rain?", "Will it rain", "Will it snow?"]
    kept, dropped = dedupe_premises(pool)
    assert kept == pool
    assert dropped == []


def test_the_first_occurrence_is_kept_verbatim_in_its_place():
    pool = ["x", "  Rain  tomorrow? ", "y", "Rain tomorrow?", "z"]
    kept, dropped = dedupe_premises(pool)
    assert kept == ["x", "  Rain  tomorrow? ", "y", "z"]
    assert dropped == [3]


def test_dropped_indices_are_the_later_copies_ascending():
    kept, dropped = dedupe_premises(["a", "b", "a", "c", "b", "a"])
    assert kept == ["a", "b", "c"]
    assert dropped == [2, 4, 5]


def test_a_pool_without_repeats_comes_back_unchanged():
    pool = ["one", "two", "three"]
    kept, dropped = dedupe_premises(pool)
    assert kept == pool
    assert dropped == []
    assert pool == ["one", "two", "three"], "the input list is not modified"


def test_the_rule_is_stated_in_words():
    """The sentence every report records says what was done."""
    assert "case sensitive" in DEDUPE_RULE
    assert "first occurrence" in DEDUPE_RULE


# One question.


def test_dedupe_premise_set_counts_both_pools():
    premise_set = PremiseSet("q", ["a", "a ", "b"], ["x", "y", "x", "x"])
    deduped, counts = dedupe_premise_set(premise_set)
    assert deduped == PremiseSet("q", ["a", "b"], ["x", "y"])
    assert counts == {
        "introspected": {"given": 3, "kept": 2, "removed": 1},
        "extracted": {"given": 4, "kept": 2, "removed": 2},
    }


def test_dedupe_premise_set_with_none_keeps_every_pool_as_given():
    premise_set = PremiseSet("q", ["a", "a ", "b"], ["x", "y", "x", "x"])
    kept, counts = dedupe_premise_set(premise_set, dedupe=None)
    assert kept == premise_set
    assert counts == {
        "introspected": {"given": 3, "kept": 3, "removed": 0},
        "extracted": {"given": 4, "kept": 4, "removed": 0},
    }


# Selection.

REPEATED_POOL = ["the same text"] * 10 + ["first distinct", "second distinct", "third distinct"]


@pytest.mark.parametrize("seed", [0, 1, 42, 2026])
@pytest.mark.parametrize("count", [2, 4, 25])
def test_selection_never_draws_a_repeat_by_default(seed, count):
    premise_set = PremiseSet("q", REPEATED_POOL, REPEATED_POOL)
    for selected in select_premises(premise_set, count, seed):
        assert len(selected) == len(set(selected)) == min(count, 4)
        assert set(selected) <= set(REPEATED_POOL)


def test_selection_with_dedupe_none_can_draw_a_repeat():
    premise_set = PremiseSet("q", REPEATED_POOL, REPEATED_POOL)
    introspected, extracted = select_premises(premise_set, 25, dedupe=None)
    assert len(introspected) == len(extracted) == 13
    assert introspected.count("the same text") == 10


@pytest.mark.parametrize("seed", [0, 42, 2026])
def test_dedupe_does_not_change_the_selection_of_a_pool_without_repeats(seed):
    """Same seed, same prefix: removing nothing leaves the draw alone."""
    premise_set = PremiseSet("q", [f"i{i}" for i in range(20)], [f"e{i}" for i in range(15)])
    for count in (1, 5, 12, 25):
        assert select_premises(premise_set, count, seed) == select_premises(
            premise_set, count, seed, dedupe=None
        )


def test_a_dedupe_function_passed_to_select_premises_is_used():
    """A caller's own rule replaces the package's, here one that removes nothing."""
    calls = []

    def keep_everything(pool):
        calls.append(list(pool))
        return list(pool), []

    premise_set = PremiseSet("q", REPEATED_POOL, REPEATED_POOL)
    selected = select_premises(premise_set, 25, dedupe=keep_everything)
    assert calls == [REPEATED_POOL, REPEATED_POOL]
    assert selected == select_premises(premise_set, 25, dedupe=None)
    assert selected[0].count("the same text") == 10


# The run plan.


def test_plan_carries_the_counts_of_every_question(tmp_path):
    premises = write_premises(tmp_path)
    planned = plan(premises, default_settings(tmp_path, premises))
    assert [p.premise_set.question_id for p in planned] == ["alpha", "beta", "gamma"]
    assert {p.premise_set.question_id: p.duplicates for p in planned} == EXPECTED_PER_QUESTION


def test_plan_keeps_the_question_as_given_and_keys_the_deduplicated_selection(tmp_path):
    premises = write_premises(tmp_path)
    alpha = plan(premises, default_settings(tmp_path, premises))[0]
    assert alpha.premise_set.introspected == ("a", "a", "b")
    assert alpha.pools == {"introspected": 2, "extracted": 2}
    assert sorted(alpha.key.items[:2]) == ["a", "b"]
    assert sorted(alpha.key.items[2:]) == ["x", "y"]


def test_appending_repeats_to_a_file_does_not_change_its_keys(tmp_path):
    """Later copies are dropped and the first kept, so the selection is the same."""
    clean = {
        "questions": [
            {"question_id": "q", "introspected": ["p", "q", "r"], "extracted": ["s", "t"]}
        ]
    }
    repeated = {
        "questions": [
            {
                "question_id": "q",
                "introspected": ["p", "q", "r", "q", " p"],
                "extracted": ["s", "t", "s  "],
            }
        ]
    }
    (tmp_path / "clean").mkdir()
    (tmp_path / "repeated").mkdir()
    clean_path = write_premises(tmp_path / "clean", clean)
    repeated_path = write_premises(tmp_path / "repeated", repeated)
    settings = default_settings(tmp_path, clean_path)
    for count in (1, 2, 25):
        settings = dataclasses.replace(settings, max_contrasts_per_side=count)
        assert plan(clean_path, settings)[0].key == plan(repeated_path, settings)[0].key


def test_deduplication_summary_totals_the_run(tmp_path):
    premises = write_premises(tmp_path)
    summary = deduplication_summary(plan(premises, default_settings(tmp_path, premises)))
    assert summary == {
        "rule": DEDUPE_RULE,
        "removed": {"introspected": 3, "extracted": 3},
        "questions_with_repeats": {"introspected": 2, "extracted": 2},
        "per_question": EXPECTED_PER_QUESTION,
    }
    assert json.loads(json.dumps(summary)) == summary


def test_deduplication_summary_of_a_run_without_repeats_is_zero(tmp_path):
    data = {"questions": [{"question_id": "q", "introspected": ["a", "b"], "extracted": ["c"]}]}
    premises = write_premises(tmp_path, data)
    summary = deduplication_summary(plan(premises, default_settings(tmp_path, premises)))
    assert summary["removed"] == {"introspected": 0, "extracted": 0}
    assert summary["questions_with_repeats"] == {"introspected": 0, "extracted": 0}


# The two commands.


def test_the_report_records_the_repeats_it_removed(tmp_path):
    recorded = json.loads(FIXTURE.read_text())
    introspected = list(recorded["introspected"])
    extracted = list(recorded["extracted"])
    first, second = introspected[0], extracted[1]
    with_repeats = PremiseSet(
        "fixture-question",
        introspected[:2] + [first] + introspected[2:],
        extracted + [second.replace(" ", "  ")],
    )
    premises = write_premises(
        tmp_path,
        {
            "questions": [
                {
                    "question_id": with_repeats.question_id,
                    "introspected": list(with_repeats.introspected),
                    "extracted": list(with_repeats.extracted),
                }
            ]
        },
    )

    # The cache of the deduplicated selection, rows permuted to match it.
    selected_introspected, selected_extracted = select_premises(with_repeats, 3)
    assert len(set(selected_introspected)) == len(set(selected_extracted)) == 3
    order = [introspected.index(text) for text in selected_introspected]
    order += [len(introspected) + extracted.index(text) for text in selected_extracted]
    key = build_score_cache_key(
        selected_introspected + selected_extracted,
        criterion=DEFAULT_MATCHING_CRITERION,
        prompt=DEFAULT_RERANKER_PROMPT,
        model_name=RERANKER_MODEL_NAME,
        model_revision=RERANKER_REVISION,
        batch_size=DEFAULT_BATCH_SIZE,
    )
    index = np.ix_(order, order)
    matrices = {name: np.array(recorded[name])[index] for name in SCORE_KEYS}
    cache = write_score_cache(
        tmp_path / "caches", key, matrices, environment={"note": "test"},
        pools={"introspected": 3, "extracted": 3},
    )

    out_dir = tmp_path / "out"
    status = report_main(
        [
            "--premises", str(premises),
            "--cache-dir", str(tmp_path / "caches"),
            "--out-dir", str(out_dir),
            "--max-contrasts-per-side", "3",
            "--no-permutation",
        ]
    )
    assert status == 0
    (written,) = out_dir.glob("set_inclusion_*.json")
    metadata = json.loads(written.read_text())["metadata"]
    assert metadata["score_caches"] == {"fixture-question": cache.name}
    assert metadata["deduplication"] == {
        "rule": DEDUPE_RULE,
        "removed": {"introspected": 1, "extracted": 1},
        "questions_with_repeats": {"introspected": 1, "extracted": 1},
        "per_question": {
            "fixture-question": {
                "introspected": {"given": 4, "kept": 3, "removed": 1},
                "extracted": {"given": 4, "kept": 3, "removed": 1},
            }
        },
    }


def test_the_score_dry_run_prints_the_repeats_removed(tmp_path, capsys):
    premises = write_premises(tmp_path)
    status = score_main(
        ["--premises", str(premises), "--cache-dir", str(tmp_path / "caches"), "--dry-run"]
    )
    assert status == 0
    out = capsys.readouterr().out
    assert "alpha: 2+2 items (1+0 repeated texts removed)" in out
    assert "beta: 2+2 items (0+2 repeated texts removed)" in out
    assert "gamma: 2+2 items (2+1 repeated texts removed)" in out
    assert (
        "repeated texts removed before selection: "
        "introspected 3 over 2 of 3 questions, extracted 3 over 2 of 3 questions"
    ) in out
    assert not (tmp_path / "caches").exists() or not list((tmp_path / "caches").iterdir())
