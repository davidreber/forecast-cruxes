"""Reading a premise file and selecting the premises a run scores.

Ported from test_premise_sets.py for the introspected/extracted vocabulary.
The cache key construction and the GPU/CPU key equivalence that used to live
here moved with build_score_cache_key to cruxes.scorecache and cruxes.runplan
and are tested in test_score_cache.py and test_runplan.py instead.
"""

import json

import pytest

from cruxes.premises import PremiseSet, load_premise_sets, select_premises

QUESTIONS = {
    "questions": [
        {
            "question_id": "second-question",
            "introspected": [f"i{i}" for i in range(8)],
            "extracted": [f"e{i}" for i in range(3)],
        },
        {
            "question_id": "first-question",
            "introspected": [f"I{i}" for i in range(4)],
            "extracted": [f"E{i}" for i in range(9)],
        },
    ]
}


def write_questions(tmp_path, data=None):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(data if data is not None else QUESTIONS))
    return path


def test_a_premise_file_reads_into_questions(tmp_path):
    sets = load_premise_sets(write_questions(tmp_path))
    assert [s.question_id for s in sets] == ["second-question", "first-question"]
    assert sets[0].introspected[0] == "i0"
    assert isinstance(sets[0].extracted, tuple)


def test_a_bare_list_reads_too(tmp_path):
    sets = load_premise_sets(write_questions(tmp_path, QUESTIONS["questions"]))
    assert len(sets) == 2


def test_a_missing_pool_names_itself(tmp_path):
    path = write_questions(tmp_path, {"questions": [{"question_id": "q", "introspected": []}]})
    with pytest.raises(ValueError, match="extracted"):
        load_premise_sets(path)


def test_a_repeated_question_id_is_refused(tmp_path):
    record = {"question_id": "q", "introspected": ["x"], "extracted": ["y"]}
    path = write_questions(tmp_path, {"questions": [record, record]})
    with pytest.raises(ValueError, match="duplicate"):
        load_premise_sets(path)


def test_an_empty_pool_is_refused():
    """Nothing to compare, so the question should be left out upstream instead."""
    with pytest.raises(ValueError, match="extracted"):
        PremiseSet("q", introspected=["x"], extracted=[])


def test_a_non_string_premise_is_refused():
    with pytest.raises(ValueError, match="not strings"):
        PremiseSet("q", introspected=["x", 5], extracted=["y"])


def test_selection_is_a_nested_prefix():
    """A curve across contrast counts is nested, not a fresh draw each time."""
    premise_set = PremiseSet("q", [f"i{i}" for i in range(20)], [f"e{i}" for i in range(20)])
    small_i, small_e = select_premises(premise_set, 5)
    large_i, large_e = select_premises(premise_set, 12)
    assert large_i[:5] == small_i
    assert large_e[:5] == small_e


def test_selection_is_capped_by_what_the_pool_holds():
    premise_set = PremiseSet("q", ["only one"], [f"e{i}" for i in range(4)])
    introspected, extracted = select_premises(premise_set, 25)
    assert introspected == ["only one"]
    assert len(extracted) == 4


def test_selection_differs_between_questions_and_between_pools():
    first = PremiseSet("first", [f"x{i}" for i in range(9)], [f"x{i}" for i in range(9)])
    second = PremiseSet("second", first.introspected, first.extracted)
    intro_one, extr_one = select_premises(first, 9)
    intro_two, _ = select_premises(second, 9)
    assert intro_one != intro_two, "two questions must not share a permutation"
    assert intro_one != extr_one, "the two pools of one question must not share a permutation"


def test_selection_rejects_a_non_positive_count():
    premise_set = PremiseSet("q", ["x"], ["y"])
    with pytest.raises(ValueError, match="at least 1"):
        select_premises(premise_set, 0)


def test_other_keys_in_a_question_record_are_ignored(tmp_path):
    """Forecasts, question text, options or per premise records may travel in
    the same file; this pipeline reads the id and the two pools and nothing else."""
    record = {
        "question_id": "q",
        "question": "Who wins?",
        "options": ["A", "B"],
        "forecasts": [{"forecast_id": "f1", "prediction": {"A": 0.7, "B": 0.3}}],
        "introspected": ["x", "y"],
        "extracted": ["z"],
        "premise_records": [{"premise_id": "p0", "pool": "introspected", "text": "x"}],
    }
    sets = load_premise_sets(write_questions(tmp_path, {"questions": [record]}))
    assert len(sets) == 1
    assert sets[0].introspected == ("x", "y")
    assert sets[0].extracted == ("z",)
    assert set(vars(sets[0])) == {"question_id", "introspected", "extracted"}
