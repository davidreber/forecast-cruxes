"""The matching criterion: what the cross encoder is asked about a pair.

New module in the restructure. The package ships exactly one criterion,
DEFAULT_MATCHING_CRITERION, named "resolution"; a caller replaces it with
their own or loads one from a file with load_matching_criterion.
"""

import json

import pytest

from cruxes.matching import DEFAULT_MATCHING_CRITERION, MatchingCriterion, load_matching_criterion


def test_default_matching_criterion_is_named_resolution():
    assert DEFAULT_MATCHING_CRITERION.name == "resolution"
    assert DEFAULT_MATCHING_CRITERION.positive_instruction
    assert DEFAULT_MATCHING_CRITERION.negative_instruction


def test_load_matching_criterion_defaults_to_the_shipped_one():
    assert load_matching_criterion(None) is DEFAULT_MATCHING_CRITERION


def test_load_matching_criterion_reads_a_file(tmp_path):
    path = tmp_path / "criterion.json"
    path.write_text(
        json.dumps(
            {
                "name": "topical",
                "positive_instruction": "same topic?",
                "negative_instruction": "different topic?",
            }
        )
    )
    criterion = load_matching_criterion(path)
    assert criterion == MatchingCriterion("topical", "same topic?", "different topic?")


def test_load_matching_criterion_defaults_the_name_to_the_filename(tmp_path):
    path = tmp_path / "unnamed_criterion.json"
    path.write_text(
        json.dumps({"positive_instruction": "same?", "negative_instruction": "different?"})
    )
    criterion = load_matching_criterion(path)
    assert criterion.name == "unnamed_criterion"


def test_load_matching_criterion_requires_both_instructions(tmp_path):
    path = tmp_path / "criterion.json"
    path.write_text(json.dumps({"positive_instruction": "same?"}))
    with pytest.raises(ValueError, match="negative_instruction"):
        load_matching_criterion(path)


def test_matching_criterion_to_dict_round_trips():
    criterion = MatchingCriterion("name", "pos", "neg")
    assert criterion.to_dict() == {"name": "name", "positive_instruction": "pos", "negative_instruction": "neg"}
