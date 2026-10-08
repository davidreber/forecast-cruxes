"""What the pinned model actually said, checked on a CPU.

``tests/fixtures/pinned_scores.json`` holds three introspected and three
extracted premises, the sixty prompt scores the pinned reranker gave them
through this package's scoring stage at batch size thirty two, the four
assembled matrices and the pos_only aggregation of them. It replaces the old
reranker_pinned_scores.json fixture (side_a/side_b vocabulary, a debiased
aggregation) with the introspected/extracted one this package now ships.

These tests feed the recorded raw scores to the package's own assembly with
the model call stubbed out, and rebuild everything downstream of them, so the
layout of the four matrices and the aggregation are testable without a GPU.
The layout is the part worth testing: which of the two directions of a pair
lands in the upper triangle of which matrix is easy to get backwards and
impossible to notice three stages later.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cruxes.matching import DEFAULT_MATCHING_CRITERION
from cruxes.scoring import SCORE_KEYS, ScoreMatrices, aggregate_scores

FIXTURE = Path(__file__).parent / "fixtures" / "pinned_scores.json"


@pytest.fixture(scope="module")
def recorded():
    with open(FIXTURE) as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def matrices(recorded):
    return ScoreMatrices.from_mapping(
        {
            **{name: recorded[name] for name in SCORE_KEYS},
            "n_introspected": len(recorded["introspected"]),
            "n_extracted": len(recorded["extracted"]),
        }
    )


def test_the_fixture_came_from_the_pinned_revision_and_a_fixed_batch(recorded):
    from cruxes.reranker.model import (
        DEFAULT_BATCH_SIZE,
        RERANKER_MODEL_NAME,
        RERANKER_REVISION,
    )
    from cruxes.reranker.prompt import RERANKER_PROMPT_TEMPLATE, RERANKER_SYSTEM_MESSAGE

    assert recorded["model_name"] == RERANKER_MODEL_NAME
    assert recorded["model_revision"] == RERANKER_REVISION
    assert recorded["batch_size"] == DEFAULT_BATCH_SIZE
    assert recorded["prompt_template"] == RERANKER_PROMPT_TEMPLATE
    assert recorded["system_message"] == RERANKER_SYSTEM_MESSAGE


def test_the_fixture_used_the_shipped_matching_criterion(recorded):
    assert recorded["criterion_name"] == DEFAULT_MATCHING_CRITERION.name
    assert recorded["positive_instruction"] == DEFAULT_MATCHING_CRITERION.positive_instruction
    assert recorded["negative_instruction"] == DEFAULT_MATCHING_CRITERION.negative_instruction


def test_the_scores_are_not_all_one_constant(recorded):
    """The June regression, stated as the test nobody had.

    A blanked prompt gives every pair the same score. A fixture that could not
    tell that apart would be worth nothing.
    """
    scores = [entry["score"] for entry in recorded["raw_scores"]]
    assert len(scores) == recorded["n_prompts"] == 60
    assert len(set(scores)) > len(scores) // 2
    assert np.std(scores) > 0.01


def _stub_reranker():
    """A LoadedReranker with no model in it; score_prompts is stubbed."""
    from cruxes.reranker.model import LoadedReranker, RERANKER_MODEL_NAME, RERANKER_REVISION

    return LoadedReranker(None, None, None, "cpu", RERANKER_MODEL_NAME, RERANKER_REVISION)


def _scores_in_prompt_order(recorded: dict) -> list:
    from cruxes.reranker.prompt import pair_prompt_order

    n = len(recorded["introspected"]) + len(recorded["extracted"])
    by_slot = {(e["i"], e["j"], e["direction"]): e["score"] for e in recorded["raw_scores"]}
    return [by_slot[slot] for slot in pair_prompt_order(n)]


def test_score_pair_matrices_assembles_the_recorded_matrices(recorded, monkeypatch):
    """The assembly primitive itself, run on a CPU with the scorer stubbed.

    The stub hands back the fixture's sixty scores in the order the package
    renders its prompts, so what is tested is exactly which direction of a pair
    lands in which triangle of which matrix. The test calls the assembly
    itself rather than re-implementing it.
    """
    from cruxes.reranker import pair_scores

    items = list(recorded["introspected"]) + list(recorded["extracted"])
    expected_prompts = 4 * len(items) * (len(items) - 1) // 2

    def fake_score_prompts(reranker, prompts, batch_size, report_every):
        assert len(prompts) == expected_prompts
        assert batch_size == recorded["batch_size"]
        return _scores_in_prompt_order(recorded)

    monkeypatch.setattr(pair_scores, "score_prompts", fake_score_prompts)
    built = pair_scores.score_pair_matrices(
        _stub_reranker(), items, DEFAULT_MATCHING_CRITERION, batch_size=recorded["batch_size"]
    )
    for name in SCORE_KEYS:
        assert np.array_equal(built[name], np.array(recorded[name])), name


def test_a_swapped_direction_would_be_caught(recorded, monkeypatch):
    """The scores reversed pairwise land in the other triangle, and the test above sees it."""
    from cruxes.reranker import pair_scores

    items = list(recorded["introspected"]) + list(recorded["extracted"])
    ordered = _scores_in_prompt_order(recorded)
    swapped = [ordered[i ^ 1] for i in range(len(ordered))]  # forward <-> reverse of each pair
    monkeypatch.setattr(pair_scores, "score_prompts", lambda *a, **k: swapped)
    built = pair_scores.score_pair_matrices(_stub_reranker(), items, DEFAULT_MATCHING_CRITERION)
    assert not np.array_equal(built["pos_fwd"], np.array(recorded["pos_fwd"]))
    assert np.array_equal(built["pos_fwd"], np.array(recorded["pos_fwd"]).T)


def test_score_cache_key_refuses_a_model_that_is_not_the_keys(recorded):
    from cruxes.reranker.pair_scores import score_cache_key
    from cruxes.scorecache import build_score_cache_key

    key = build_score_cache_key(list(recorded["introspected"]) + list(recorded["extracted"]))
    wrong = _stub_reranker()
    wrong.model_revision = "0000000000000000000000000000000000000000"
    with pytest.raises(ValueError, match="does not match the cache key"):
        score_cache_key(wrong, key)


def test_the_reverse_matrices_are_the_transposes(recorded):
    assert np.array_equal(np.array(recorded["pos_rev"]), np.array(recorded["pos_fwd"]).T)
    assert np.array_equal(np.array(recorded["neg_rev"]), np.array(recorded["neg_fwd"]).T)


def test_the_diagonals_are_the_unscored_convention(recorded):
    assert np.array_equal(np.diag(np.array(recorded["pos_fwd"])), np.ones(6))
    assert np.array_equal(np.diag(np.array(recorded["neg_fwd"])), np.zeros(6))


def test_the_two_directions_of_a_pair_disagree_somewhere(recorded):
    """Order matters to this model, which is why every pair is asked twice."""
    pos_fwd = np.array(recorded["pos_fwd"])
    assert not np.allclose(pos_fwd, pos_fwd.T), "forward and reverse never differ"


def test_pos_only_aggregation_reproduces_the_recorded_similarity(recorded, matrices):
    """aggregate_scores is the pos_only aggregation; it is the only one that ships."""
    computed = aggregate_scores(matrices)
    assert np.allclose(computed, np.array(recorded["aggregated"]), atol=0, rtol=0)


def test_the_aggregated_similarity_is_symmetric(recorded):
    similarity = np.array(recorded["aggregated"])
    assert np.allclose(similarity, similarity.T)


def test_the_scorer_separates_the_matching_pair_from_the_rest(recorded):
    """A sanity check on the model, not on the code.

    The first premise of each pool asks the same factual question in
    different words, so under the positive instruction that cross pair should
    score above the median cross pair. If this fails the fixture is not
    measuring what the method assumes it measures.
    """
    similarity = np.array(recorded["aggregated"])
    n_introspected = len(recorded["introspected"])
    cross = similarity[:n_introspected, n_introspected:]
    assert cross[0, 0] > np.median(cross)
