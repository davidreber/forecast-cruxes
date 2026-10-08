"""Seeded selection and score aggregation.

The prefix monotonicity and seed stability tests are ported from the previous
test_selection_and_scoring.py for cruxes.selection. The aggregation tests are
rewritten for the single shipped aggregation, aggregate_scores, now that
AGGREGATION_METHODS and the method= argument are gone; a caller wanting
another aggregation passes their own callable, which set_inclusion_curve
accepts as aggregate=.
"""

import numpy as np
import pytest

from cruxes.scoring import (
    DEFAULT_AGGREGATION_NAME,
    SCORE_KEYS,
    ScoreMatrices,
    aggregate_scores,
    aggregation_name,
    item_ids,
)
from cruxes.selection import derive_seed, select_prefix

# -- selection -----------------------------------------------------------


def test_prefix_is_nested():
    """The selection at five is a strict prefix of the selection at ten."""
    items = [f"statement {i}" for i in range(20)]
    assert select_prefix(items, 5, seed=42) == select_prefix(items, 10, seed=42)[:5]


def test_prefix_is_a_permutation_when_count_covers_everything():
    items = [f"statement {i}" for i in range(7)]
    assert sorted(select_prefix(items, 99, seed=1)) == sorted(items)


def test_different_seeds_give_different_orders():
    items = [f"statement {i}" for i in range(20)]
    assert select_prefix(items, 6, 1) != select_prefix(items, 6, 2)


# -- seeding ---------------------------------------------------------------


def test_derive_seed_is_stable_across_processes():
    """MD5, not the salted builtin hash, so the value is fixed for all time."""
    assert derive_seed(42, "duke-vs-siena", "apriori") == derive_seed(42, "duke-vs-siena", "apriori")
    assert derive_seed(42, "duke-vs-siena", "apriori") == 437635574


def test_derive_seed_is_a_valid_numpy_seed():
    for question in ("a", "bb", "duke-vs-siena"):
        seed = derive_seed(42, question, 25)
        assert 0 <= seed < 2**31


# -- score matrices ----------------------------------------------------------


@pytest.fixture()
def raw_scores():
    """Three items with deliberately asymmetric forward and reverse scores."""
    pos_fwd = np.array(
        [
            [1.0, 0.8, 0.2],
            [0.6, 1.0, 0.4],
            [0.1, 0.3, 1.0],
        ]
    )
    pos_rev = pos_fwd.T.copy()
    neg_fwd = np.array(
        [
            [0.0, 0.1, 0.7],
            [0.3, 0.0, 0.5],
            [0.9, 0.6, 0.0],
        ]
    )
    neg_rev = neg_fwd.T.copy()
    return ScoreMatrices(
        pos_fwd=pos_fwd, pos_rev=pos_rev, neg_fwd=neg_fwd, neg_rev=neg_rev,
        n_introspected=2, n_extracted=1,
    )


def test_item_ids_prefixes_each_pool():
    assert item_ids(2, 1) == ["introspected:0", "introspected:1", "extracted:0"]


def test_score_matrices_expose_item_ids_and_counts(raw_scores):
    assert raw_scores.n_items == 3
    assert raw_scores.item_ids == ["introspected:0", "introspected:1", "extracted:0"]


def test_score_matrices_reject_the_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        ScoreMatrices(
            pos_fwd=np.eye(2), pos_rev=np.eye(2), neg_fwd=np.eye(2), neg_rev=np.eye(2),
            n_introspected=2, n_extracted=1,
        )


def test_score_matrices_reject_a_non_finite_entry():
    n = 3
    pos_fwd = np.eye(n)
    pos_fwd[0, 2] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        ScoreMatrices(
            pos_fwd=pos_fwd, pos_rev=np.eye(n), neg_fwd=np.zeros((n, n)), neg_rev=np.zeros((n, n)),
            n_introspected=2, n_extracted=1,
        )


def test_score_matrices_from_mapping_uses_the_pool_counts():
    n = 2
    data = {
        "pos_fwd": np.eye(n).tolist(),
        "pos_rev": np.eye(n).tolist(),
        "neg_fwd": np.zeros((n, n)).tolist(),
        "neg_rev": np.zeros((n, n)).tolist(),
        "n_introspected": 1,
        "n_extracted": 1,
    }
    matrices = ScoreMatrices.from_mapping(data)
    assert matrices.n_introspected == 1
    assert matrices.n_extracted == 1
    assert matrices.n_items == 2


# -- aggregation ---------------------------------------------------------


def test_aggregate_scores_is_symmetric_and_hand_checkable(raw_scores):
    """The default aggregation averages the directions and then symmetrizes."""
    result = aggregate_scores(raw_scores)
    assert np.allclose(result, result.T)
    assert result[0, 1] == pytest.approx((0.8 + 0.6) / 2)


def test_aggregate_scores_returns_a_square_matrix(raw_scores):
    result = aggregate_scores(raw_scores)
    assert result.shape == (3, 3)


def test_aggregation_name_for_the_default():
    assert aggregation_name(aggregate_scores) == DEFAULT_AGGREGATION_NAME


def test_aggregation_name_for_a_custom_callable():
    def my_aggregation(matrices):
        return matrices.pos_fwd

    assert aggregation_name(my_aggregation) == "custom:my_aggregation"


def test_score_keys_are_the_four_raw_matrices():
    assert SCORE_KEYS == ("pos_fwd", "pos_rev", "neg_fwd", "neg_rev")
