"""The label shuffling null and the set inclusion measurement built on it.

New coverage for the restructure: permutation_null now takes an explicit int
seed with no default, default_permutation_seed's formula, and
set_inclusion_curve/summarize_curves's new output keys
(per_contrasts_per_side, not per_k; no per_matchup, no n_matchups).
"""

import numpy as np
import pytest

from cruxes.permutation import default_permutation_seed, permutation_null
from cruxes.scoring import ScoreMatrices
from cruxes.selection import derive_seed
from cruxes.setinclusion import set_inclusion_curve, summarize_curves


# -- default_permutation_seed -------------------------------------------


def test_default_permutation_seed_matches_derive_seed_for_set_inclusion():
    assert default_permutation_seed("q", 25) == derive_seed(42, "q", 25)


def test_default_permutation_seed_respects_base_seed():
    assert default_permutation_seed("q", 25, base_seed=1) == derive_seed(1, "q", 25)


# -- permutation_null ------------------------------------------------------


def test_permutation_null_has_no_default_seed():
    import inspect

    assert inspect.signature(permutation_null).parameters["seed"].default is inspect.Parameter.empty


def test_permutation_null_empty_clusters_returns_zeros():
    result = permutation_null([], set(), seed=1)
    assert all(value == 0.0 for value in result.values())


def test_permutation_null_is_deterministic_given_a_seed():
    clusters = [["introspected:0", "extracted:0"], ["introspected:1"], ["extracted:1"]]
    introspected = {"introspected:0", "introspected:1"}
    first = permutation_null(clusters, introspected, seed=7, n_permutations=100)
    second = permutation_null(clusters, introspected, seed=7, n_permutations=100)
    assert first == second


def test_permutation_null_differs_with_a_different_seed():
    clusters = [["introspected:0", "extracted:0"], ["introspected:1"], ["extracted:1"]]
    introspected = {"introspected:0", "introspected:1"}
    first = permutation_null(clusters, introspected, seed=7, n_permutations=200)
    second = permutation_null(clusters, introspected, seed=8, n_permutations=200)
    assert first != second


# -- set_inclusion_curve ---------------------------------------------------


def make_cross_pool_matrices():
    pos = np.array(
        [
            [1.00, 0.05, 0.90, 0.05],
            [0.05, 1.00, 0.05, 0.90],
            [0.90, 0.05, 1.00, 0.05],
            [0.05, 0.90, 0.05, 1.00],
        ]
    )
    return ScoreMatrices(
        pos_fwd=pos, pos_rev=pos.T, neg_fwd=np.zeros((4, 4)), neg_rev=np.zeros((4, 4)),
        n_introspected=2, n_extracted=2,
    )


def test_set_inclusion_curve_uses_the_given_aggregate():
    matrices = make_cross_pool_matrices()

    def all_similar(_matrices):
        return np.ones((4, 4))

    def all_distinct(_matrices):
        return np.eye(4)

    similar_curve = set_inclusion_curve(
        matrices, 2, "q", aggregate=all_similar, with_permutation=False
    )
    distinct_curve = set_inclusion_curve(
        matrices, 2, "q", aggregate=all_distinct, with_permutation=False
    )
    assert similar_curve[2]["shared_frac"] == pytest.approx(1.0)
    assert distinct_curve[2]["shared_frac"] == pytest.approx(0.0)


def test_set_inclusion_curve_uses_the_given_permutation_seed():
    """A custom permutation_seed callable is called and changes the null."""
    matrices = make_cross_pool_matrices()

    def seed_a(question_id, count):
        return 1

    def seed_b(question_id, count):
        return 2

    curve_a = set_inclusion_curve(matrices, 2, "q", n_permutations=300, permutation_seed=seed_a)
    curve_b = set_inclusion_curve(matrices, 2, "q", n_permutations=300, permutation_seed=seed_b)
    assert curve_a[2]["permutation"] != curve_b[2]["permutation"]


def test_set_inclusion_curve_different_question_ids_get_different_default_seeds():
    matrices = make_cross_pool_matrices()
    curve_one = set_inclusion_curve(matrices, 2, "question-one", n_permutations=300)
    curve_two = set_inclusion_curve(matrices, 2, "question-two", n_permutations=300)
    assert curve_one[2]["permutation"] != curve_two[2]["permutation"]


def test_set_inclusion_curve_skips_counts_with_fewer_than_two_items():
    matrices = ScoreMatrices(
        pos_fwd=np.eye(1), pos_rev=np.eye(1), neg_fwd=np.zeros((1, 1)), neg_rev=np.zeros((1, 1)),
        n_introspected=0, n_extracted=1,
    )
    curve = set_inclusion_curve(matrices, 3, "q", with_permutation=False)
    assert curve == {}


# -- summarize_curves --------------------------------------------------------


def test_summarize_curves_uses_the_new_key_names():
    matrices = make_cross_pool_matrices()
    curves = {"q": set_inclusion_curve(matrices, 2, "q", n_permutations=50)}
    summary = summarize_curves(curves, 2)

    assert set(summary) == {"metadata", "per_contrasts_per_side"}
    assert summary["metadata"]["n_questions"] == 1
    entry = summary["per_contrasts_per_side"]["2"]
    assert entry["contrasts_per_side"] == 2
    assert set(entry["per_question"]) == {"q"}
    assert entry["summary"]["n_questions"] == 1
    assert entry["permutation"]["n_questions"] == 1

    assert "per_k" not in summary
    assert "k" not in entry
    assert "per_matchup" not in entry
    assert "n_matchups" not in entry["summary"]
