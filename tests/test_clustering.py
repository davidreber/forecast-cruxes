"""Clustering tests, ported from crux/tests/test_core.py.

The twelve tests that cover the consensus cut and the cluster count choice move
across unchanged in substance. Names and call sites follow the new API.
"""

import numpy as np
import pytest

from cruxes import (
    clusters_at_resolution,
    consensus_clustering,
    linkage_from_similarity,
    optimal_cluster_count,
)


# -- choosing where to cut ---------------------------------------------------

def test_optimal_cluster_count_obvious_gap():
    """A large jump in merge heights signals a natural cluster boundary."""
    heights = np.array([0.01, 0.02, 0.03, 0.9])
    assert optimal_cluster_count(heights, n_items=5) == 2


def test_optimal_cluster_count_single_item():
    """One item cannot be split, so the count must be one."""
    assert optimal_cluster_count(np.array([]), n_items=1) == 1


def test_optimal_cluster_count_returns_int():
    """Downstream code indexes with the count, so it must be a plain int."""
    assert isinstance(optimal_cluster_count(np.array([0.1, 0.5, 0.8]), n_items=4), int)


def test_optimal_cluster_count_in_valid_range():
    """The chosen count lies between one cluster and all singletons."""
    count = optimal_cluster_count(np.array([0.1, 0.3, 0.5, 0.9]), n_items=5)
    assert 1 <= count <= 5


# -- the consensus cut -------------------------------------------------------

def test_consensus_clustering_completeness():
    """Every item appears in some cluster; a lost item is a lost premise."""
    rng = np.random.default_rng(42)
    matrix = rng.uniform(0.2, 0.8, size=(6, 6))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1.0)
    items = [f"q{i}" for i in range(6)]
    clusters, _ = consensus_clustering(matrix, items)
    assert set().union(*clusters) == set(items)


def test_consensus_clustering_disjoint():
    """Clusters are disjoint; overlap would double count a premise."""
    rng = np.random.default_rng(99)
    matrix = rng.uniform(0.0, 1.0, size=(7, 7))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1.0)
    items = [f"crux_{i}" for i in range(7)]
    clusters, _ = consensus_clustering(matrix, items)
    seen = set()
    for cluster in clusters:
        assert seen.isdisjoint(cluster)
        seen.update(cluster)


def test_consensus_clustering_identity_gives_singletons():
    """Zero off diagonal similarity means nothing merges."""
    items = [f"s{i}" for i in range(5)]
    clusters, _ = consensus_clustering(np.eye(5), items)
    assert len(clusters) == 5
    assert all(len(c) == 1 for c in clusters)


def test_consensus_clustering_all_ones_gives_one_cluster():
    """Perfect similarity everywhere means everything belongs together."""
    items = [f"t{i}" for i in range(6)]
    clusters, _ = consensus_clustering(np.ones((6, 6)), items)
    assert len(clusters) == 1
    assert set(clusters[0]) == set(items)


def test_consensus_clustering_two_blocks():
    """A block diagonal matrix with two blocks gives exactly two clusters."""
    matrix = np.zeros((6, 6))
    matrix[:3, :3] = 1.0
    matrix[3:, 3:] = 1.0
    items = [f"item_{i}" for i in range(6)]
    clusters, _ = consensus_clustering(matrix, items)
    as_sets = [set(c) for c in clusters]
    assert len(clusters) == 2
    assert {f"item_{i}" for i in range(3)} in as_sets
    assert {f"item_{i}" for i in range(3, 6)} in as_sets


def test_consensus_clustering_one_item():
    """A single item cannot form a pair for linkage, so there are no clusters."""
    clusters, diagnostics = consensus_clustering(np.array([[1.0]]), ["solo"])
    assert clusters == []
    assert diagnostics.n_clusters == 0


def test_consensus_clustering_raises_on_a_missing_score():
    """A NaN or inf similarity is an error, not a value to fill in.

    An earlier version of the authors' code silently filled a missing score
    with zero, maximum dissimilarity; on the paper's data that fill never
    fired and here it cannot, so an unscored pair is caught rather than passed to the clustering.
    """
    matrix = np.array([
        [1.0, np.nan, 0.0],
        [np.nan, 1.0, 0.8],
        [0.0, 0.8, 1.0],
    ])
    items = ["a", "b", "c"]
    with pytest.raises(ValueError, match="not a finite number"):
        consensus_clustering(matrix, items)


def test_consensus_clustering_error_names_the_entry():
    matrix = np.array([
        [1.0, np.inf],
        [np.inf, 1.0],
    ])
    with pytest.raises(ValueError, match="crux_a.*crux_b|crux_b.*crux_a"):
        consensus_clustering(matrix, ["crux_a", "crux_b"])


def test_consensus_clustering_does_not_modify_input():
    """The caller's matrix is untouched."""
    matrix = np.array([
        [1.00, 0.30, 0.00],
        [0.30, 1.00, 0.80],
        [0.00, 0.80, 1.00],
    ])
    before = matrix.copy()
    consensus_clustering(matrix, ["a", "b", "c"])
    assert np.array_equal(matrix, before)


def test_clustering_diagnostics_has_no_n_filled_field():
    """n_filled belonged to the old missing-score fill and is gone."""
    import dataclasses

    from cruxes.clustering import ClusteringDiagnostics

    assert {f.name for f in dataclasses.fields(ClusteringDiagnostics)} == {
        "n_items",
        "n_clusters",
        "max_lifetime",
    }


# -- cutting at a chosen resolution ------------------------------------------

@pytest.fixture()
def cross_set_matrix():
    """Four item union with two cross set pairs, a:0 with b:0 and a:1 with b:1."""
    similarity = np.array([
        [1.00, 0.05, 0.90, 0.05],
        [0.05, 1.00, 0.05, 0.90],
        [0.90, 0.05, 1.00, 0.05],
        [0.05, 0.90, 0.05, 1.00],
    ])
    return similarity, ["a:0", "a:1", "b:0", "b:1"], {"a:0", "a:1"}


def test_linkage_below_two_items_is_none():
    """Fewer than two items yields no linkage."""
    assert linkage_from_similarity(np.zeros((0, 0)), [])[1] is None
    assert linkage_from_similarity(np.array([[1.0]]), ["a:0"])[1] is None


def test_clusters_at_resolution_coarsest(cross_set_matrix):
    """Cutting at one cluster returns a single cluster with every item."""
    similarity, ids, _ = cross_set_matrix
    _, linkage = linkage_from_similarity(similarity, ids)
    clusters = clusters_at_resolution(linkage, ids, 1)
    assert len(clusters) == 1
    assert set(clusters[0]) == set(ids)


def test_clusters_at_resolution_finest(cross_set_matrix):
    """Cutting at the item count returns all singletons."""
    similarity, ids, _ = cross_set_matrix
    _, linkage = linkage_from_similarity(similarity, ids)
    clusters = clusters_at_resolution(linkage, ids, len(ids))
    assert len(clusters) == len(ids)
    assert all(len(c) == 1 for c in clusters)


def test_clusters_at_resolution_degenerate():
    """With no linkage: one cluster if there are items, otherwise empty."""
    assert clusters_at_resolution(None, ["a:0"], 1) == [["a:0"]]
    assert clusters_at_resolution(None, [], 1) == []


def test_consensus_clustering_recovers_cross_pairs(cross_set_matrix):
    """The consensus cut still finds the two cross set pairs."""
    similarity, ids, _ = cross_set_matrix
    clusters, _ = consensus_clustering(similarity, ids)
    assert sorted(sorted(c) for c in clusters) == [["a:0", "b:0"], ["a:1", "b:1"]]
