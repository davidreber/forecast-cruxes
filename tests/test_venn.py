"""Venn decomposition tests, updated for the introspected/extracted vocabulary."""

import pytest

from cruxes import introspected_ids, venn_decomposition


def test_all_clusters_shared():
    """Every cluster draws from both pools, so the shared fraction is one."""
    clusters = [
        ["introspected:0", "extracted:0"],
        ["introspected:1", "extracted:1", "introspected:2"],
        ["introspected:3", "extracted:2", "extracted:3"],
    ]
    result = venn_decomposition(
        clusters, {"introspected:0", "introspected:1", "introspected:2", "introspected:3"}
    )
    assert result["shared"] == 3
    assert result["introspected_only"] == 0
    assert result["extracted_only"] == 0
    assert result["shared_frac"] == pytest.approx(1.0)


def test_no_cluster_shared():
    """No cluster is mixed, so the shared fraction is zero."""
    clusters = [
        ["introspected:0", "introspected:1"],
        ["extracted:0", "extracted:1"],
        ["introspected:2"],
        ["extracted:2"],
    ]
    result = venn_decomposition(clusters, {"introspected:0", "introspected:1", "introspected:2"})
    assert result["shared"] == 0
    assert result["introspected_only"] == 2
    assert result["extracted_only"] == 2
    assert result["introspected_only_frac"] == pytest.approx(0.5)
    assert result["extracted_only_frac"] == pytest.approx(0.5)


def test_fractions_sum_to_one():
    """The three fractions partition the clusters."""
    clusters = [
        ["introspected:0", "extracted:0"],
        ["introspected:1"],
        ["extracted:1", "extracted:2"],
        ["introspected:2", "extracted:3"],
    ]
    result = venn_decomposition(clusters, {"introspected:0", "introspected:1", "introspected:2"})
    assert (result["shared"], result["introspected_only"], result["extracted_only"]) == (2, 1, 1)
    assert result["total"] == 4
    total = result["shared_frac"] + result["introspected_only_frac"] + result["extracted_only_frac"]
    assert total == pytest.approx(1.0)


def test_empty_clustering_gives_zeros():
    """No clusters is a legitimate state and returns zeros rather than raising."""
    result = venn_decomposition([], set())
    assert result["total"] == 0
    assert result["shared_frac"] == 0.0


def test_singletons_are_categorised_by_pool():
    """A singleton can never be shared, which is what depresses the null."""
    clusters = [
        ["introspected:0"],
        ["introspected:1"],
        ["extracted:0"],
        ["extracted:1"],
        ["extracted:2"],
    ]
    result = venn_decomposition(clusters, {"introspected:0", "introspected:1"})
    assert result["shared"] == 0
    assert result["introspected_only"] == 2
    assert result["extracted_only"] == 3
    assert result["introspected_only_frac"] == pytest.approx(2 / 5)
    assert result["extracted_only_frac"] == pytest.approx(3 / 5)


def test_introspected_ids_reads_the_prefix():
    """Pool membership is carried by the id prefix, which is a data contract."""
    assert introspected_ids(["introspected:0", "extracted:3", "introspected:11"]) == {
        "introspected:0",
        "introspected:11",
    }
