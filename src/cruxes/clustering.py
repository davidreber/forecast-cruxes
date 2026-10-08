"""Average linkage clustering of a similarity matrix.

The pipeline's cut picks the number of clusters whose lifetime in the merge
sequence is longest, a parameter free choice. It is built from three
primitives, each usable on its own: :func:`linkage_from_similarity`,
:func:`optimal_cluster_count` and :func:`clusters_at_resolution`, which cuts at
a given count. The pipeline passes through the last with the count the first
two choose, which is why a procedure that wants a different count can call it
directly.

A similarity that is not a finite number is an error, not a zero. The research
code filled missing entries with zero, maximum dissimilarity, silently; on the
paper's data that fill never fired, and here it cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

__all__ = [
    "ClusteringDiagnostics",
    "linkage_from_similarity",
    "optimal_cluster_count",
    "clusters_at_resolution",
    "consensus_clustering",
]


@dataclass(frozen=True)
class ClusteringDiagnostics:
    """What the consensus cut chose, for a caller that wants to log it."""

    n_items: int
    n_clusters: int
    max_lifetime: float


def _prepare(similarity: np.ndarray, items: list) -> np.ndarray:
    """Copy, check and set the diagonal to one."""
    prepared = np.array(similarity, dtype=float, copy=True)
    n_items = len(items)
    if prepared.shape != (n_items, n_items):
        raise ValueError(
            f"similarity has shape {prepared.shape} for {n_items} items"
        )
    bad = np.argwhere(~np.isfinite(prepared))
    if len(bad):
        i, j = (int(x) for x in bad[0])
        raise ValueError(
            f"similarity[{i}, {j}] between {items[i]!r} and {items[j]!r} is not a "
            f"finite number ({len(bad)} such entries). An unscored pair is an error."
        )
    if n_items >= 2:
        np.fill_diagonal(prepared, 1.0)
    return prepared


def linkage_from_similarity(similarity: np.ndarray, items: list) -> tuple:
    """Build an average linkage dendrogram from a similarity matrix.

    Returns ``(prepared_similarity, linkage)``. The similarity is converted to
    a distance with ``1 - clip(similarity, 0, 1)`` before linkage. With fewer
    than two items there is nothing to merge and the linkage is ``None``.
    """
    prepared = _prepare(similarity, items)
    if len(items) < 2:
        return prepared, None
    distances = ssd.squareform(1 - np.clip(prepared, 0, 1))
    return prepared, sch.linkage(distances, method="average")


def _lifetime(merge_heights: np.ndarray, n_items: int, k: int) -> float:
    """How long the cut at ``k`` clusters survives in the merge sequence.

    The all singletons cut is given the first merge height, since nothing
    precedes it.
    """
    if k == n_items:
        return merge_heights[0] if len(merge_heights) > 0 else 1.0
    return merge_heights[n_items - k] - merge_heights[n_items - 1 - k]


def optimal_cluster_count(merge_heights: np.ndarray, n_items: int) -> int:
    """Number of clusters whose lifetime in the merge sequence is longest.

    A long lifetime means the structure at that granularity survives a wide
    range of thresholds, so it is the least arbitrary place to cut. Ties go to
    the smallest count.
    """
    best_lifetime = -1.0
    best_k = 1
    for k in range(2, n_items + 1):
        lifetime = _lifetime(merge_heights, n_items, k)
        if lifetime > best_lifetime:
            best_lifetime = lifetime
            best_k = k
    return best_k


def _labels_to_clusters(labels: np.ndarray, items: list) -> list:
    clusters = []
    for label in sorted(set(labels)):
        indices = np.where(labels == label)[0]
        clusters.append([items[i] for i in indices])
    return clusters


def clusters_at_resolution(linkage, items: list, n_clusters: int) -> list:
    """Cut a dendrogram at a chosen number of clusters.

    The target is clamped to the range one to the number of items. With no
    linkage, which happens below two items, every item forms one cluster
    together, or the result is empty when there are no items at all.
    """
    n_items = len(items)
    if linkage is None or n_items < 2:
        return [list(items)] if n_items else []
    target = max(1, min(n_items, int(n_clusters)))
    labels = sch.fcluster(linkage, t=target, criterion="maxclust")
    return _labels_to_clusters(labels, items)


def consensus_clustering(similarity: np.ndarray, items: list) -> tuple:
    """Cluster at the longest lived cut of the average linkage dendrogram.

    Returns ``(clusters, diagnostics)``. ``clusters`` is a list of lists of the
    item labels, partitioning ``items``. Below two items the partition is
    empty: a single item yields no clusters rather than one singleton.
    """
    n_items = len(items)
    prepared, linkage = linkage_from_similarity(similarity, items)
    if linkage is None:
        return [], ClusteringDiagnostics(n_items, 0, 0.0)

    heights = linkage[:, 2]
    best_k = optimal_cluster_count(heights, n_items)
    max_lifetime = max(_lifetime(heights, n_items, k) for k in range(2, n_items + 1))
    clusters = clusters_at_resolution(linkage, items, best_k)
    return clusters, ClusteringDiagnostics(
        n_items=n_items, n_clusters=len(clusters), max_lifetime=float(max_lifetime)
    )
