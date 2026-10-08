"""Venn decomposition of a clustering of the two pools.

The introspected and extracted premises of a question are pooled, clustered
together, and each cluster is then asked which pools it draws from. A cluster
holding members of both is a premise both processes arrived at. A cluster
holding members of one is a premise only that process produced.

The counting unit is the cluster, not the item, so a cluster of eight items
counts once. That is deliberate: the question is how many distinct things each
process talked about, and an item count would let a verbose process dominate.
"""

from __future__ import annotations

from .premises import INTROSPECTED

__all__ = [
    "INTROSPECTED_PREFIX",
    "introspected_ids",
    "venn_counts",
    "venn_decomposition",
]

#: Items from the introspected pool carry this prefix in their id.
INTROSPECTED_PREFIX = f"{INTROSPECTED}:"


def introspected_ids(ids) -> set:
    """The subset of ids belonging to the introspected pool, by prefix."""
    return {item for item in ids if str(item).startswith(INTROSPECTED_PREFIX)}


def venn_counts(clusters: list, introspected: set) -> tuple:
    """Count ``(shared, introspected_only, extracted_only)`` clusters.

    A cluster is shared when it holds at least one id in ``introspected`` and
    at least one id outside it. Membership of the extracted pool is defined by
    absence from ``introspected``, not by a prefix, because the permutation
    null reassigns the introspected label to an arbitrary subset of the ids.
    """
    shared = introspected_only = extracted_only = 0
    for cluster in clusters:
        has_introspected = any(item in introspected for item in cluster)
        has_extracted = any(item not in introspected for item in cluster)
        if has_introspected and has_extracted:
            shared += 1
        elif has_introspected:
            introspected_only += 1
        else:
            extracted_only += 1
    return shared, introspected_only, extracted_only


def venn_decomposition(clusters: list, introspected: set) -> dict:
    """Cluster counts and fractions for one clustering.

    Returns the three counts, their total, and each as a fraction of the total.
    An empty clustering returns zeros rather than raising, because a question
    with a single premise overall legitimately produces no clusters.
    """
    shared, introspected_only, extracted_only = venn_counts(clusters, introspected)
    total = shared + introspected_only + extracted_only
    return {
        "shared": shared,
        "introspected_only": introspected_only,
        "extracted_only": extracted_only,
        "total": total,
        "shared_frac": shared / total if total else 0.0,
        "introspected_only_frac": introspected_only / total if total else 0.0,
        "extracted_only_frac": extracted_only / total if total else 0.0,
    }
