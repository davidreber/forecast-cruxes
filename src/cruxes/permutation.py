"""Label shuffling null for the Venn decomposition, and how it is seeded.

The clustering is held fixed and the introspected label is reassigned at random
to the same number of items, then the Venn decomposition is recomputed.
Repeating that gives the overlap a clustering of this shape would show if the
two pools were drawn from the same source.

The null is much lower than a naive guess, because a singleton cluster can
never be shared however the labels fall. When the clustering is mostly
singletons the null approaches zero on its own, and an observed overlap near
zero then means very little. That is why the null is reported beside every
observed number.

Seeding is a research choice, so it is one default a caller can replace. The
default derives a seed per question and per count from a stable hash, so the
draws are independent across questions and reproducible across runs. The
pipeline takes ``permutation_seed``, a function ``(question_id,
contrasts_per_side) -> int``, to replace it; :func:`permutation_null` itself
takes a plain seed, so a procedure composed from the primitives seeds it
however it likes.
"""

from __future__ import annotations

import numpy as np

from .premises import DEFAULT_SEED
from .selection import derive_seed
from .venn import venn_counts

__all__ = [
    "DEFAULT_N_PERMUTATIONS",
    "default_permutation_seed",
    "permutation_null",
]

#: Draw count the paper used.
DEFAULT_N_PERMUTATIONS = 1000

def default_permutation_seed(
    question_id: str, contrasts_per_side: int, base_seed: int = DEFAULT_SEED
) -> int:
    """The pipeline's seed for one null: ``derive_seed(base_seed, question_id, contrasts_per_side)``."""
    return derive_seed(base_seed, question_id, contrasts_per_side)


def permutation_null(
    clusters: list,
    introspected: set,
    seed: int,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
) -> dict:
    """Null distribution of the Venn fractions under random pool labels.

    The number of introspected labels is held at the number of ids in
    ``introspected`` that appear in the clustering, so the null matches the
    observed split. Returns the mean and the sample standard deviation of each
    fraction over the draws.
    """
    all_ids = [item for cluster in clusters for item in cluster]
    keys = ("shared", "introspected_only", "extracted_only")
    if not clusters:
        return {
            f"permutation_{key}_{stat}": 0.0 for key in keys for stat in ("mean", "std")
        }

    n_introspected = len(set(introspected) & set(all_ids))
    rng = np.random.RandomState(seed)
    # One contiguous array per fraction, so the mean and deviation are the same
    # floating point sums on every platform numpy supports.
    fractions = [np.empty(n_permutations) for _ in keys]
    for i in range(n_permutations):
        shuffled = all_ids.copy()
        rng.shuffle(shuffled)
        counts = venn_counts(clusters, set(shuffled[:n_introspected]))
        total = sum(counts)
        for column, count in enumerate(counts):
            fractions[column][i] = count / total if total else 0.0

    result = {}
    for column, key in enumerate(keys):
        result[f"permutation_{key}_mean"] = float(fractions[column].mean())
        result[f"permutation_{key}_std"] = float(fractions[column].std(ddof=1))
    return result
