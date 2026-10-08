"""Seeded selection of a nested prefix of premises, and the one seeding idiom.

Both pools of a question are subsampled to at most ``contrasts_per_side``
premises before scoring. The subsample is a prefix of a seeded permutation, so
the selection at five is always a prefix of the selection at ten and the curve
across counts is nested rather than a fresh draw at every count.

Every seed in the package comes from :func:`derive_seed`, a stable MD5 digest
rather than Python's builtin ``hash``, which is salted per process and would
give a different subsample on every run.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["derive_seed", "select_prefix"]


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable non-negative 31-bit seed from a base seed and key parts.

    The parts are joined with colons and hashed with MD5. The digest is added
    to the base seed and masked to 31 bits, so the result is a valid numpy seed
    and changing any part gives an unrelated stream.
    """
    key = ":".join(str(p) for p in parts).encode()
    digest = int(hashlib.md5(key).hexdigest(), 16)
    return (base_seed + digest) & 0x7FFFFFFF


def select_prefix(items: list, count: int, seed: int) -> list:
    """Return the first ``count`` items of a seeded permutation of ``items``.

    Nested in ``count``: the same seed gives the same permutation, so a
    smaller request is a strict prefix of a larger one. Requesting more items
    than the list holds returns the whole permuted list.
    """
    rng = np.random.default_rng(seed)
    indices = [int(i) for i in rng.permutation(len(items))[:count]]
    return [items[i] for i in indices]
