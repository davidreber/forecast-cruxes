"""Raw pairwise scores and their aggregation into one similarity matrix.

The cross encoder is asked four questions about every unordered pair of
premises, a positive and a negative instruction in each direction, giving four
N by N raw score matrices. This module holds the container for them and the
one aggregation the method uses.

The aggregation is a research choice, so it is one default that a caller can
replace: every function that aggregates takes ``aggregate``, a function from
:class:`ScoreMatrices` to a square symmetric array. The default,
:func:`aggregate_scores`, averages the positive scores over both directions and
symmetrizes, which is what the paper used (it was called ``pos_only`` there).
No alternative ships.

A missing score is an error here, not a value. The container refuses a non
finite entry anywhere in the four matrices, so an unscored pair cannot travel
on into the clustering disguised as a number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .premises import EXTRACTED, INTROSPECTED

__all__ = [
    "SCORE_KEYS",
    "DEFAULT_AGGREGATION_NAME",
    "ScoreMatrices",
    "aggregate_scores",
    "item_ids",
    "aggregation_name",
]

#: The four raw score matrices, in the order a cache stores them.
SCORE_KEYS = ("pos_fwd", "pos_rev", "neg_fwd", "neg_rev")

#: The name reports record for the default aggregation.
DEFAULT_AGGREGATION_NAME = "pos_only"


def item_ids(n_introspected: int, n_extracted: int) -> list:
    """Row and column labels: ``introspected:0 ...`` then ``extracted:0 ...``.

    The prefix is what the Venn decomposition reads to tell the pools apart,
    so it is part of the data contract rather than a display detail.
    """
    return [f"{INTROSPECTED}:{i}" for i in range(n_introspected)] + [
        f"{EXTRACTED}:{j}" for j in range(n_extracted)
    ]


@dataclass(frozen=True)
class ScoreMatrices:
    """Four raw N by N score matrices for one question's pooled premises.

    Rows and columns are the ``n_introspected`` selected introspected premises
    followed by the ``n_extracted`` selected extracted ones, in selection
    order, labelled by :func:`item_ids`.
    """

    pos_fwd: np.ndarray
    pos_rev: np.ndarray
    neg_fwd: np.ndarray
    neg_rev: np.ndarray
    n_introspected: int
    n_extracted: int

    def __post_init__(self):
        n = self.n_introspected + self.n_extracted
        for name in SCORE_KEYS:
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != (n, n):
                raise ValueError(
                    f"{name} has shape {array.shape}, expected ({n}, {n}) from "
                    f"{self.n_introspected} introspected and {self.n_extracted} "
                    "extracted premises"
                )
            bad = np.argwhere(~np.isfinite(array))
            if len(bad):
                i, j = (int(x) for x in bad[0])
                raise ValueError(
                    f"{name}[{i}, {j}] is {array[i, j]!r} and {len(bad)} entries in "
                    "total are not finite. An unscored pair is an error: rescore "
                    "the question rather than guess its similarity."
                )
            object.__setattr__(self, name, array)

    @property
    def n_items(self) -> int:
        return self.n_introspected + self.n_extracted

    @property
    def item_ids(self) -> list:
        return item_ids(self.n_introspected, self.n_extracted)

    @classmethod
    def from_mapping(cls, data: dict) -> "ScoreMatrices":
        """Build from a plain dict holding the four matrices and the two counts."""
        return cls(
            n_introspected=int(data["n_introspected"]),
            n_extracted=int(data["n_extracted"]),
            **{key: data[key] for key in SCORE_KEYS},
        )

    @classmethod
    def from_json(cls, path) -> "ScoreMatrices":
        """Load one file holding the four matrices and the two counts. Read only."""
        with open(Path(path)) as handle:
            return cls.from_mapping(json.load(handle))


def aggregate_scores(matrices: ScoreMatrices) -> np.ndarray:
    """The default aggregation: positive scores, both directions, symmetrized.

    ``pos_fwd`` holds each pair's forward score above the diagonal and its
    reverse score below it, and ``pos_rev`` the transpose, so their mean holds
    the two directions' average on both sides. The final symmetrization is a
    no op on scores laid out that way and is kept because it guarantees a
    symmetric matrix for any input.
    """
    symmetric = (matrices.pos_fwd + matrices.pos_rev) / 2
    return (symmetric + symmetric.T) / 2


def aggregation_name(aggregate) -> str:
    """What a report records for an aggregation function."""
    if aggregate is aggregate_scores:
        return DEFAULT_AGGREGATION_NAME
    return f"custom:{getattr(aggregate, '__name__', repr(aggregate))}"
