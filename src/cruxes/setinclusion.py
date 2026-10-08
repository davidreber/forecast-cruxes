"""The pipeline: the set inclusion measurement, from score matrices to the table.

This is the package's one pipeline over its primitives (select, score,
aggregate, cluster, decompose, permute). Every main body number of the paper
runs through it with its defaults; anything else is a different value passed
to it, a different input, or a procedure a caller composes from the
primitives outside the package.

Two functions. :func:`set_inclusion_curve` takes one question's score matrices
and returns the Venn decomposition at every contrast count, which is the whole
measurement for that question and involves no file access.
:func:`summarize_curves` takes those per question results and produces the
nested dictionary a report holds, with the mean, the sample standard deviation
and the standard error at each count.

Output keys use the package's names throughout: ``per_contrasts_per_side``,
``contrasts_per_side``, ``per_question``, ``n_questions``, ``shared_frac``,
``introspected_only_frac``, ``extracted_only_frac``.
"""

from __future__ import annotations

import numpy as np

from .clustering import consensus_clustering
from .permutation import DEFAULT_N_PERMUTATIONS, default_permutation_seed, permutation_null
from .scoring import ScoreMatrices, aggregate_scores
from .stats import mean_and_error
from .venn import introspected_ids, venn_decomposition

__all__ = ["set_inclusion_curve", "summarize_curves"]

_FRACTIONS = ("shared", "introspected_only", "extracted_only")


def set_inclusion_curve(
    matrices: ScoreMatrices,
    max_contrasts_per_side: int,
    question_id: str,
    aggregate=aggregate_scores,
    with_permutation: bool = True,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    permutation_seed=default_permutation_seed,
) -> dict:
    """Venn decomposition at every contrast count from one to the maximum.

    At each count the first that many items of each pool are taken, which is a
    prefix of the selection made before scoring, so the curve is nested. A
    count that leaves fewer than two items in total is skipped, since there is
    nothing to cluster.

    ``aggregate`` and ``permutation_seed`` are the two research choices this
    function makes; pass your own to replace the package's.

    Returns ``{contrasts_per_side: record}`` where the record holds the Venn
    counts and fractions and, when asked for, the permutation null under the
    key ``permutation``.
    """
    similarity = np.asarray(aggregate(matrices), dtype=float)
    ids = matrices.item_ids
    n_introspected = matrices.n_introspected
    n_extracted = matrices.n_extracted

    curve = {}
    for count in range(1, max_contrasts_per_side + 1):
        take_introspected = min(count, n_introspected)
        take_extracted = min(count, n_extracted)
        indices = list(range(take_introspected)) + list(
            range(n_introspected, n_introspected + take_extracted)
        )
        if len(indices) < 2:
            continue

        sub_ids = [ids[i] for i in indices]
        clusters, _ = consensus_clustering(similarity[np.ix_(indices, indices)], sub_ids)
        introspected = introspected_ids(sub_ids)

        record = venn_decomposition(clusters, introspected)
        if with_permutation:
            record["permutation"] = permutation_null(
                clusters,
                introspected,
                seed=permutation_seed(question_id, count),
                n_permutations=n_permutations,
            )
        curve[count] = record
    return curve


def summarize_curves(
    curves: dict,
    max_contrasts_per_side: int,
    metadata: dict | None = None,
    with_permutation: bool = True,
) -> dict:
    """Aggregate per question curves into the summary table.

    ``curves`` maps a question id to the output of :func:`set_inclusion_curve`.
    Questions are visited in the order the mapping gives, so a caller that
    wants a stable file passes a sorted mapping. A count at which no question
    has a record is left out rather than written as an empty entry.
    """
    output = {"metadata": dict(metadata or {}), "per_contrasts_per_side": {}}
    output["metadata"].setdefault("n_questions", len(curves))

    for count in range(1, max_contrasts_per_side + 1):
        per_question = {}
        values = {key: [] for key in _FRACTIONS}
        null_shared = []

        for question_id, curve in curves.items():
            if count not in curve:
                continue
            record = curve[count]
            per_question[question_id] = record
            for key in _FRACTIONS:
                values[key].append(record[f"{key}_frac"])
            if with_permutation and "permutation" in record:
                null_shared.append(record["permutation"]["permutation_shared_mean"])

        if not values["shared"]:
            continue

        summary = {}
        n = 0
        for key in _FRACTIONS:
            mean, std, se, n = mean_and_error(values[key])
            summary[f"{key}_frac_mean"] = mean
            summary[f"{key}_frac_std"] = std
            summary[f"{key}_frac_se"] = se
        summary["n_questions"] = n

        entry = {
            "contrasts_per_side": count,
            "per_question": per_question,
            "summary": summary,
        }
        if with_permutation and null_shared:
            mean, std, se, n_null = mean_and_error(null_shared)
            entry["permutation"] = {
                "permutation_shared_mean": mean,
                "permutation_shared_std": std,
                "permutation_shared_se": se,
                "n_questions": n_null,
            }
        output["per_contrasts_per_side"][str(count)] = entry

    return output
