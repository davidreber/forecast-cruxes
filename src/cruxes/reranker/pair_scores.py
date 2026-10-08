"""Turning selected premises into the four raw score matrices.

Every unordered pair is asked four questions, so fifty premises cost four
thousand nine hundred forward passes. The layout is worth spelling out,
because it is easy to get backwards. For a pair (i, j) with i < j, write
``forward`` for the score with item i in the query slot and ``reverse`` for the
score with item j there. Then

    pos_fwd[i, j] = positive forward     pos_fwd[j, i] = positive reverse
    pos_rev[i, j] = positive reverse     pos_rev[j, i] = positive forward

and the same for the negative pair of matrices. So ``pos_rev`` is the transpose
of ``pos_fwd``. The diagonal is one for the positive matrices and zero for the
negative ones, never scored.
"""

from __future__ import annotations

import numpy as np

from ..matching import MatchingCriterion
from ..scorecache import ScoreCacheKey
from .model import DEFAULT_BATCH_SIZE, LoadedReranker, score_prompts
from .prompt import DEFAULT_RERANKER_PROMPT, RerankerPrompt, pair_prompt_order, render_pair_prompts

__all__ = ["score_pair_matrices", "score_cache_key"]


def score_pair_matrices(
    reranker: LoadedReranker,
    items: list,
    criterion: MatchingCriterion,
    prompt: RerankerPrompt = DEFAULT_RERANKER_PROMPT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    report_every: int = 0,
) -> dict:
    """Score every pair of ``items`` four ways and assemble the four matrices.

    Returns a dict with keys ``pos_fwd``, ``pos_rev``, ``neg_fwd`` and
    ``neg_rev``, each an N by N float array. Prompts run in the fixed order
    :func:`cruxes.reranker.prompt.pair_prompt_order` defines, in fixed size
    batches, so two runs over the same items give the same numbers.
    """
    n_items = len(items)
    matrices = {
        "pos_fwd": np.eye(n_items),
        "pos_rev": np.eye(n_items),
        "neg_fwd": np.zeros((n_items, n_items)),
        "neg_rev": np.zeros((n_items, n_items)),
    }
    if n_items < 2:
        return matrices

    prompts = render_pair_prompts(items, criterion, prompt)
    scores = score_prompts(reranker, prompts, batch_size=batch_size, report_every=report_every)
    if len(scores) != len(prompts):
        raise RuntimeError(f"scorer returned {len(scores)} scores for {len(prompts)} prompts")

    pos_fwd, pos_rev = matrices["pos_fwd"], matrices["pos_rev"]
    neg_fwd, neg_rev = matrices["neg_fwd"], matrices["neg_rev"]
    for (i, j, direction), score in zip(pair_prompt_order(n_items), scores):
        if direction == "positive_forward":
            pos_fwd[i, j] = score
            pos_rev[j, i] = score
        elif direction == "positive_reverse":
            pos_fwd[j, i] = score
            pos_rev[i, j] = score
        elif direction == "negative_forward":
            neg_fwd[i, j] = score
            neg_rev[j, i] = score
        else:
            neg_fwd[j, i] = score
            neg_rev[i, j] = score
    return matrices


def score_cache_key(reranker: LoadedReranker, key: ScoreCacheKey, report_every: int = 0) -> dict:
    """Score exactly what a cache key describes, refusing a mismatched model.

    The loaded model's name, revision, precision and truncation must equal the
    key's, so a cache can never carry a key that describes a different
    computation from the one that produced its numbers.
    """
    loaded = {
        "model_name": reranker.model_name,
        "model_revision": reranker.model_revision,
        "dtype": reranker.dtype,
        "max_length": reranker.max_length,
    }
    wanted = {name: getattr(key, name) for name in loaded}
    if loaded != wanted:
        raise ValueError(f"loaded reranker {loaded} does not match the cache key {wanted}")
    return score_pair_matrices(
        reranker,
        key.premises,
        MatchingCriterion("from_key", key.positive_instruction, key.negative_instruction),
        RerankerPrompt(key.system_message, key.prompt_template),
        batch_size=key.batch_size,
        report_every=report_every,
    )
