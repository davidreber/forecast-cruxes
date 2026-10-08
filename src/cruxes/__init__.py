"""How far two pools of premises about the same forecasting question overlap.

A forecaster, asked in advance what would decide a question, states some
premises: the introspected pool. Forecasts of the same question are later
compared, and the premises they actually disagreed on are extracted: the
extracted pool. Pool the two, cluster them by meaning, and count how many
clusters draw from both. That count, against a label shuffling null, is the
measurement this package implements, for any forecaster and any question.

Two stages, two commands. ``cruxes-score`` scores every pair of selected
premises with a cross encoder and needs a GPU and the ``gpu`` extra.
``cruxes-report`` turns the scores into the report and needs numpy and scipy
only. Importing this package never imports torch.

The package is small primitives (dedupe, select, score, aggregate, cluster,
decompose, permute) and one pipeline over them (``set_inclusion_curve`` and
``summarize_curves``, driven by the two commands). Every research choice the
pipeline makes ships one default, ours, and takes the caller's own instead:
the matching criterion, the reranker prompt, the aggregation, the permutation
seeding, the removal of repeated premises and the premise selection. Procedures beyond the pipeline, such as
the paper's baselines, are composed from the primitives outside the package.
"""

from .dedupe import DEDUPE_RULE, dedupe_premises, normalize_premise_text
from .clustering import (
    ClusteringDiagnostics,
    clusters_at_resolution,
    consensus_clustering,
    linkage_from_similarity,
    optimal_cluster_count,
)
from .matching import DEFAULT_MATCHING_CRITERION, MatchingCriterion, load_matching_criterion
from .permutation import DEFAULT_N_PERMUTATIONS, default_permutation_seed, permutation_null
from .premises import (
    DEFAULT_SEED,
    EXTRACTED,
    INTROSPECTED,
    PremiseSet,
    dedupe_premise_set,
    load_premise_sets,
    select_premises,
)
from .reranker.prompt import DEFAULT_RERANKER_PROMPT, RerankerPrompt, load_reranker_prompt
from .scorecache import (
    CACHE_FORMAT_VERSION,
    READABLE_FORMAT_VERSIONS,
    pool_item_ids,
    ScoreCacheKey,
    build_score_cache_key,
    cache_digest,
    cache_filename,
    cache_path,
    find_cached_scores,
    load_score_matrices,
    read_score_cache,
    write_score_cache,
)
from .scoring import SCORE_KEYS, ScoreMatrices, aggregate_scores, item_ids
from .selection import derive_seed, select_prefix
from .setinclusion import set_inclusion_curve, summarize_curves
from .stats import mean_and_error
from .venn import INTROSPECTED_PREFIX, introspected_ids, venn_counts, venn_decomposition

__version__ = "0.4.0"

__all__ = [
    "CACHE_FORMAT_VERSION",
    "READABLE_FORMAT_VERSIONS",
    "ClusteringDiagnostics",
    "DEDUPE_RULE",
    "DEFAULT_MATCHING_CRITERION",
    "DEFAULT_N_PERMUTATIONS",
    "DEFAULT_RERANKER_PROMPT",
    "DEFAULT_SEED",
    "EXTRACTED",
    "INTROSPECTED",
    "INTROSPECTED_PREFIX",
    "MatchingCriterion",
    "PremiseSet",
    "RerankerPrompt",
    "SCORE_KEYS",
    "ScoreCacheKey",
    "ScoreMatrices",
    "aggregate_scores",
    "build_score_cache_key",
    "cache_digest",
    "cache_filename",
    "cache_path",
    "clusters_at_resolution",
    "consensus_clustering",
    "dedupe_premise_set",
    "dedupe_premises",
    "default_permutation_seed",
    "derive_seed",
    "find_cached_scores",
    "introspected_ids",
    "item_ids",
    "linkage_from_similarity",
    "load_matching_criterion",
    "load_premise_sets",
    "load_reranker_prompt",
    "load_score_matrices",
    "mean_and_error",
    "normalize_premise_text",
    "optimal_cluster_count",
    "permutation_null",
    "pool_item_ids",
    "read_score_cache",
    "select_prefix",
    "select_premises",
    "set_inclusion_curve",
    "summarize_curves",
    "venn_counts",
    "venn_decomposition",
    "write_score_cache",
]
