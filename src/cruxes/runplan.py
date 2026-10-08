"""What the two commands share: the arguments and the key of every question.

The GPU command and the CPU command must compute the same cache key for a
question from the same arguments, or the CPU command cannot find what the GPU
command wrote. Both therefore parse their shared arguments here and compute
keys here, and nowhere else.

No torch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .matching import MatchingCriterion, load_matching_criterion
from .dedupe import DEDUPE_RULE
from .premises import (
    DEFAULT_SEED,
    EXTRACTED,
    INTROSPECTED,
    PremiseSet,
    dedupe_premise_set,
    load_premise_sets,
    select_premises,
)
from .reranker.model import (
    DEFAULT_BATCH_SIZE,
    RERANKER_DTYPE,
    RERANKER_MAX_LENGTH,
    RERANKER_MODEL_NAME,
    RERANKER_REVISION,
)
from .reranker.prompt import RerankerPrompt, load_reranker_prompt
from .scorecache import ScoreCacheKey, build_score_cache_key

__all__ = [
    "RunSettings",
    "PlannedQuestion",
    "add_shared_arguments",
    "settings_from_args",
    "plan",
    "deduplication_summary",
]

#: The count the paper reports its headline at. Only a default.
DEFAULT_MAX_CONTRASTS_PER_SIDE = 25


@dataclass(frozen=True)
class RunSettings:
    """Everything that decides which premises are scored and how."""

    max_contrasts_per_side: int
    seed: int
    criterion: MatchingCriterion
    prompt: RerankerPrompt
    model_name: str
    model_revision: str
    batch_size: int
    max_length: int
    dtype: str

    def describe(self) -> dict:
        """The settings as a report's metadata records them."""
        return {
            "max_contrasts_per_side": self.max_contrasts_per_side,
            "seed": self.seed,
            "matching_criterion": self.criterion.to_dict(),
            "reranker_system_message": self.prompt.system_message,
            "reranker_prompt_template": self.prompt.template,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class PlannedQuestion:
    """One question as the commands will score and report it.

    ``premise_set`` is the question as the file gave it; ``duplicates`` says
    how many repeated texts were removed from each pool before selection
    (``{pool: {"given", "kept", "removed"}}``); ``key`` names the cache of the
    selection made from the deduplicated pools, its items being the
    introspected selection followed by the extracted one; ``pools`` is that
    split, ``{"introspected": n, "extracted": m}``, which the key does not
    carry.
    """

    premise_set: PremiseSet
    key: ScoreCacheKey
    duplicates: dict
    pools: dict


def deduplication_summary(planned: list) -> dict:
    """What a report records about the repeats removed across a run."""
    per_question = {p.premise_set.question_id: p.duplicates for p in planned}
    summary = {"rule": DEDUPE_RULE, "removed": {}, "questions_with_repeats": {}}
    for name in (INTROSPECTED, EXTRACTED):
        removed = [d[name]["removed"] for d in per_question.values()]
        summary["removed"][name] = int(sum(removed))
        summary["questions_with_repeats"][name] = int(sum(1 for r in removed if r))
    summary["per_question"] = per_question
    return summary


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--premises", required=True, type=Path, help="The premise file.")
    parser.add_argument("--cache-dir", required=True, type=Path, help="Where score caches live.")
    parser.add_argument(
        "--max-contrasts-per-side",
        type=int,
        default=DEFAULT_MAX_CONTRASTS_PER_SIDE,
        help="How many premises (contrasts) to select from each pool (side) at most.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base of every derived seed: the premise selection of each pool and the "
        "permutation null of each question and count.",
    )
    parser.add_argument(
        "--matching-criterion",
        type=Path,
        default=None,
        help="JSON with positive_instruction, negative_instruction and name. "
        "Default: the package's one criterion, 'resolution'.",
    )
    parser.add_argument(
        "--reranker-prompt",
        type=Path,
        default=None,
        help="JSON with system_message and template. Default: the package's prompt.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model-name", default=RERANKER_MODEL_NAME)
    parser.add_argument("--model-revision", default=RERANKER_REVISION)
    parser.add_argument(
        "--questions", nargs="*", default=None, help="Only these question ids. Default: all."
    )


def settings_from_args(args) -> RunSettings:
    return RunSettings(
        max_contrasts_per_side=args.max_contrasts_per_side,
        seed=args.seed,
        criterion=load_matching_criterion(args.matching_criterion),
        prompt=load_reranker_prompt(args.reranker_prompt),
        model_name=args.model_name,
        model_revision=args.model_revision,
        batch_size=args.batch_size,
        max_length=RERANKER_MAX_LENGTH,
        dtype=RERANKER_DTYPE,
    )


def plan(premises_path, settings: RunSettings, questions=None) -> list:
    """Every question in the file, sorted by id, with its selection's cache key.

    Repeated texts are removed from each pool before the selection, and the
    counts travel with the planned question so both commands can report them.
    """
    premise_sets = load_premise_sets(premises_path)
    if questions:
        wanted = set(questions)
        unknown = wanted - {p.question_id for p in premise_sets}
        if unknown:
            raise ValueError(f"not in {premises_path}: {sorted(unknown)}")
        premise_sets = [p for p in premise_sets if p.question_id in wanted]
    planned = []
    for premise_set in sorted(premise_sets, key=lambda p: p.question_id):
        deduped, duplicates = dedupe_premise_set(premise_set)
        introspected, extracted = select_premises(
            deduped, settings.max_contrasts_per_side, settings.seed
        )
        key = build_score_cache_key(
            list(introspected) + list(extracted),
            criterion=settings.criterion,
            prompt=settings.prompt,
            model_name=settings.model_name,
            model_revision=settings.model_revision,
            batch_size=settings.batch_size,
            max_length=settings.max_length,
            dtype=settings.dtype,
        )
        pools = {INTROSPECTED: len(introspected), EXTRACTED: len(extracted)}
        planned.append(PlannedQuestion(premise_set, key, duplicates, pools))
    return planned
