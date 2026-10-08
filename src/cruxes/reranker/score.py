"""Command line entry for the GPU stage: premise pools in, score caches out.

    cruxes-score --premises questions.json --cache-dir runs/my_run

One cache per distinct selection, named after a digest of exactly what
influences its scores. A question whose cache is already on disk with exactly
this key is skipped and reported; a file at that path with a different key
stops the run. Nothing is ever overwritten.

``--dry-run`` reports the work without loading the model, which is the cheap
way to find out what a run will cost and the only part of this module that
works without a GPU.
"""

from __future__ import annotations

import argparse
import sys
import time

from .. import __version__
from ..runplan import add_shared_arguments, deduplication_summary, plan, settings_from_args
from ..scorecache import cache_filename, find_cached_scores, write_score_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cruxes-score",
        description="Score every pair of selected premises for every question (GPU).",
    )
    add_shared_arguments(parser)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--array-task-id",
        type=int,
        default=None,
        help="With --num-array-tasks, take every nth question, so several array "
        "tasks can share one premise file without touching one another.",
    )
    parser.add_argument("--num-array-tasks", type=int, default=None)
    parser.add_argument(
        "--dry-run", action="store_true", help="Report the work and write nothing."
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.array_task_id is None) != (args.num_array_tasks is None):
        parser.error("--array-task-id and --num-array-tasks go together")
    if args.array_task_id is not None and not 0 <= args.array_task_id < args.num_array_tasks:
        parser.error(
            f"--array-task-id {args.array_task_id} is not in 0 to {args.num_array_tasks - 1}"
        )
    settings = settings_from_args(args)
    planned = plan(args.premises, settings, args.questions)
    if args.array_task_id is not None:
        planned = planned[args.array_task_id :: args.num_array_tasks]

    print(f"premises:   {args.premises}")
    print(f"cache dir:  {args.cache_dir}")
    print(f"criterion:  {settings.criterion.name}")
    print(f"model:      {settings.model_name} @ {settings.model_revision} ({settings.dtype})")
    print(f"batch size: {settings.batch_size} (fixed)")
    print(f"selection:  at most {settings.max_contrasts_per_side} a side, seed {settings.seed}")

    todo, seen = [], set()
    for entry in planned:
        key = entry.key
        n_items = key.n_items
        n_prompts = 4 * n_items * (n_items - 1) // 2
        cached = find_cached_scores(args.cache_dir, key) is not None
        duplicate = cache_filename(key) in seen
        seen.add(cache_filename(key))
        state = "cached" if cached else ("same selection as an earlier question" if duplicate else "to score")
        removed = entry.duplicates
        print(
            f"  {entry.premise_set.question_id}: {entry.pools['introspected']}+{entry.pools['extracted']} "
            f"items ({removed['introspected']['removed']}+{removed['extracted']['removed']} "
            f"repeated texts removed), {n_prompts} prompts, {state} -> {cache_filename(key)}"
        )
        if not cached and not duplicate:
            todo.append((entry, n_prompts))

    summary = deduplication_summary(planned)
    print(
        "repeated texts removed before selection: "
        + ", ".join(
            f"{pool} {summary['removed'][pool]} over "
            f"{summary['questions_with_repeats'][pool]} of {len(planned)} questions"
            for pool in ("introspected", "extracted")
        )
    )
    print(f"\n{len(todo)} caches to write, {sum(n for _, n in todo)} prompts in total")
    if args.dry_run:
        print("dry run, nothing written")
        return 0
    if not todo:
        print("every question already has a cache with this key, nothing to do")
        return 0

    from .model import load_reranker
    from .pair_scores import score_cache_key

    started = time.time()
    reranker = load_reranker(
        model_name=settings.model_name,
        revision=settings.model_revision,
        device=args.device,
        dtype=settings.dtype,
        max_length=settings.max_length,
    )
    print(f"model loaded in {time.time() - started:.0f}s: {reranker.environment}")

    for entry, n_prompts in todo:
        question_started = time.time()
        matrices = score_cache_key(reranker, entry.key, report_every=50)
        path = write_score_cache(
            args.cache_dir,
            entry.key,
            matrices,
            environment={
                **reranker.environment,
                "cruxes_version": __version__,
                "first_question_id": entry.premise_set.question_id,
            },
            pools=entry.pools,
        )
        print(
            f"  {entry.premise_set.question_id}: {n_prompts} prompts in "
            f"{time.time() - question_started:.0f}s -> {path.name}",
            flush=True,
        )

    print(f"done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
