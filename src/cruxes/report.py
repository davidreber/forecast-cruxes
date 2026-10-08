"""Command line entry for the CPU stage: score caches in, the report out.

    cruxes-report --premises questions.json --cache-dir runs/my_run --out-dir runs/my_run

The caches are found by recomputing the key the GPU stage computed, from the
same premise file and the same arguments, so nothing depends on a filename
convention or a symlink. A missing cache is named and the run stops, rather
than producing a summary over whatever happened to be there. The report's
metadata records how many repeated premise texts were removed from each pool
before selection, per question and in total.

The report is written as ``set_inclusion_{UTC timestamp}.json`` inside
``--out-dir``, created exclusively. There is no ``_latest`` link and no way to
name an existing file.

numpy and scipy only. Importing this module does not import torch.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .permutation import DEFAULT_N_PERMUTATIONS, default_permutation_seed
from .runplan import add_shared_arguments, deduplication_summary, plan, settings_from_args
from .scorecache import find_cached_scores, load_score_matrices
from .scoring import DEFAULT_AGGREGATION_NAME
from .setinclusion import set_inclusion_curve, summarize_curves


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cruxes-report",
        description="Build the set inclusion report from score caches (CPU).",
    )
    add_shared_arguments(parser)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--permutations", type=int, default=DEFAULT_N_PERMUTATIONS)
    parser.add_argument(
        "--no-permutation",
        action="store_true",
        help="Skip the label shuffling null, which is most of the run time.",
    )
    return parser


def _write_new(path: Path, payload: dict) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
    print(f"wrote {path}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    settings = settings_from_args(args)
    planned = plan(args.premises, settings, args.questions)
    with_permutation = not args.no_permutation

    loaded, missing = {}, []
    for entry in planned:
        path = find_cached_scores(args.cache_dir, entry.key)
        if path is None:
            missing.append(entry.premise_set.question_id)
            continue
        loaded[entry.premise_set.question_id] = (load_score_matrices(path, entry.pools), path.name)

    if missing:
        print(f"{len(missing)} questions have no score cache under {args.cache_dir}:")
        for question_id in missing:
            print(f"  {question_id}")
        print("Run cruxes-score with the same arguments on a GPU node first.")
        return 1

    curves = {}
    for question_id, (matrices, cache_name) in loaded.items():
        curves[question_id] = set_inclusion_curve(
            matrices,
            args.max_contrasts_per_side,
            question_id,
            with_permutation=with_permutation,
            n_permutations=args.permutations,
            permutation_seed=lambda q, c: default_permutation_seed(q, c, settings.seed),
        )
        print(f"  {question_id}: {cache_name}")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    metadata = {
        "generated": stamp,
        "cruxes_version": __version__,
        "premises_file": str(args.premises),
        "cache_dir": str(args.cache_dir),
        **settings.describe(),
        "aggregation": DEFAULT_AGGREGATION_NAME,
        "permutation_seeding": (
            f"hash of question id and contrasts_per_side on base seed {settings.seed}"
        ),
        "n_permutations": args.permutations if with_permutation else 0,
        "deduplication": deduplication_summary(planned),
        "score_caches": {q: name for q, (_, name) in loaded.items()},
    }
    report = summarize_curves(
        curves, args.max_contrasts_per_side, metadata=metadata, with_permutation=with_permutation
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_new(args.out_dir / f"set_inclusion_{stamp}.json", report)
    top = report["per_contrasts_per_side"].get(str(args.max_contrasts_per_side))
    if top is not None:
        summary = top["summary"]
        line = (
            f"at {args.max_contrasts_per_side} contrasts a side: shared fraction "
            f"{summary['shared_frac_mean']:.4f} (se {summary['shared_frac_se']:.4f}) "
            f"over {summary['n_questions']} questions"
        )
        if "permutation" in top:
            line += f", permutation null {top['permutation']['permutation_shared_mean']:.4f}"
        print(line)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
