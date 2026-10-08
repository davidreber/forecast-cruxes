"""The input a run starts from: two premise pools per question.

One file holds every question, and every question holds two lists of strings.

``introspected`` holds the premises a forecaster states in advance, when asked
what would decide the question. ``extracted`` holds the premises extracted
afterwards from where forecasts of the question actually disagreed. The
measurement is how much of the first turns up again in the second, so which
list is which is not a display detail.

```json
{
  "questions": [
    {"question_id": "...", "introspected": ["..."], "extracted": ["..."]}
  ]
}
```

A premise is an opaque string. The package never parses it. A question record
may carry any other keys (a question text, options, forecasts, per premise
records with provenance); this pipeline ignores them and they are left for
the adapters and pipelines that need them.

This module also holds the two steps between that file and the scoring
stage. First, repeated premise texts are removed from each pool
(:mod:`cruxes.dedupe`), so the selection is over distinct premises. Then each
pool is cut to at most ``max_contrasts_per_side`` premises by a seeded prefix
selection, so a curve across counts is nested rather than a fresh draw at
every count. The seeds come from the question id and the pool's name.

No torch, no file writing, and nothing that needs a GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .dedupe import dedupe_premises
from .selection import derive_seed, select_prefix

__all__ = [
    "INTROSPECTED",
    "EXTRACTED",
    "DEFAULT_SEED",
    "PremiseSet",
    "load_premise_sets",
    "dedupe_premise_set",
    "select_premises",
]

#: The two pool names. They are the input file's keys, the item id prefixes and
#: the strings hashed into each pool's selection seed.
INTROSPECTED = "introspected"
EXTRACTED = "extracted"

#: The base seed every derived seed starts from, unless a caller passes another.
DEFAULT_SEED = 42


@dataclass(frozen=True)
class PremiseSet:
    """One question and its two premise pools, before any selection."""

    question_id: str
    introspected: tuple
    extracted: tuple

    def __post_init__(self):
        object.__setattr__(self, "question_id", str(self.question_id))
        object.__setattr__(self, "introspected", tuple(self.introspected))
        object.__setattr__(self, "extracted", tuple(self.extracted))
        for name in (INTROSPECTED, EXTRACTED):
            pool = getattr(self, name)
            if not pool:
                raise ValueError(
                    f"question {self.question_id!r} has an empty {name} pool; "
                    "there is nothing to compare, so leave the question out"
                )
            bad = [i for i, text in enumerate(pool) if not isinstance(text, str)]
            if bad:
                raise ValueError(
                    f"question {self.question_id!r}: {name} premises {bad[:5]} "
                    "are not strings"
                )


def load_premise_sets(path) -> list:
    """Read a premise file. Raises on a shape it does not recognise.

    Accepts either ``{"questions": [...]}`` or a bare list of question objects,
    since the second is what a hand written file usually looks like.
    """
    with open(Path(path), encoding="utf-8") as handle:
        data = json.load(handle)
    records = data["questions"] if isinstance(data, dict) else data

    sets, seen = [], set()
    for index, record in enumerate(records):
        missing = [k for k in ("question_id", INTROSPECTED, EXTRACTED) if k not in record]
        if missing:
            raise ValueError(f"question {index} is missing {', '.join(missing)}")
        question_id = str(record["question_id"])
        if question_id in seen:
            raise ValueError(f"duplicate question_id {question_id!r}")
        seen.add(question_id)
        sets.append(
            PremiseSet(
                question_id=question_id,
                introspected=record[INTROSPECTED],
                extracted=record[EXTRACTED],
            )
        )
    return sets


def dedupe_premise_set(premise_set: PremiseSet, dedupe=dedupe_premises) -> tuple:
    """``(deduped, counts)``: the question with repeats removed from both pools.

    ``counts`` is ``{pool name: {"given": n, "kept": m, "removed": n - m}}``,
    which is what the commands print and the report records. ``dedupe`` is a
    function from a pool to ``(kept, dropped indices)``; ``None`` keeps every
    pool as given, with ``removed`` 0 everywhere.
    """
    pools, counts = {}, {}
    for name in (INTROSPECTED, EXTRACTED):
        given = list(getattr(premise_set, name))
        kept, dropped = dedupe(given) if dedupe is not None else (given, [])
        pools[name] = kept
        counts[name] = {"given": len(given), "kept": len(kept), "removed": len(dropped)}
    return PremiseSet(premise_set.question_id, pools[INTROSPECTED], pools[EXTRACTED]), counts


def select_premises(
    premise_set: PremiseSet,
    max_contrasts_per_side: int,
    seed: int = DEFAULT_SEED,
    dedupe=dedupe_premises,
) -> tuple:
    """The premises actually scored for this question, in scoring order.

    Repeated texts are removed from each pool first (``dedupe``; pass ``None``
    to keep them), then each pool is a seeded prefix of its own permutation,
    capped at the number of distinct premises the pool has. The seed of a pool
    is derived from ``seed``, the question id and the pool's name. Returns
    ``(introspected, extracted)`` as lists.

    This is the package's one default selection. A caller who wants another
    selects the premises themselves and passes them to the scoring and report
    functions directly.
    """
    if max_contrasts_per_side < 1:
        raise ValueError(
            f"max_contrasts_per_side must be at least 1, got {max_contrasts_per_side}"
        )
    deduped, _ = dedupe_premise_set(premise_set, dedupe)
    selected = []
    for name in (INTROSPECTED, EXTRACTED):
        pool = list(getattr(deduped, name))
        selected.append(
            select_prefix(
                pool,
                min(max_contrasts_per_side, len(pool)),
                derive_seed(seed, premise_set.question_id, name),
            )
        )
    return selected[0], selected[1]
