"""Removing repeated premise texts from a pool before anything is selected.

A pool can hold the same premise more than once: the extraction stage sees a
premise again in every pair of forecasts that disagreed on it. Selection draws
from the pool as a list, so a premise present three times is three times as
likely to be drawn and two copies can both be drawn, score about 1 against
each other, and form a cluster from one pool out of nothing new. The
measurement counts distinct premises, so repeats are removed first.

The rule: whitespace-collapsed exact match, case sensitive, first occurrence
kept. In detail:

- normalize a premise by ``" ".join(text.split())``: strip, and collapse every
  run of whitespace to one space;
- two premises of one pool are repeats when their normalized strings are
  exactly equal, case sensitive, with no fuzzy matching;
- the first occurrence is kept verbatim in its original position; later
  copies are dropped.

This is a primitive of the package and the default pipeline applies it to both
pools of every question before selection, reporting how many texts it removed.
A caller who wants repeats kept passes ``dedupe=None`` to
:func:`cruxes.premises.select_premises`, or their own function of the same
shape.

No numpy, no file access.
"""

from __future__ import annotations

__all__ = ["DEDUPE_RULE", "normalize_premise_text", "dedupe_premises"]

#: The rule in one sentence, recorded in every report that applied it.
DEDUPE_RULE = (
    "repeated premise texts removed before selection: texts equal after "
    "collapsing whitespace runs to one space (case sensitive, no fuzzy matching) "
    "are repeats; the first occurrence is kept verbatim in place"
)


def normalize_premise_text(text: str) -> str:
    """The form two premises are compared in: whitespace runs collapsed, case kept."""
    return " ".join(text.split())


def dedupe_premises(pool) -> tuple:
    """``(kept, dropped)``: the pool without repeats, and the indices dropped.

    ``kept`` holds the first occurrence of every distinct normalized text, in
    the pool's own order and with its original spelling; ``dropped`` lists the
    positions of the later copies, ascending. A pool without repeats comes
    back unchanged with an empty ``dropped``.
    """
    seen = set()
    kept, dropped = [], []
    for index, text in enumerate(pool):
        key = normalize_premise_text(text)
        if key in seen:
            dropped.append(index)
            continue
        seen.add(key)
        kept.append(text)
    return kept, dropped
