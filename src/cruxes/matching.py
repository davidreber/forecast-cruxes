"""The matching criterion: what the cross encoder is asked about a pair.

Every pair of premises is asked once whether the two are the same question
and once whether they are different questions, and both answers are scored, so
the negative instruction is not decoration. Together the two instructions say
what "the same premise" means, which is a research choice and not a property
of the method.

So the package ships exactly one criterion, the one the paper's main text
uses, and a caller replaces it by passing their own. It ships no alternatives:
the paper's other two criteria are appendix material and live in the data
artifact, as data, not here.

The shipped criterion is called ``resolution``: two premises match when
resolving one would resolve the other. Its text is verbatim what produced the
paper's numbers.

No torch, no file access at import.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = [
    "MatchingCriterion",
    "DEFAULT_MATCHING_CRITERION",
    "load_matching_criterion",
]


@dataclass(frozen=True)
class MatchingCriterion:
    """A named pair of instructions for the cross encoder.

    ``name`` is a label for reports and has no effect on a score. The two
    instruction texts do, and both are part of every score cache key.
    """

    name: str
    positive_instruction: str
    negative_instruction: str

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_MATCHING_CRITERION = MatchingCriterion(
    name="resolution",
    positive_instruction=(
        "Two crux questions are equivalent if resolving one would necessarily "
        "resolve the other, because they ask about the same underlying factual "
        "question. Ignore differences in wording, specificity, or answer-label "
        "ordering (A/B may be swapped). Does the following crux question target "
        "the same factual uncertainty?"
    ),
    negative_instruction=(
        "Two crux questions are different if they could be resolved independently "
        "— knowing the answer to one would not determine the answer to the "
        "other. Even if they use similar vocabulary or concern the same topic, "
        "determine whether the following crux question targets a genuinely "
        "different factual uncertainty. Answer labels (A/B) may be swapped."
    ),
)


def load_matching_criterion(path=None) -> MatchingCriterion:
    """The shipped criterion when ``path`` is None, otherwise the one in the file.

    The file is JSON with ``positive_instruction`` and ``negative_instruction``
    and optionally ``name``. Missing instruction keys are an error rather than
    a silent fall back to the default, since a half replaced criterion is
    neither one.
    """
    if path is None:
        return DEFAULT_MATCHING_CRITERION
    with open(Path(path), encoding="utf-8") as handle:
        data = json.load(handle)
    missing = [
        key for key in ("positive_instruction", "negative_instruction") if key not in data
    ]
    if missing:
        raise ValueError(f"{path} is missing {', '.join(missing)}")
    return MatchingCriterion(
        name=str(data.get("name", Path(path).stem)),
        positive_instruction=data["positive_instruction"],
        negative_instruction=data["negative_instruction"],
    )
