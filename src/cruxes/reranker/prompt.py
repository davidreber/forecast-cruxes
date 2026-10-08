"""The reranker prompt, as an explicit string that lives in this package.

The cross encoder is asked, for a triple of an instruction, a query and a
document, whether the document meets the instruction's requirement given the
query, and the probability it puts on the token ``yes`` is the score. Getting
that question in front of the model means rendering it in the chat format the
model was trained on.

An earlier version of the authors' code rendered it by handing a message list
to the tokenizer's ``apply_chat_template``. On 2026-06-29 the upstream repository added a
``chat_template.jinja`` whose rendering drops the user message entirely and
emits an empty query and an empty document. The weights were unchanged and no
call failed. Every pair was scored through one identical prompt, so every score
was the same constant, every item clustered alone and every overlap statistic
came out zero, for three months. Every score in the paper was computed at the
pinned revision, after that pin was in place.

So the template is here, as a string, and ``apply_chat_template`` is never
called. The string below is what revision 5fa94080 of Qwen/Qwen3-Reranker-8B
renders for the two message list this package builds. A fixture of renderings
taken from that tokenizer is committed with the tests, and a CPU test compares
this template's output against it byte for byte, so the next upstream template
change is a failed string comparison rather than a plausible looking number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..matching import MatchingCriterion

__all__ = [
    "RERANKER_SYSTEM_MESSAGE",
    "RERANKER_PROMPT_TEMPLATE",
    "RerankerPrompt",
    "DEFAULT_RERANKER_PROMPT",
    "load_reranker_prompt",
    "PROMPT_DIRECTIONS",
    "render_reranker_prompt",
    "render_pair_prompts",
    "pair_prompt_order",
]

#: The instruction the model is given about its own task. Verbatim what produced
#: the paper's scores; changing it changes every score.
RERANKER_SYSTEM_MESSAGE = (
    "Judge whether the Document meets the requirements based on the "
    "Query and the Instruct provided. Note that the answer can only "
    'be "yes" or "no".'
)

#: The full rendered prompt, with the four fields left to format. Byte for byte
#: what the pinned tokenizer's chat template produces with
#: ``add_generation_prompt=True`` and no trailing thinking block.
RERANKER_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "{system}<|im_end|>\n"
    "<|im_start|>user\n"
    "<Instruct>: {instruction}\n"
    "<Query>: {query}\n"
    "<Document>: {document}<|im_end|>\n"
    "<|im_start|>assistant\n"
)



@dataclass(frozen=True)
class RerankerPrompt:
    """The text around the instruction, query and document: a research choice.

    The package ships one, :data:`DEFAULT_RERANKER_PROMPT`, the one that
    produced the paper's scores. A caller replaces it by passing another; both
    fields are in every score cache key, so caches under two prompts cannot
    collide.
    """

    system_message: str
    template: str


DEFAULT_RERANKER_PROMPT = RerankerPrompt(
    system_message=RERANKER_SYSTEM_MESSAGE, template=RERANKER_PROMPT_TEMPLATE
)


def load_reranker_prompt(path=None) -> RerankerPrompt:
    """The shipped prompt when ``path`` is None, otherwise the one in the file.

    The file is JSON with ``system_message`` and ``template``; the template
    must contain the four fields ``{system}``, ``{instruction}``, ``{query}``
    and ``{document}``.
    """
    if path is None:
        return DEFAULT_RERANKER_PROMPT
    with open(Path(path), encoding="utf-8") as handle:
        data = json.load(handle)
    prompt = RerankerPrompt(system_message=data["system_message"], template=data["template"])
    for field in ("{system}", "{instruction}", "{query}", "{document}"):
        if field not in prompt.template:
            raise ValueError(f"{path}: template has no {field} field")
    return prompt


#: The four questions asked about every pair, in the order they are scored.
#: ``positive_forward`` puts item i in the query slot under the positive
#: instruction, ``positive_reverse`` swaps the two items, and the two negative
#: entries do the same under the negative instruction.
PROMPT_DIRECTIONS = (
    "positive_forward",
    "positive_reverse",
    "negative_forward",
    "negative_reverse",
)


def render_reranker_prompt(
    instruction: str,
    query: str,
    document: str,
    system_message: str = RERANKER_SYSTEM_MESSAGE,
    template: str = RERANKER_PROMPT_TEMPLATE,
) -> str:
    """Render one instruction, query and document triple into one prompt string.

    No escaping and no normalisation. Whatever the three strings hold, including
    newlines, quotation marks and chat markers, goes in unchanged, which is what
    the tokenizer's own template did and therefore what the recorded scores were
    produced from.
    """
    return template.format(
        system=system_message,
        instruction=instruction,
        query=query,
        document=document,
    )


def pair_prompt_order(n_items: int) -> list:
    """Every (i, j, direction) triple to score, in the fixed order they run in.

    Pairs ascend in ``(i, j)`` with ``i < j``, and the four directions of a pair
    are adjacent, so a pair's four prompts always share a batch when the batch
    size is a multiple of four. The order is a property of the item count alone,
    so two runs over the same items batch identically.

    Returns a list of ``(i, j, direction)`` tuples of length ``4 * n_pairs``.
    """
    order = []
    for i in range(n_items):
        for j in range(i + 1, n_items):
            for direction in PROMPT_DIRECTIONS:
                order.append((i, j, direction))
    return order


def render_pair_prompts(
    items: list,
    criterion: MatchingCriterion,
    prompt: RerankerPrompt = DEFAULT_RERANKER_PROMPT,
) -> list:
    """Render every prompt for every pair of ``items``, in scoring order.

    The returned list lines up index for index with :func:`pair_prompt_order`.
    """
    instruction_for = {
        "positive_forward": criterion.positive_instruction,
        "positive_reverse": criterion.positive_instruction,
        "negative_forward": criterion.negative_instruction,
        "negative_reverse": criterion.negative_instruction,
    }
    prompts = []
    for i, j, direction in pair_prompt_order(len(items)):
        query, document = (items[i], items[j])
        if direction.endswith("_reverse"):
            query, document = document, query
        prompts.append(
            render_reranker_prompt(
                instruction_for[direction],
                query,
                document,
                system_message=prompt.system_message,
                template=prompt.template,
            )
        )
    return prompts
