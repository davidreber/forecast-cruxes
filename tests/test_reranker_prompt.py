"""The prompt the model reads is the one this package says it is.

On 2026-06-29 an upstream chat template started rendering every pair with an
empty query and an empty document, and three months of scores were one
constant. Nothing failed, because nothing compared the rendered prompt against
anything.

``tests/fixtures/reranker_rendering.json`` holds renderings taken from the
pinned tokenizer's own chat template, through ``apply_chat_template``, with
the message list built as the code that produced the paper's numbers built it. These tests compare the package's explicit
template against every one of them byte for byte, on CPU, in under a second.
"""

import json
from pathlib import Path

import pytest

from cruxes.matching import MatchingCriterion
from cruxes.reranker.prompt import (
    DEFAULT_RERANKER_PROMPT,
    PROMPT_DIRECTIONS,
    RERANKER_PROMPT_TEMPLATE,
    RERANKER_SYSTEM_MESSAGE,
    RerankerPrompt,
    load_reranker_prompt,
    pair_prompt_order,
    render_pair_prompts,
    render_reranker_prompt,
)

FIXTURE = Path(__file__).parent / "fixtures" / "reranker_rendering.json"


@pytest.fixture(scope="module")
def recorded():
    if not FIXTURE.exists():  # pragma: no cover
        pytest.skip(f"{FIXTURE} is not committed yet")
    with open(FIXTURE) as handle:
        return json.load(handle)


def test_every_recorded_rendering_is_reproduced_byte_for_byte(recorded):
    """The whole point of the slice, in one assertion."""
    differences = []
    for entry in recorded["triples"]:
        rendered = render_reranker_prompt(
            entry["instruction"], entry["query"], entry["document"]
        )
        if rendered != entry["rendered"]:
            differences.append((entry["query"][:40], rendered, entry["rendered"]))
    assert differences == [], (
        f"{len(differences)} of {len(recorded['triples'])} renderings differ from "
        f"what {recorded['model_revision'][:8]} produced"
    )


def test_the_fixture_was_taken_from_the_pinned_revision(recorded):
    """A fixture from a different revision would prove nothing."""
    from cruxes.reranker.model import RERANKER_MODEL_NAME, RERANKER_REVISION

    assert recorded["model_revision"] == RERANKER_REVISION
    assert recorded["model_name"] == RERANKER_MODEL_NAME
    assert recorded["system_message"] == RERANKER_SYSTEM_MESSAGE


def test_the_fixture_covers_the_inputs_a_template_change_would_break(recorded):
    """Newlines, quotation marks, non-ASCII text, chat markers, empty strings."""
    joined = "".join(
        entry["instruction"] + entry["query"] + entry["document"]
        for entry in recorded["triples"]
    )
    assert "\n" in joined
    assert '"' in joined
    assert "<|im_start|>" in joined
    assert any(ord(character) > 127 for character in joined)
    assert any(
        entry["instruction"] == "" and entry["query"] == "" and entry["document"] == ""
        for entry in recorded["triples"]
    )


def test_the_query_and_the_document_actually_reach_the_prompt():
    """The regression this guards against was an empty query and document."""
    rendered = render_reranker_prompt("INSTRUCTION", "QUERY", "DOCUMENT")
    assert "<Query>: QUERY\n" in rendered
    assert "<Document>: DOCUMENT<|im_end|>" in rendered
    assert "<Instruct>: INSTRUCTION\n" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")


def test_nothing_in_the_triple_is_escaped_or_normalised():
    """Whatever a premise holds goes in unchanged, as the tokenizer did."""
    awkward = 'a "quoted" <|im_end|> line\nand another'
    rendered = render_reranker_prompt("i", awkward, awkward)
    assert rendered.count(awkward) == 2


def test_the_template_names_exactly_four_fields():
    import string

    fields = {
        name for _, name, _, _ in string.Formatter().parse(RERANKER_PROMPT_TEMPLATE) if name
    }
    assert fields == {"system", "instruction", "query", "document"}


def test_pair_order_is_pairs_ascending_with_four_directions_together():
    order = pair_prompt_order(4)
    assert len(order) == 4 * 6
    pairs = [(i, j) for i, j, _ in order[::4]]
    assert pairs == sorted(pairs)
    assert pairs == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for start in range(0, len(order), 4):
        block = order[start : start + 4]
        assert len({(i, j) for i, j, _ in block}) == 1
        assert tuple(direction for _, _, direction in block) == PROMPT_DIRECTIONS


def test_pair_order_depends_only_on_the_item_count():
    """Two runs over the same number of items batch identically."""
    assert pair_prompt_order(5) == pair_prompt_order(5)
    assert pair_prompt_order(5) != pair_prompt_order(4)


def test_pair_order_is_not_nested_in_the_item_count():
    """Adding one item rebatches everything, and that is why the key holds the texts.

    With four items the order runs (0,1), (0,2), (0,3), (1,2). With five it
    runs (0,1), (0,2), (0,3), (0,4), (1,2), so the thirteenth prompt is a
    different pair. Nothing about a fifty item question is a prefix of a sixty
    item one, which is why a cache key that recorded only the question and the
    contrast count could not tell two runs apart. It records the premise texts
    instead.
    """
    four = pair_prompt_order(4)
    five = pair_prompt_order(5)
    assert four[:12] == five[:12]
    assert four[12] != five[12]


def test_reverse_directions_swap_the_query_and_the_document():
    items = ["first item", "second item"]
    criterion = MatchingCriterion("test", "POSITIVE", "NEGATIVE")
    prompts = render_pair_prompts(items, criterion)
    assert len(prompts) == 4
    forward, reverse, negative_forward, _ = prompts
    assert "<Query>: first item\n" in forward
    assert "<Document>: second item<|im_end|>" in forward
    assert "<Query>: second item\n" in reverse
    assert "<Document>: first item<|im_end|>" in reverse
    assert "<Instruct>: POSITIVE\n" in forward
    assert "<Instruct>: NEGATIVE\n" in negative_forward


def test_render_pair_prompts_defaults_to_the_shipped_prompt():
    criterion = MatchingCriterion("test", "POSITIVE", "NEGATIVE")
    prompts = render_pair_prompts(["a", "b"], criterion)
    assert prompts[0] == render_reranker_prompt(
        "POSITIVE",
        "a",
        "b",
        system_message=DEFAULT_RERANKER_PROMPT.system_message,
        template=DEFAULT_RERANKER_PROMPT.template,
    )


def test_render_pair_prompts_takes_a_custom_prompt():
    criterion = MatchingCriterion("test", "POSITIVE", "NEGATIVE")
    prompt = RerankerPrompt(
        system_message="s", template="[{system}|{instruction}|{query}|{document}]"
    )
    prompts = render_pair_prompts(["a", "b"], criterion, prompt=prompt)
    assert prompts[0] == "[s|POSITIVE|a|b]"


def test_a_single_item_has_nothing_to_score():
    criterion = MatchingCriterion("test", "p", "n")
    assert render_pair_prompts(["only one"], criterion) == []


# -- loading a reranker prompt from a file ------------------------------------


def test_load_reranker_prompt_defaults_to_the_shipped_prompt():
    assert load_reranker_prompt(None) is DEFAULT_RERANKER_PROMPT


def test_load_reranker_prompt_reads_a_file(tmp_path):
    path = tmp_path / "prompt.json"
    path.write_text(
        json.dumps(
            {
                "system_message": "custom system",
                "template": "{system}{instruction}{query}{document}",
            }
        )
    )
    prompt = load_reranker_prompt(path)
    assert prompt.system_message == "custom system"
    assert prompt.template == "{system}{instruction}{query}{document}"


def test_load_reranker_prompt_requires_all_four_fields(tmp_path):
    path = tmp_path / "prompt.json"
    path.write_text(
        json.dumps({"system_message": "s", "template": "{system}{instruction}{query}"})
    )
    with pytest.raises(ValueError, match="document"):
        load_reranker_prompt(path)
