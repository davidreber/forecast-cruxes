"""Loading the pinned cross encoder and scoring a list of prompts with it.

This module is the only place in the package that imports torch or
transformers, and it is not imported by ``cruxes/__init__.py``. Installing the
package without the ``gpu`` extra leaves everything else working.

Scoring is the linear projection form: the last hidden state of the final
position is projected onto the difference between the ``yes`` and ``no`` rows of
the language modelling head and passed through a sigmoid. That is algebraically
the same number as taking the softmax over the two token logits, because

    sigmoid(h . (w_yes - w_no)) = softmax([h . w_no, h . w_yes])[1]

and it avoids computing logits over a hundred and fifty thousand token
vocabulary for every prompt. The code that produced the paper's numbers did
the same thing, so the scores are comparable with the paper's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "RERANKER_MODEL_NAME",
    "RERANKER_REVISION",
    "RERANKER_MAX_LENGTH",
    "RERANKER_DTYPE",
    "DEFAULT_BATCH_SIZE",
    "LoadedReranker",
    "load_reranker",
    "score_prompts",
]

#: The cross encoder the paper's scores were produced with.
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-8B"

#: The one place the revision is pinned. Every caller, the cache key and the
#: metadata of every written cache read it from here.
#:
#: Upstream revision 77d193c, which the hub's ``main`` ref has pointed at since
#: 2026-06-29, carries identical weights and a ``chat_template.jinja`` that
#: renders every pair with an empty query and an empty document. This package
#: never calls that template (see ``prompt.py``), so the pin matters only for
#: the weights and the tokenizer, but an unpinned load is still an invitation
#: for the next upstream change to arrive unannounced.
RERANKER_REVISION = "5fa94080caafeaa45a15d11f969d7978e087a3db"

#: Prompts longer than this are truncated, as in the code that produced the
#: paper's numbers. A premise
#: pair is two or three hundred tokens, so truncation is not expected to fire.
RERANKER_MAX_LENGTH = 8192

#: Precision of the forward pass. Part of every cache key, since half and full
#: precision give different scores.
RERANKER_DTYPE = "float16"

#: Prompts per forward pass. Fixed, not adapted from free memory. A multiple of
#: four, so the four prompts of a pair never straddle a batch boundary. A fixed
#: size keeps every prompt's batch and padding the same from run to run, so two
#: runs of the same input on the same hardware give the same numbers; the cost
#: is that spare GPU memory goes unused and a GPU with too little memory fails
#: rather than shrinking the batch.
DEFAULT_BATCH_SIZE = 32


@dataclass
class LoadedReranker:
    """A loaded cross encoder and the two pieces needed to score with it."""

    tokenizer: object
    base_model: object
    score_linear: object
    device: str
    model_name: str
    model_revision: str
    max_length: int = RERANKER_MAX_LENGTH
    dtype: str = RERANKER_DTYPE
    environment: dict = field(default_factory=dict)


def load_reranker(
    model_name: str = RERANKER_MODEL_NAME,
    revision: str = RERANKER_REVISION,
    device: str = "cuda",
    dtype: str = RERANKER_DTYPE,
    max_length: int = RERANKER_MAX_LENGTH,
) -> LoadedReranker:
    """Load the tokenizer, the base model and the yes-minus-no projection.

    Requires a CUDA device. Raises rather than falling back to CPU, because an
    eight billion parameter cross encoder on CPU is not a slower answer, it is
    a job that never finishes.
    """
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "the reranker scoring stage needs a CUDA device and "
            "torch.cuda.is_available() returned False"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, padding_side="left",
    )
    try:
        language_model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, dtype=torch_dtype,
        )
    except TypeError:  # transformers below 4.56 spells it torch_dtype
        language_model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=torch_dtype,
        )
    language_model = language_model.to(device)

    base_model = language_model.model
    base_model.eval()

    token_yes = tokenizer.convert_tokens_to_ids("yes")
    token_no = tokenizer.convert_tokens_to_ids("no")
    unknown = getattr(tokenizer, "unk_token_id", None)
    if token_yes == unknown or token_no == unknown:
        raise ValueError(
            f"tokenizer for {model_name}@{revision} has no 'yes'/'no' tokens: "
            f"yes={token_yes}, no={token_no}"
        )

    weight_yes = language_model.lm_head.weight.data[token_yes]
    weight_no = language_model.lm_head.weight.data[token_no]
    score_linear = torch.nn.Linear(weight_yes.size(0), 1, bias=False)
    with torch.no_grad():
        score_linear.weight[0] = weight_yes - weight_no
    score_linear = score_linear.to(device).to(torch_dtype)
    score_linear.eval()

    del language_model
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    environment = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
    }
    return LoadedReranker(
        tokenizer=tokenizer,
        base_model=base_model,
        score_linear=score_linear,
        device=device,
        model_name=model_name,
        model_revision=revision,
        max_length=max_length,
        dtype=dtype,
        environment=environment,
    )


def score_prompts(
    reranker: LoadedReranker,
    prompts: list,
    batch_size: int = DEFAULT_BATCH_SIZE,
    report_every: int = 0,
) -> list:
    """Score every prompt, in order, in fixed size batches.

    Batches are consecutive slices of ``prompts``. No growing, no shrinking and
    no retry on an out of memory error: a batch size that does not fit is a
    configuration to change and record, not something to discover at run time
    and forget. Returns one float per prompt, in the order given.
    """
    import torch

    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    tokenizer = reranker.tokenizer
    base_model = reranker.base_model
    score_linear = reranker.score_linear
    device = next(base_model.parameters()).device

    scores = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=reranker.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            hidden = base_model(**inputs).last_hidden_state[:, -1]
            batch_scores = torch.sigmoid(score_linear(hidden))
            scores.extend(batch_scores.squeeze(-1).tolist())
        del inputs, hidden, batch_scores
        if report_every and (start // batch_size) % report_every == 0:
            print(f"    scored {len(scores)}/{len(prompts)} prompts", flush=True)
    return scores
