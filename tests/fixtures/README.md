# Test fixtures

These files are inputs to the CPU test suite. They are test fixtures, not the
paper's numbers: none of them is a result the paper reports, and passing the
tests is not verification of the paper (see
[forecast-cruxes-verify](https://github.com/davidreber/forecast-cruxes-verify)
for that). They are not installed with the package.

All three were recorded on the authors' cluster. What each one records about
itself:

## `reranker_rendering.json`

Sixty four instruction, query and document triples and, for each, the prompt
that the tokenizer of `Qwen/Qwen3-Reranker-8B` at revision
`5fa94080caafeaa45a15d11f969d7978e087a3db` rendered through its own chat
template (`apply_chat_template`, transformers 5.4.0). The package never calls
that template; `tests/test_reranker_prompt.py` checks that the package's
explicit template reproduces every rendering byte for byte.

## `pinned_scores.json`

Three introspected and three extracted premises (invented, not from the
study), the sixty prompt scores that `Qwen/Qwen3-Reranker-8B` at revision
`5fa94080caafeaa45a15d11f969d7978e087a3db` gave them at batch size 32 under the
`resolution` criterion, the four assembled score matrices and their
aggregation. The file records its environment (torch 2.11.0+cu130,
transformers 5.4.0, an NVIDIA A100 80GB PCIe, CUDA 13.0), and its
`converted_from` field names the earlier fixture of the authors' code it was
converted from. `tests/test_recorded_scores.py` feeds the raw scores to the
package's assembly with the model call stubbed out.

## `recorded_replay_objects59.json`

Score matrices for three March Madness games (`akron-vs-texas-tech`,
`arizona-vs-purdue`, `cal-baptist-vs-kansas`), 25 introspected and 25
extracted premises each, under the `resolution` criterion, together with the
set inclusion, granularity sweep and within-pool numbers gathered from them at
the time. They come from the authors' run of 2026-09-21, which predates the
removal of repeated premises. The file holds no premise text, only matrices
and numbers. Its `sources` field records, for each original file, the path on
the authors' cluster and its sha256. That run seeded every permutation null
with the constant 42, which the file records and
`tests/test_developer_replay.py` puts back to replay the CPU arithmetic
exactly. The file itself does not record the model revision.
