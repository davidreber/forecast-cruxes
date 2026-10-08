# forecast-cruxes

**Version 0.4.0**, released with the paper "Can LLMs Anticipate Their
Cruxes?" by David Reber, James Reber, Ari Holtzman and Victor Veitch
(University of Chicago). The paper's numbers were verified with this code;
the verdict is in the verification repository
([VERIFICATION.md](https://github.com/davidreber/forecast-cruxes-verify/blob/main/VERIFICATION.md)).
Companion repositories:

- data release: [forecast-cruxes-data](https://github.com/davidreber/forecast-cruxes-data)
- expected numbers and verdict: [forecast-cruxes-verify](https://github.com/davidreber/forecast-cruxes-verify)

## What it measures

This is the crux-set comparison package. For each forecasting question it
takes two sets of cruxes: the **zero-shot** cruxes a forecaster predicted in
advance would decide the question, and the **many-shot** cruxes extracted from
pairs of its forecasts that disagreed. It measures how far the two sets
overlap and compares that with a permutation baseline.

In the package's own words: the forecaster's stated premises are the
**introspected** pool, the premises its forecasts actually disagreed on are
the **extracted** pool. Pool the two, cluster by meaning with a cross
encoder's pairwise scores, and count the clusters that draw from both pools.
Compare that with what random relabelling of the same clustering gives.
Overlap far below that null means the forecaster's introspection missed what
its forecasts turned on.

Any forecaster, any forecasting question. A question is an id and two lists of
strings; the package never parses a premise.

The package does not generate or extract cruxes. Both pools are inputs. The
prompts of the generation stage (prediction and extraction) are printed
verbatim in the paper's appendix.

The package ships no data and no recorded numbers. The paper's data are in
[forecast-cruxes-data](https://github.com/davidreber/forecast-cruxes-data);
the paper's expected numbers, their tolerances and the verdict are in
[forecast-cruxes-verify](https://github.com/davidreber/forecast-cruxes-verify).

### Glossary

| this package | the paper |
|---|---|
| premise | crux |
| introspected pool | zero-shot (or one-shot) cruxes, written by the predictor |
| extracted pool | many-shot cruxes, from the extractor |
| permutation null | permutation baseline |
| `shared_frac` | overlap |
| `equivalence` / `resolution` / `debate` matching criteria | strict / middle / broad matching levels |
| `max_contrasts_per_side` | k, cruxes per side |

## What you need to bring

The package compares two sets of cruxes per question; it does not generate
either of them. You bring both:

- the **introspected** pool: zero-shot cruxes, written by the predictor before
  any forecast is made, saying what it expects the question to turn on;
- the **extracted** pool: many-shot cruxes, written by the extractor from
  pairs of the predictor's forecasts that disagreed.

Each crux is a plain string; the package never parses it. In the paper every
crux in both pools is a question followed by two answer options labelled A
and B, one per line, for example (from the data release's March Madness
introspected pool):

```
Will Miami win the rebound battle overall?
A) Yes—Miami finishes with a higher total rebound count than Missouri.
B) No—Missouri finishes with equal or more rebounds than Miami.
```

The shipped matching criterion is written for that layout (it tells the
cross encoder that the A/B labels may be swapped). The worked example in
`examples/premises_example.json` uses shorter one-line questions without
options, so it shows the file format, not the paper's crux format.

The prompts for scouting, forecasting, zero-shot prediction and pairwise
extraction are printed verbatim in the paper's appendix "Prompts"; they are
not shipped as code. The paper's actual pools are in the data release,
[forecast-cruxes-data](https://github.com/davidreber/forecast-cruxes-data).

The package ships one matching criterion, `resolution` (the paper's middle
level). The strict (`equivalence`) and broad (`debate`) criteria are in the
data release's `method/matching_criteria.json`; pass one with
`--matching-criterion` (see "A matching criterion of your own" below).

## Install

Not on PyPI. Install from GitHub at the release tag:

```
pip install 'forecast-cruxes @ git+https://github.com/davidreber/forecast-cruxes@v0.4.0'        # numpy, scipy: the CPU stage
pip install 'forecast-cruxes[gpu] @ git+https://github.com/davidreber/forecast-cruxes@v0.4.0'   # adds torch and transformers: the scoring stage
```

or from a clone:

```
git clone https://github.com/davidreber/forecast-cruxes && cd forecast-cruxes && pip install -e '.[gpu,test]'
```

Running the tests (`pytest tests`) needs the `test` extra, which the clone
route above includes; `pip install -e '.[test]'` alone is enough for them.

The import name is `cruxes`. An unrelated package on PyPI is named `cruxes`
and installs the same import name; do not install both in one environment.

### Requirements

- Python 3.10 or later.
- The scoring stage (`cruxes-score`) needs a CUDA GPU with memory for an 8B
  parameter model in float16. The authors used A100 80GB and H200 class GPUs.
- Its first run downloads about 16 GB: `Qwen/Qwen3-Reranker-8B` at the pinned
  revision, from the Hugging Face Hub. No token is needed. `HF_HOME` chooses
  the cache directory.
- `cruxes-score --dry-run` and `cruxes-report` run on a CPU and do not need
  torch.

## Use

One input file holds every question and its two pools:

```json
{"questions": [{"question_id": "q1", "introspected": ["..."], "extracted": ["..."]}]}
```

### Worked example

`examples/premises_example.json` holds three invented questions. From the
repository root:

```
# 1. Dry run (CPU): what would be scored, and the repeats removed. Loads no model.
cruxes-score  --premises examples/premises_example.json --cache-dir run/caches --dry-run

# 2. Score (GPU): writes one score cache per question under run/caches.
cruxes-score  --premises examples/premises_example.json --cache-dir run/caches

# 3. Report (CPU): reads the caches, writes run/set_inclusion_<timestamp>.json.
cruxes-report --premises examples/premises_example.json --cache-dir run/caches --out-dir run
```

The first two commands write one score cache per distinct selection of
premises, named after a digest of exactly what influences its scores, and
never overwrite a file. `--dry-run` prints the cost without loading a model.
The report command recomputes the same keys from the same arguments, reads the
caches and writes the report. A missing cache stops it and names the question.
Importing the package, or running the report command, never imports torch.
The paper's two baselines are not commands of the package; they are
procedures in the verification repository, composed from the primitives.

The same steps from Python, after step 2:

```python
import argparse

from cruxes import find_cached_scores, load_score_matrices, set_inclusion_curve, summarize_curves
from cruxes.permutation import default_permutation_seed
from cruxes.runplan import add_shared_arguments, plan, settings_from_args

parser = argparse.ArgumentParser()
add_shared_arguments(parser)
args = parser.parse_args(
    ["--premises", "examples/premises_example.json", "--cache-dir", "run/caches"]
)
settings = settings_from_args(args)  # defaults: k = 25, the resolution criterion, the pinned model

curves = {}
for entry in plan(args.premises, settings):  # one entry per question, sorted by id
    question_id = entry.premise_set.question_id
    key = entry.key  # the ScoreCacheKey cruxes-score wrote this question's cache under
    matrices = load_score_matrices(find_cached_scores(args.cache_dir, key), entry.pools)
    curves[question_id] = set_inclusion_curve(
        matrices,
        settings.max_contrasts_per_side,
        question_id,
        permutation_seed=lambda q, c: default_permutation_seed(q, c, settings.seed),
    )

report = summarize_curves(curves, settings.max_contrasts_per_side)
print(report["per_contrasts_per_side"]["25"]["summary"]["shared_frac_mean"])
```

A single cache file can also be read directly with `load_score_matrices(path)`.

### A matching criterion of your own

`--matching-criterion file.json` (or `load_matching_criterion(path)`) reads
one criterion, a single JSON object:

```json
{
  "name": "my-criterion",
  "positive_instruction": "Two crux questions are equivalent if ...",
  "negative_instruction": "Two crux questions are different if ..."
}
```

Both instruction keys are required; `name` is optional (the file name is used
without it) and is only a label. The data release's
`method/matching_criteria.json` holds the paper's three criteria as a list
under `criteria`; to use one of them, take that one entry and write it as a
single object like the above.

## What the commands report

Before selection, both commands remove repeated premise texts from each pool
and say how many they removed. `cruxes-score` prints the count per question
and pool and the total over the run, under `--dry-run` too. `cruxes-report`
records the same counts in the report under `metadata.deduplication`: the rule
in words, the total removed per pool, how many questions had repeats in each
pool, and the counts per question. A dry run of the worked example reports two
repeats removed from one extracted pool.

## One default per research choice

The package is small primitives (dedupe, select, score, aggregate, cluster,
decompose, permute) and one pipeline over them, `set_inclusion_curve` and
`summarize_curves`, which the two commands drive. Every research choice the
pipeline makes ships one default, ours, and takes the caller's own instead.

| choice | the package's default | to use your own |
|---|---|---|
| matching criterion | `resolution`: two premises match if resolving one resolves the other | `--matching-criterion file.json`, or `MatchingCriterion(...)` |
| reranker prompt | the explicit template of `Qwen/Qwen3-Reranker-8B` at a pinned revision | `--reranker-prompt file.json`, or `RerankerPrompt(...)` |
| aggregation | positive instruction, both directions averaged | `aggregate=` a function of `ScoreMatrices` |
| permutation seeding | a hash of question id and `contrasts_per_side` | `permutation_seed=` a function |
| repeated premises | removed before selection: texts equal after collapsing whitespace, case sensitive; first kept | `dedupe=` a function or `None` to `select_premises` |
| premise selection | a seeded prefix, seed from question id and pool name | select yourself and call the Python API |

## Guarantees

- An unscored pair, anywhere, is an error, never a number.
- A score cache is never overwritten and is read back and checked before reuse.
- Reports are timestamped files; there are no `_latest` links.
- Scores run pair by pair in fixed batches of 32, so two runs of the same input
  on the same hardware give the same numbers.

## Relation to the paper's numbers

A rerun from the data release selects its own 25 premises per pool and scores
them in fixed batches on its own hardware, so it reproduces the paper's
numbers within the tolerances the verification repository defines, not
exactly. Those tolerances, the expected numbers and the verdict live in
[forecast-cruxes-verify](https://github.com/davidreber/forecast-cruxes-verify).

## Known limitations (v0.4.0)

- The scoring stage requires a CUDA device: `load_reranker` checks
  `torch.cuda.is_available()` and calls `torch.cuda` functions
  unconditionally, so `--device mps` or `--device cpu` does not work. A reader
  reported that the 8B model runs on Apple MPS in float16 with a 20-line
  patch; that will be considered for a later version. That reader measured
  about 2.3 prompts/s on real cruxes on an Apple M5; a question with 25 + 25
  premises is 4,900 prompts.
- The `set_inclusion` report records the Venn counts, the fractions and the
  permutation null per question, but not the cluster memberships or the
  `ClusteringDiagnostics` (`n_clusters`, `max_lifetime`); reconstruct them
  through the Python API with `aggregate_scores` and `consensus_clustering`
  (or `linkage_from_similarity`, `optimal_cluster_count` and
  `clusters_at_resolution`).
- The package picks no representative crux for a cluster.
- The longest-lifetime cut can put most of a pool into one cluster when the
  premises are long and similar to one another (a reader saw 35 of 44 premises
  in one cluster at a mean pairwise similarity of 0.46 under the `resolution`
  criterion); the worked example's short premises do not show this.
- The paper's baselines are composed in the verification repository, not
  here, and a second premise set passed under its own question id gets its own
  seeded draw, so reproducing one selection for a baseline needs the Python
  API (`derive_seed` and `select_prefix` in `src/cruxes/selection.py`).
- `cruxes-score` does not flush its header lines, so a log redirected to a
  file stays empty until the first batch is scored.
- Newer transformers (5.x) prints a progress bar for loading the weights to
  stderr.

## Tests

```
pip install -e '.[test]' && pytest tests
```

190 tests, CPU only, no model download, no network.
`tests/test_developer_replay.py` replays recorded matrices bit for bit as a
check of the arithmetic; it is not what verifying the paper means. The wheel
is built from `src/cruxes` alone, so the test fixtures (described in
`tests/fixtures/README.md`) are test only and are not installed, and neither
is the worked example.

## Citation

```bibtex
@misc{reber2026cruxes,
  title  = {Can {LLMs} Anticipate Their Cruxes?},
  author = {Reber, David and Reber, James and Holtzman, Ari and Veitch, Victor},
  year   = {2026},
  note   = {arXiv preprint, forthcoming}
}
```

## License

MIT. See `LICENSE`.
