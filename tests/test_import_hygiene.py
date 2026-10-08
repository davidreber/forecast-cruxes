"""The package must not reach a model, a GPU or a network.

This is the test the whole slice exists for. Importing the CPU only report
must never pull in torch, and no module may mention the vocabulary this
restructure retired.

The check is run in a subprocess, because by the time this test file is
collected pytest may already have imported something heavy for another
reason.
"""

import subprocess
import sys

FORBIDDEN = (
    "torch",
    "transformers",
    "networkx",
    "openai",
    "langchain",
    "langchain_openai",
    "dotenv",
    "sklearn",
    "pandas",
    "matplotlib",
)

_PROBE = """
import sys
import cruxes
loaded = sorted(
    name for name in sys.modules
    if name.split(".")[0] in {forbidden!r}
)
print(",".join(loaded))
"""


def test_importing_the_package_loads_nothing_heavy():
    """Importing cruxes must not drag in a model runtime or a client."""
    probe = _PROBE.format(forbidden=set(FORBIDDEN))
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = [name for name in completed.stdout.strip().split(",") if name]
    assert loaded == [], f"cruxes pulled in {loaded}"


#: Every module a CPU only install has to be able to import. ``reranker.model``
#: and ``reranker.score`` are on the list on purpose: ``model`` holds the
#: pinned revision and the batch size, which the CPU stage needs to rebuild a
#: cache key, and ``score`` (the GPU command) must support ``--dry-run`` on a
#: machine with no GPU. Their torch imports are inside the functions that
#: score, not at the top of the file.
CPU_IMPORTABLE_MODULES = (
    "cruxes",
    "cruxes.premises",
    "cruxes.scorecache",
    "cruxes.matching",
    "cruxes.runplan",
    "cruxes.report",
    "cruxes.reranker.prompt",
    "cruxes.reranker.model",
    "cruxes.reranker.score",
)

_MODULE_PROBE = """
import importlib, sys
importlib.import_module({module!r})
loaded = sorted(
    name for name in sys.modules
    if name.split(".")[0] in {forbidden!r}
)
print(",".join(loaded))
"""


def test_the_scoring_stage_modules_import_without_torch():
    """A CPU only install must be able to read a prompt and build a cache key."""
    for module in CPU_IMPORTABLE_MODULES:
        probe = _MODULE_PROBE.format(module=module, forbidden=set(FORBIDDEN))
        completed = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        loaded = [name for name in completed.stdout.strip().split(",") if name]
        assert loaded == [], f"{module} pulled in {loaded}"


def test_no_module_imports_torch_at_the_top_level():
    """torch may be imported inside a function, never at module scope.

    An indented import is paid for only by a caller that asked to score. A
    module level one is paid for by everyone who imports anything nearby.
    """
    import pathlib

    import cruxes

    package_dir = pathlib.Path(cruxes.__file__).parent
    offenders = {}
    for path in sorted(package_dir.rglob("*.py")):
        for line in path.read_text().splitlines():
            if line[:1].isspace() or not line.startswith(("import ", "from ")):
                continue
            head = line.split()[1].split(".")[0]
            if head in FORBIDDEN:
                offenders.setdefault(str(path.relative_to(package_dir)), []).append(line)
    assert offenders == {}, offenders


def test_only_numpy_and_scipy_are_imported_by_name():
    """Every module in the package imports numpy, scipy, the standard library or
    itself, and nothing else, at any depth of the package.

    An allowlist, not a denylist: a new hard dependency that transformers
    happens to install (tqdm, requests) would pass a denylist and break
    ``pip install forecast-cruxes`` for a CPU only user. Module level only;
    the function level torch and transformers imports of the GPU stage are
    covered by the subprocess probes above.
    """
    import ast
    import pathlib

    import cruxes

    allowed = {"numpy", "scipy", "cruxes"} | set(sys.stdlib_module_names)
    package_dir = pathlib.Path(cruxes.__file__).parent
    offenders = {}
    for path in sorted(package_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:  # module level only; function level imports are the probes' job
            if isinstance(node, ast.Import):
                heads = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import inside the package
                    continue
                heads = [node.module.split(".")[0]]
            else:
                continue
            for head in heads:
                if head not in allowed:
                    offenders.setdefault(str(path.relative_to(package_dir)), []).append(head)
    assert offenders == {}, offenders


def test_the_figures_subpackage_is_gone():
    """verify/ and the figure scripts left the package for the verification repo."""
    import pathlib

    import cruxes

    package_dir = pathlib.Path(cruxes.__file__).parent
    assert not (package_dir / "figures").exists()


#: Vocabulary the restructure retired. "pos_fwd" itself must never be in this
#: list: it is a real matrix name (cruxes.scoring.SCORE_KEYS) and appears
#: throughout the source legitimately.
RETIRED_VOCABULARY = (
    "matchup",
    "March Madness",
    "apriori",
    "aposteriori",
    "side_a",
    "side_b",
    "legacy_constant_seed",
    "pos_fwd_method",
    "debiased",
)


def test_no_source_file_mentions_the_retired_vocabulary():
    import pathlib

    import cruxes

    package_dir = pathlib.Path(cruxes.__file__).parent
    offenders = {}
    for path in sorted(package_dir.rglob("*.py")):
        lowered = path.read_text().lower()
        hits = [word for word in RETIRED_VOCABULARY if word.lower() in lowered]
        if hits:
            offenders[str(path.relative_to(package_dir))] = hits
    assert offenders == {}, offenders
