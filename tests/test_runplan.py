"""What the two commands share: parsed arguments and the key of every question.

New module in the restructure. cruxes.reranker.score (GPU) and cruxes.report
(CPU) both call add_shared_arguments, settings_from_args and plan here, and
must compute the same cache key for a question from the same arguments or the
CPU command can never find what the GPU command wrote.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cruxes.matching import DEFAULT_MATCHING_CRITERION
from cruxes.reranker.model import RERANKER_MODEL_NAME, RERANKER_REVISION
from cruxes.reranker.prompt import DEFAULT_RERANKER_PROMPT
from cruxes.reranker.score import build_parser as gpu_parser
from cruxes.report import build_parser as cpu_parser
from cruxes.runplan import plan, settings_from_args

QUESTIONS = {
    "questions": [
        {
            "question_id": "second-question",
            "introspected": [f"i{i}" for i in range(8)],
            "extracted": [f"e{i}" for i in range(3)],
        },
        {
            "question_id": "first-question",
            "introspected": [f"I{i}" for i in range(4)],
            "extracted": [f"E{i}" for i in range(9)],
        },
    ]
}


def write_premises(tmp_path: Path, data=None) -> Path:
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(data if data is not None else QUESTIONS))
    return path


def test_settings_from_args_uses_the_package_defaults(tmp_path):
    premises = write_premises(tmp_path)
    args = cpu_parser().parse_args(
        ["--premises", str(premises), "--cache-dir", str(tmp_path), "--out-dir", str(tmp_path)]
    )
    settings = settings_from_args(args)
    assert settings.criterion == DEFAULT_MATCHING_CRITERION
    assert settings.prompt == DEFAULT_RERANKER_PROMPT
    assert settings.model_name == RERANKER_MODEL_NAME
    assert settings.model_revision == RERANKER_REVISION


def test_plan_sorts_questions_by_id_and_attaches_a_key(tmp_path):
    premises = write_premises(tmp_path)
    args = cpu_parser().parse_args(
        ["--premises", str(premises), "--cache-dir", str(tmp_path), "--out-dir", str(tmp_path)]
    )
    planned = plan(premises, settings_from_args(args))
    assert [p.premise_set.question_id for p in planned] == ["first-question", "second-question"]
    assert all(p.key is not None for p in planned)


def test_plan_raises_on_an_unknown_question_id(tmp_path):
    premises = write_premises(tmp_path)
    args = cpu_parser().parse_args(
        ["--premises", str(premises), "--cache-dir", str(tmp_path), "--out-dir", str(tmp_path)]
    )
    with pytest.raises(ValueError, match="not-a-real-question"):
        plan(premises, settings_from_args(args), questions=["not-a-real-question"])


def test_the_gpu_and_cpu_commands_compute_the_same_keys(tmp_path):
    """The GPU stage writes a path the CPU stage can recompute without a listing."""
    premises = write_premises(tmp_path)
    gpu_args = gpu_parser().parse_args(["--premises", str(premises), "--cache-dir", str(tmp_path)])
    cpu_args = cpu_parser().parse_args(
        ["--premises", str(premises), "--cache-dir", str(tmp_path), "--out-dir", str(tmp_path)]
    )
    gpu_planned = plan(premises, settings_from_args(gpu_args))
    cpu_planned = plan(premises, settings_from_args(cpu_args))
    assert [p.key for p in gpu_planned] == [p.key for p in cpu_planned]


def test_a_matching_criterion_file_changes_the_key(tmp_path):
    premises = write_premises(tmp_path)
    criterion_path = tmp_path / "criterion.json"
    criterion_path.write_text(
        json.dumps({"positive_instruction": "same?", "negative_instruction": "different?"})
    )
    default_args = cpu_parser().parse_args(
        ["--premises", str(premises), "--cache-dir", str(tmp_path), "--out-dir", str(tmp_path)]
    )
    custom_args = cpu_parser().parse_args(
        [
            "--premises", str(premises), "--cache-dir", str(tmp_path), "--out-dir", str(tmp_path),
            "--matching-criterion", str(criterion_path),
        ]
    )
    default_keys = [p.key for p in plan(premises, settings_from_args(default_args))]
    custom_keys = [p.key for p in plan(premises, settings_from_args(custom_args))]
    assert default_keys != custom_keys


def test_a_reranker_prompt_file_changes_the_key(tmp_path):
    premises = write_premises(tmp_path)
    prompt_path = tmp_path / "prompt.json"
    prompt_path.write_text(
        json.dumps(
            {"system_message": "custom", "template": "{system}{instruction}{query}{document}"}
        )
    )
    default_args = cpu_parser().parse_args(
        ["--premises", str(premises), "--cache-dir", str(tmp_path), "--out-dir", str(tmp_path)]
    )
    custom_args = cpu_parser().parse_args(
        [
            "--premises", str(premises), "--cache-dir", str(tmp_path), "--out-dir", str(tmp_path),
            "--reranker-prompt", str(prompt_path),
        ]
    )
    default_keys = [p.key for p in plan(premises, settings_from_args(default_args))]
    custom_keys = [p.key for p in plan(premises, settings_from_args(custom_args))]
    assert default_keys != custom_keys


def test_gpu_dry_run_does_not_import_torch(tmp_path):
    """--dry-run is the only part of the GPU command that works without a GPU."""
    premises = write_premises(tmp_path)
    probe = f"""
import sys
from cruxes.reranker.score import main
status = main(["--premises", {str(premises)!r}, "--cache-dir", {str(tmp_path)!r}, "--dry-run"])
assert status == 0, status
assert "torch" not in sys.modules, sorted(sys.modules)
print("ok")
"""
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
