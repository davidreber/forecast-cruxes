"""A cache says what produced it, and a write never destroys one.

These tests are the April incident written down. A job wrote files named after
a question, a contrast count and a seed on top of files of the same names
produced from different premises. The names could not tell the two apart and
the write could not refuse.

The key no longer carries a question_id, a max_contrasts_per_side or a seed:
none of those change a score, so a filename built from them let two different
computations collide. The filename is now scores_{digest}.json, a digest of
exactly what does influence a score.
"""

import dataclasses
import json

import numpy as np
import pytest

from cruxes.matching import MatchingCriterion
from cruxes.reranker.prompt import RerankerPrompt
from cruxes.scorecache import (
    CACHE_FORMAT_VERSION,
    DIGEST_LENGTH,
    ScoreCacheKey,
    build_score_cache_key,
    cache_digest,
    cache_filename,
    cache_path,
    find_cached_scores,
    load_score_matrices,
    read_score_cache,
    write_score_cache,
)
from cruxes.scoring import item_ids

BASE_FIELDS = dict(
    items=("premise one", "premise two", "premise three"),
    positive_instruction="the same question?",
    negative_instruction="a different question?",
    system_message="answer yes or no",
    prompt_template="<s>{system}{instruction}{query}{document}",
    model_name="Qwen/Qwen3-Reranker-8B",
    model_revision="5fa94080caafeaa45a15d11f969d7978e087a3db",
    batch_size=32,
    max_length=8192,
    dtype="float16",
)


#: The pool split the key's items came from; recorded in the file, not keyed.
POOLS = {"introspected": 2, "extracted": 1}


def make_key(**overrides) -> ScoreCacheKey:
    return ScoreCacheKey(**{**BASE_FIELDS, **overrides})


def make_matrices(n: int = 3) -> dict:
    rng = np.random.default_rng(0)
    raw = rng.random((n, n))
    return {
        "pos_fwd": raw,
        "pos_rev": raw.T,
        "neg_fwd": 1 - raw,
        "neg_rev": (1 - raw).T,
    }


def test_the_digest_is_stable_across_processes():
    """Not Python's builtin hash, which is salted per interpreter."""
    import subprocess
    import sys

    probe = (
        "from cruxes.scorecache import ScoreCacheKey, cache_digest;"
        f"print(cache_digest(ScoreCacheKey(**{BASE_FIELDS!r})))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == cache_digest(make_key())


@pytest.mark.parametrize(
    "field, value",
    [
        ("items", ("premise one", "premise two edited", "premise three")),
        ("items", ("premise two", "premise one", "premise three")),
        ("items", ("premise one", "premise two", "premise three", "premise four")),
        ("positive_instruction", "the same question？"),
        ("negative_instruction", "really different?"),
        ("system_message", "answer YES or NO"),
        ("prompt_template", "<s>{system}{instruction}{query}{document} "),
        ("model_name", "Qwen/Qwen3-Reranker-0.6B"),
        ("model_revision", "77d193c791ed757ca307ee72715aa132723da912"),
        ("batch_size", 1),
        ("max_length", 4096),
        ("dtype", "float32"),
    ],
)
def test_every_field_changes_the_digest(field, value):
    """Including the premise texts and their order, which is the whole point."""
    assert cache_digest(make_key(**{field: value})) != cache_digest(make_key())


def test_the_parametrization_covers_every_dataclass_field():
    """A field added later without a digest test would be silently ignored."""
    covered = {
        "items",
        "positive_instruction",
        "negative_instruction",
        "system_message",
        "prompt_template",
        "model_name",
        "model_revision",
        "batch_size",
        "max_length",
        "dtype",
    }
    assert {f.name for f in dataclasses.fields(ScoreCacheKey)} == covered


def test_the_key_has_no_question_id_max_contrasts_or_seed():
    fields = {f.name for f in dataclasses.fields(ScoreCacheKey)}
    assert "question_id" not in fields
    assert "max_contrasts_per_side" not in fields
    assert "seed" not in fields


def test_two_questions_with_the_same_selected_premises_share_one_filename():
    """A rename must not cost a rescore."""
    key_for_first_question = build_score_cache_key(list(BASE_FIELDS["items"]))
    key_for_renamed_question = build_score_cache_key(list(BASE_FIELDS["items"]))
    assert cache_filename(key_for_first_question) == cache_filename(key_for_renamed_question)


def test_build_score_cache_key_uses_the_criterion_and_prompt():
    criterion = MatchingCriterion("custom", "pos", "neg")
    prompt = RerankerPrompt("system", "{system}{instruction}{query}{document}")
    default_key = build_score_cache_key(["a", "b"])
    custom_key = build_score_cache_key(["a", "b"], criterion=criterion, prompt=prompt)
    assert custom_key.positive_instruction == "pos"
    assert custom_key.system_message == "system"
    assert cache_digest(custom_key) != cache_digest(default_key)


def test_the_filename_is_scores_and_a_digest_with_no_question_id():
    name = cache_filename(make_key())
    assert name.startswith("scores_")
    assert name.endswith(".json")
    digest = name[len("scores_") : -len(".json")]
    assert len(digest) == DIGEST_LENGTH
    assert digest == cache_digest(make_key())[:DIGEST_LENGTH]


def test_round_trip_through_a_file(tmp_path):
    key = make_key()
    matrices = make_matrices()
    path = write_score_cache(
        tmp_path, key, matrices, environment={"gpu": "test"}, pools=POOLS
    )

    assert path == cache_path(tmp_path, key)
    payload = read_score_cache(path)
    assert payload["key"] == key
    assert payload["pools"] == POOLS
    assert payload["item_ids"] == item_ids(2, 1) == ["introspected:0", "introspected:1", "extracted:0"]
    assert payload["environment"] == {"gpu": "test"}
    assert payload["cache_format_version"] == CACHE_FORMAT_VERSION
    for name, expected in matrices.items():
        assert np.allclose(np.array(payload[name]), expected)


def test_the_cache_loads_into_the_cpu_core_container(tmp_path):
    key = make_key()
    path = write_score_cache(tmp_path, key, make_matrices(), pools=POOLS)
    loaded = load_score_matrices(path)
    assert loaded.n_introspected == 2
    assert loaded.n_extracted == 1
    assert loaded.item_ids == ["introspected:0", "introspected:1", "extracted:0"]
    assert loaded.pos_fwd.shape == (3, 3)
    assert load_score_matrices(path, POOLS).n_extracted == 1


def test_the_pool_split_is_outside_the_key(tmp_path):
    """Two questions that score the same items in different pool splits share a cache."""
    path = write_score_cache(tmp_path, make_key(), make_matrices(), pools=POOLS)
    other_split = {"introspected": 1, "extracted": 2}
    loaded = load_score_matrices(path, other_split)
    assert (loaded.n_introspected, loaded.n_extracted) == (1, 2)
    assert find_cached_scores(tmp_path, make_key()) == path


def test_pools_that_do_not_add_up_are_refused(tmp_path):
    with pytest.raises(ValueError, match="sum to"):
        write_score_cache(tmp_path, make_key(), make_matrices(), pools={"introspected": 2, "extracted": 2})
    path = write_score_cache(tmp_path, make_key(), make_matrices(), pools=POOLS)
    with pytest.raises(ValueError, match="sum to"):
        load_score_matrices(path, {"introspected": 3, "extracted": 1})
    with pytest.raises(ValueError, match="introspected and extracted"):
        load_score_matrices(path, {"facts": 2, "considerations": 1})


def test_a_cache_written_without_pools_needs_them_to_load(tmp_path):
    path = write_score_cache(tmp_path, make_key(), make_matrices())
    assert read_score_cache(path)["pools"] is None
    with pytest.raises(ValueError, match="records no pools"):
        load_score_matrices(path)
    assert load_score_matrices(path, POOLS).n_introspected == 2


def test_a_version_2_cache_is_read_as_the_concatenated_items(tmp_path):
    """Format version 2 caches keyed on two pool lists; their numbers are unchanged."""
    import hashlib

    version_2_key = {k: v for k, v in BASE_FIELDS.items() if k != "items"}
    version_2_key["introspected"] = ["premise one", "premise two"]
    version_2_key["extracted"] = ["premise three"]
    canonical = json.dumps(version_2_key, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    matrices = make_matrices()
    payload = {
        "cache_format_version": 2,
        "key": version_2_key,
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "n_introspected": 2,
        "n_extracted": 1,
        "item_ids": item_ids(2, 1),
        "environment": {},
        **{name: matrices[name].tolist() for name in matrices},
    }
    path = tmp_path / "scores_version2.json"
    path.write_text(json.dumps(payload))
    read = read_score_cache(path)
    assert read["key"] == make_key()
    assert read["pools"] == POOLS
    loaded = load_score_matrices(path)
    assert (loaded.n_introspected, loaded.n_extracted) == (2, 1)
    assert np.allclose(loaded.pos_fwd, matrices["pos_fwd"])
    payload["digest"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        read_score_cache(path)


def test_a_second_write_to_the_same_path_raises(tmp_path):
    """The April incident, in one assertion. There is no force flag."""
    key = make_key()
    write_score_cache(tmp_path, key, make_matrices())
    with pytest.raises(FileExistsError):
        write_score_cache(tmp_path, key, make_matrices())


def test_the_first_file_survives_the_refused_write(tmp_path):
    key = make_key()
    first = make_matrices()
    path = write_score_cache(tmp_path, key, first)
    second = {name: array * 0.5 for name, array in first.items()}
    with pytest.raises(FileExistsError):
        write_score_cache(tmp_path, key, second)
    assert np.allclose(np.array(read_score_cache(path)["pos_fwd"]), first["pos_fwd"])


def test_different_premises_get_different_paths(tmp_path):
    """What the recorded naming could not do."""
    original = make_key()
    replacement = make_key(items=("a shorter premise list", "premise three"))
    write_score_cache(tmp_path, original, make_matrices())
    write_score_cache(tmp_path, replacement, make_matrices(2))
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_a_matrix_of_the_wrong_size_is_refused(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        write_score_cache(tmp_path, make_key(), make_matrices(4))


def test_a_non_finite_matrix_is_refused(tmp_path):
    matrices = make_matrices()
    matrices["pos_fwd"][0, 1] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        write_score_cache(tmp_path, make_key(), matrices)


def test_find_cached_scores_reports_presence(tmp_path):
    key = make_key()
    assert find_cached_scores(tmp_path, key) is None
    path = write_score_cache(tmp_path, key, make_matrices())
    assert find_cached_scores(tmp_path, key) == path


def test_a_file_carrying_a_different_key_is_an_error_not_a_reuse(tmp_path):
    key = make_key()
    path = write_score_cache(tmp_path, key, make_matrices())
    payload = json.loads(path.read_text())
    payload["key"]["items"] = ["something else entirely", "premise two", "premise three"]
    payload["digest"] = cache_digest(ScoreCacheKey.from_dict(payload["key"]))
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different key"):
        find_cached_scores(tmp_path, key)


def test_a_file_whose_digest_does_not_match_its_key_is_refused(tmp_path):
    path = write_score_cache(tmp_path, make_key(), make_matrices())
    payload = json.loads(path.read_text())
    payload["key"]["batch_size"] = 999
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hashes to"):
        read_score_cache(path)


def test_a_wrong_cache_format_version_is_refused(tmp_path):
    path = write_score_cache(tmp_path, make_key(), make_matrices())
    payload = json.loads(path.read_text())
    payload["cache_format_version"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="cache format"):
        read_score_cache(path)
