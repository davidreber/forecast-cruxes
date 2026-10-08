"""Score caches that say what produced them and refuse to be overwritten.

A score cache is expensive to produce and cheap to destroy. On 2026-04-01 a job
wrote files named after a question, a contrast count and a seed on top of
files with the same names produced from different premises, twenty minutes
after a gather had read them. Nothing looked wrong, because the filename never
recorded which statements were scored.

Three rules follow, and this module is nothing but those rules.

The key is exactly what influences a score: the selected premise texts in
scoring order (one ``items`` list, whatever pools they came from), the two
instruction texts, the system message and the prompt template, the model name
and revision, the batch size, the truncation length and the precision. Nothing
else. The question id, the contrast count, the seed and the pool boundaries
are not in it, since they change no score: renaming a question costs no
rescore, and two questions that select the same premises share one cache. The
pool split is recorded in the file as ``pools`` for whoever reads it without a
plan, and a caller who knows their pools passes them when loading.

The filename is ``scores_{digest}.json``, a digest of the key, and the full key
is written inside the file, so a cache found on disk years later says what it
is without a log or a naming convention.

A write never overwrites. The file is created exclusively and an existing path
raises. Before a cache is reused its key is read back and compared with the
one wanted; the filename alone is never trusted.

Hardware and library versions are recorded in each file's ``environment``
block and are deliberately not part of the key: they do move scores slightly,
and that movement is what a tolerance is for, not a reason a cache from
another machine should be unreadable.

This module imports json, hashlib and numpy. Reading a cache is a CPU act.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .matching import DEFAULT_MATCHING_CRITERION, MatchingCriterion
from .reranker.model import (
    DEFAULT_BATCH_SIZE,
    RERANKER_DTYPE,
    RERANKER_MAX_LENGTH,
    RERANKER_MODEL_NAME,
    RERANKER_REVISION,
)
from .reranker.prompt import DEFAULT_RERANKER_PROMPT, RerankerPrompt
from .scoring import SCORE_KEYS, ScoreMatrices

__all__ = [
    "CACHE_FORMAT_VERSION",
    "READABLE_FORMAT_VERSIONS",
    "DIGEST_LENGTH",
    "pool_item_ids",
    "ScoreCacheKey",
    "build_score_cache_key",
    "cache_digest",
    "cache_filename",
    "cache_path",
    "write_score_cache",
    "read_score_cache",
    "load_score_matrices",
    "find_cached_scores",
]

#: Written into every file. Version 3 keys on one ``items`` list; version 2
#: keyed on two pool lists and is still read (its items are the two lists
#: concatenated, which gives identical numbers), never written.
CACHE_FORMAT_VERSION = 3
READABLE_FORMAT_VERSIONS = (2, 3)

#: Hex characters of the SHA-256 digest used in the filename. Sixty four bits;
#: a collision would be caught anyway, since the key inside is compared.
DIGEST_LENGTH = 16


@dataclass(frozen=True)
class ScoreCacheKey:
    """Everything that determines the numbers in one score cache, and no more."""

    items: tuple
    positive_instruction: str
    negative_instruction: str
    system_message: str
    prompt_template: str
    model_name: str
    model_revision: str
    batch_size: int
    max_length: int
    dtype: str

    def __post_init__(self):
        object.__setattr__(self, "items", tuple(self.items))
        bad = [i for i, text in enumerate(self.items) if not isinstance(text, str)]
        if bad:
            raise ValueError(f"cache key items {bad[:5]} are not strings")

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def premises(self) -> list:
        """Every scored premise in scoring order, as a list."""
        return list(self.items)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["items"] = list(self.items)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ScoreCacheKey":
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"cache key has fields this version does not know: {sorted(unknown)}")
        return cls(**{key: data[key] for key in cls.__dataclass_fields__})

    def canonical_json(self) -> bytes:
        """The exact bytes the digest is taken over.

        Sorted keys, no insignificant whitespace, UTF-8 without escaping, so
        the digest depends on the values and not on how a writer spelled them.
        """
        return json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")


def build_score_cache_key(
    items: list,
    criterion: MatchingCriterion = DEFAULT_MATCHING_CRITERION,
    prompt: RerankerPrompt = DEFAULT_RERANKER_PROMPT,
    model_name: str = RERANKER_MODEL_NAME,
    model_revision: str = RERANKER_REVISION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = RERANKER_MAX_LENGTH,
    dtype: str = RERANKER_DTYPE,
) -> ScoreCacheKey:
    """The key for scoring these selected premises, in this order, under these settings.

    ``items`` is every premise to score in scoring order; the pipeline passes
    the introspected selection followed by the extracted one. Both commands
    call this, which is what lets the CPU stage find a cache without a naming
    convention, an index file or a symlink: it recomputes the key from the
    same inputs and asks for that path.
    """
    return ScoreCacheKey(
        items=items,
        positive_instruction=criterion.positive_instruction,
        negative_instruction=criterion.negative_instruction,
        system_message=prompt.system_message,
        prompt_template=prompt.template,
        model_name=model_name,
        model_revision=model_revision,
        batch_size=int(batch_size),
        max_length=int(max_length),
        dtype=str(dtype),
    )


def cache_digest(key: ScoreCacheKey) -> str:
    """Full SHA-256 digest of a cache key, as hex."""
    return hashlib.sha256(key.canonical_json()).hexdigest()


def cache_filename(key: ScoreCacheKey) -> str:
    """``scores_{digest}.json``. No question id: a rename must not cost a rescore."""
    return f"scores_{cache_digest(key)[:DIGEST_LENGTH]}.json"


def cache_path(directory, key: ScoreCacheKey) -> Path:
    """Where this key's cache lives inside ``directory``."""
    return Path(directory) / cache_filename(key)


def pool_item_ids(pools: dict) -> list:
    """``pool:index`` labels for an ordered ``{pool: size}`` mapping."""
    return [f"{pool}:{i}" for pool, size in pools.items() for i in range(int(size))]


def _check_pools(pools, n_items: int) -> dict:
    pools = {str(name): int(size) for name, size in dict(pools).items()}
    if any(size < 0 for size in pools.values()):
        raise ValueError(f"pool sizes must be non negative: {pools}")
    if sum(pools.values()) != n_items:
        raise ValueError(f"pools {pools} sum to {sum(pools.values())}, the key has {n_items} items")
    return pools


def _checked_matrices(matrices: dict, n_items: int) -> dict:
    checked = {}
    for name in SCORE_KEYS:
        if name not in matrices:
            raise ValueError(f"matrices are missing {name}")
        array = np.asarray(matrices[name], dtype=float)
        if array.shape != (n_items, n_items):
            raise ValueError(
                f"{name} has shape {array.shape}, expected ({n_items}, {n_items}) from the key's items"
            )
        bad = np.argwhere(~np.isfinite(array))
        if len(bad):
            i, j = (int(x) for x in bad[0])
            raise ValueError(
                f"{name}[{i}, {j}] is {array[i, j]!r} and {len(bad)} entries in total are not "
                "finite. An unscored pair is an error: rescore rather than write a guess."
            )
        checked[name] = array
    return checked


def write_score_cache(
    directory,
    key: ScoreCacheKey,
    matrices: dict,
    environment: dict | None = None,
    pools: dict | None = None,
) -> Path:
    """Write one set of four raw score matrices. Never overwrites.

    ``pools`` is the ordered ``{pool name: size}`` split of the key's items,
    recorded in the file for a reader without a plan; it is not part of the
    key. Raises ``FileExistsError`` if the path is taken, whatever is in it,
    and ``ValueError`` if any score is missing, the shapes disagree with the
    key, or the pools do not add up to the items. The directory is created
    if it does not exist.
    """
    checked = _checked_matrices(matrices, key.n_items)
    recorded_pools = _check_pools(pools, key.n_items) if pools is not None else None
    path = cache_path(directory, key)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "key": key.to_dict(),
        "digest": cache_digest(key),
        "n_items": key.n_items,
        "pools": recorded_pools,
        "item_ids": pool_item_ids(recorded_pools) if recorded_pools else None,
        "environment": dict(environment or {}),
    }
    for name in SCORE_KEYS:
        payload[name] = checked[name].tolist()

    # Exclusive creation. This one character is the whole of the April fix.
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
    return path


def read_score_cache(path) -> dict:
    """Read one cache file and check its recorded digest against its key.

    A file whose digest does not match the key it carries has been edited or
    truncated, and is refused rather than used. Returns the payload with
    ``key`` as a :class:`ScoreCacheKey`.
    """
    with open(Path(path), encoding="utf-8") as handle:
        payload = json.load(handle)
    version = payload.get("cache_format_version")
    if version not in READABLE_FORMAT_VERSIONS:
        raise ValueError(
            f"{path} has cache format {version!r}, this version reads {READABLE_FORMAT_VERSIONS}"
        )
    if version == 2:
        return _read_version_2(path, payload)
    key = ScoreCacheKey.from_dict(payload["key"])
    actual = cache_digest(key)
    if payload.get("digest") != actual:
        raise ValueError(
            f"{path} records digest {payload.get('digest')} but its key hashes to {actual}"
        )
    payload["key"] = key
    return payload


def _read_version_2(path, payload: dict) -> dict:
    """A version 2 file: two pool lists in the key. Its digest is checked over
    the version 2 key, then the key is returned as a version 3 key whose items
    are the two lists concatenated, with the pools recorded beside it."""
    recorded = payload["key"]
    version_2_json = json.dumps(
        recorded, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    actual = hashlib.sha256(version_2_json).hexdigest()
    if payload.get("digest") != actual:
        raise ValueError(
            f"{path} records digest {payload.get('digest')} but its key hashes to {actual}"
        )
    introspected, extracted = list(recorded["introspected"]), list(recorded["extracted"])
    fields = {k: v for k, v in recorded.items() if k not in ("introspected", "extracted")}
    payload["key"] = ScoreCacheKey(items=introspected + extracted, **fields)
    payload["pools"] = {"introspected": len(introspected), "extracted": len(extracted)}
    payload["n_items"] = len(introspected) + len(extracted)
    return payload


def load_score_matrices(path, pools: dict | None = None) -> ScoreMatrices:
    """Read a cache into the CPU core's container, refusing any missing score.

    The container holds the two pools of the set inclusion measurement, so
    ``pools`` must be ``{"introspected": n, "extracted": m}`` in that order,
    adding up to the key's items. A caller with a plan passes its own; without
    one the pools recorded in the file are used, and a file that recorded
    none cannot be loaded this way.
    """
    payload = read_score_cache(path)
    if pools is None:
        pools = payload.get("pools")
        if not pools:
            raise ValueError(f"{path} records no pools; pass pools= to load it")
    pools = _check_pools(pools, payload["key"].n_items)
    if list(pools) != ["introspected", "extracted"]:
        raise ValueError(
            f"ScoreMatrices holds the pools introspected and extracted, in that order; got {list(pools)}"
        )
    return ScoreMatrices(
        n_introspected=pools["introspected"],
        n_extracted=pools["extracted"],
        **{name: payload[name] for name in SCORE_KEYS},
    )


def find_cached_scores(directory, key: ScoreCacheKey):
    """The path of this key's cache if one is on disk with this key inside.

    None when no file is there. The path when a file is there and carries
    exactly this key. Raises when a file is there and carries a different key,
    which is a digest collision or a file written by something else, and is not
    something to work around.
    """
    path = cache_path(directory, key)
    if not path.exists():
        return None
    existing = read_score_cache(path)["key"]
    if existing != key:
        raise ValueError(
            f"{path} exists but holds a different key: the digest collided or the "
            "file was written by something else. Move it aside and rerun."
        )
    return path
