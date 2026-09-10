"""Deduplication: arXiv version collapsing + chunk-level near-duplicate detection.

Implements decisions/0004-arxiv-base-id-deduplication.md. Two independent
mechanisms, both exact-match-plus-near-match halves of the same problem:

1. ``resolve_versions`` collapses ``2401.01234v1`` .. ``v5`` to one document,
   keyed by base id, keeping the latest version live. This is the *document*
   axis: without it, top-k fills with five copies of one paper and recall@k
   stops measuring the retriever.
2. ``ChunkDeduplicator`` collapses near-identical chunk *text* -- e.g. a
   related-work paragraph quoted verbatim across many papers -- using MinHash
   + LSH over word shingles. This is the *chunk* axis: two chunks can be
   near-duplicates even when their documents are unrelated.

Neither mechanism deletes anything. Both only annotate: a resolved document
keeps ``versions_seen``; a resolved chunk keeps a ``dup_of`` pointer. Deciding
what to *do* with that annotation (drop from the index, downweight at
ranking, etc.) is a retrieval-time or pipeline concern, not this module's.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

from datasketch import MinHash, MinHashLSH
from pydantic import BaseModel, Field

from selfrag.ids import canonical_arxiv_id, config_hash

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


class DedupConfig(BaseModel):
    """Configuration for both halves of dedup, hashed into ``dedup_config_id``.

    ``dedup_config_id`` is shaped exactly like ``registry.Component.config_id``
    (``"<name>.v<version>@<8 hex>"``) and follows the same rules for the same
    reasons: it is fed into ``RunManifest.dedup_config_id``, so it must be
    *stable* (same values -> same id, forever) and *sensitive* (any value
    change -> a different id). A dedup config is not registered in
    ``selfrag.registry`` -- there is exactly one implementation, not a family
    of interchangeable ones -- but it still needs a ``name``/``version`` pair
    for the same reason ``Component`` does: a future change to *how* dedup
    works (not just its parameters) must be able to invalidate old ids by
    bumping ``version``, even if every field value stays byte-identical.
    """

    name: ClassVar[str] = "dedup"
    version: ClassVar[int] = 1

    minhash_num_perm: int = Field(
        128,
        gt=0,
        description="MinHash permutation count. Higher = tighter Jaccard estimate, more memory/time.",
    )
    jaccard_threshold: float = Field(
        0.85,
        gt=0.0,
        le=1.0,
        description="Minimum estimated shingle-set Jaccard similarity to call two chunks near-duplicates.",
    )
    shingle_size: int = Field(
        5,
        gt=0,
        description="Word-shingle width used to build MinHash signatures.",
    )
    enabled: bool = True

    @property
    def dedup_config_id(self) -> str:
        """Short deterministic id: ``"dedup.v<version>@<8 hex>"``.

        Hashes the *validated* config (post-default-application), not the raw
        input dict, so a config that only omits fields left at their defaults
        collapses to the same id as one that spells them out -- mirroring
        ``registry.Component.config_id``.
        """
        payload = {
            "name": self.name,
            "version": self.version,
            "config": self.model_dump(mode="json"),
        }
        digest = config_hash(payload)[:8]
        return f"{self.name}.v{self.version}@{digest}"


# -----------------------------------------------------------------------------
# Version resolution
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VersionedRecord:
    """One raw input identifying an arXiv paper, plus opaque caller data.

    ``payload`` is never inspected here -- it rides along unchanged and comes
    back attached to whichever record wins as the live version. This lets
    ``resolve_versions`` stay decoupled from ``selfrag.schema.Document``
    (constructing a real ``Document`` needs title/authors/etc. that this
    module has no business knowing about); the caller attaches whatever it
    already has -- a parsed ``Document``, a raw metadata dict, a file path --
    and gets it back on the winning record.
    """

    raw_id: str
    payload: Any = None


@dataclass(frozen=True, slots=True)
class ResolvedDocument:
    """The live version of one arXiv paper after ``resolve_versions``."""

    base_id: str
    version: int | None
    versions_seen: list[int]
    raw_id: str
    payload: Any = None


def _version_sort_key(version: int | None) -> tuple[int, int]:
    """Sort key: unversioned sorts below every explicit version.

    ``(1, version)`` beats ``(0, 0)`` for any int ``version`` (including 0),
    and among explicit versions the second element compares *numerically*,
    which is the whole point -- ``v10`` must outrank ``v9``, and string
    comparison would get that backwards.
    """
    if version is None:
        return (0, 0)
    return (1, version)


def resolve_versions(
    records: Iterable[VersionedRecord],
) -> tuple[list[ResolvedDocument], dict[str, str]]:
    """Group records by arXiv base id; keep the latest version as live.

    Args:
        records: an iterable of ``VersionedRecord``.

    Returns:
        A pair ``(resolved, id_to_base)``:

        - ``resolved``: one ``ResolvedDocument`` per distinct base id, in
          first-seen order, carrying the winning (latest-version) record's
          ``raw_id``/``payload`` plus the full sorted ``versions_seen``.
        - ``id_to_base``: every input ``raw_id`` (exactly as given, not
          normalised) mapped to its resolved base id, so a caller holding
          references keyed by the original ids (citations, qrels staged
          before resolution, etc.) can rewrite them.

    Raises:
        ValueError: propagated from ``canonical_arxiv_id`` on any
            unrecognised id. Not caught here on purpose: a silently
            mis-parsed id would corrupt the document key space (see
            decisions/0004), and grouping is exactly the place that failure
            has to surface, before it can hide inside a merged group.

    Ties (two records resolving to the same base id *and* the same version,
    e.g. the same paper fetched twice from two sources) are broken by
    keeping whichever was seen first in ``records`` -- deterministic given a
    fixed input order, and consistent with "keep the earliest occurrence"
    used by ``ChunkDeduplicator`` below.
    """
    best: dict[str, tuple[tuple[int, int], int | None, str, Any]] = {}
    versions_seen: dict[str, set[int]] = {}
    order: list[str] = []
    id_to_base: dict[str, str] = {}

    for record in records:
        base_id, version = canonical_arxiv_id(record.raw_id)
        id_to_base[record.raw_id] = base_id

        if base_id not in versions_seen:
            versions_seen[base_id] = set()
            order.append(base_id)
        if version is not None:
            versions_seen[base_id].add(version)

        key = _version_sort_key(version)
        if base_id not in best or key > best[base_id][0]:
            best[base_id] = (key, version, record.raw_id, record.payload)

    resolved = [
        ResolvedDocument(
            base_id=base_id,
            version=best[base_id][1],
            versions_seen=sorted(versions_seen[base_id]),
            raw_id=best[base_id][2],
            payload=best[base_id][3],
        )
        for base_id in order
    ]
    return resolved, id_to_base


# -----------------------------------------------------------------------------
# Chunk-level near-duplicate detection
# -----------------------------------------------------------------------------

# Word characters excluding "_", so "don't" -> ["don", "t"] and punctuation
# never contributes a shingle token -- comparison is purely lexical.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _normalized_words(text: str) -> list[str]:
    """Casefold + tokenize for *comparison only* -- never mutates stored text.

    Case, whitespace run-length, and punctuation are all irrelevant to
    "is this the same passage quoted again"; folding them out here (and
    nowhere else) means the shingle signature is insensitive to them while
    ``chunk.text`` stays byte-exact for retrieval and display.
    """
    return _WORD_RE.findall(text.casefold())


def _shingles(words: list[str], size: int) -> list[str]:
    """Contiguous word n-grams of width ``size``; ``[]`` if too few words."""
    if len(words) < size:
        return []
    return [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]


class ChunkDeduplicator:
    """MinHash + LSH near-duplicate detection over word shingles.

    Usage: call ``add(chunk_uid, text)`` once per chunk, in a stable,
    meaningful order (ingest order is the natural choice), then call
    ``resolve()`` once at the end.

    **Order determines outcome, by design.** "Keep the earliest occurrence"
    means the chunk added first among a near-duplicate group becomes the
    canonical target; add the same chunks in a different order and a
    different one wins. This is documented, not an accident: the same input
    in the same order always gives the same result (MinHash's hashing is
    seeded and deterministic, not based on Python's randomised
    ``hash()``/``PYTHONHASHSEED``, and this class never depends on
    dict/set iteration order for anything order-sensitive -- insertion order
    is tracked explicitly). Feed a different order and you get a different,
    equally deterministic, answer.

    **Short text is never deduplicated, against anything, including other
    short text.** A chunk whose normalised word count is below
    ``shingle_size`` cannot form even one shingle, so there is no signal to
    estimate Jaccard from. Rather than fabricate a degenerate signature (which
    would either crash on an empty MinHash or -- worse -- make many unrelated
    short chunks spuriously collide on the empty signature), such chunks are
    always treated as originals: never matched against the index, never
    inserted into it, so they cannot become any other chunk's match either.

    **Transitivity.** A new chunk is matched against a *root* candidate set
    (every previously-added chunk, duplicate or not, is kept in the LSH index
    so later chunks can still chain through it), and ``resolve()`` walks each
    chunk's direct match to the end of the chain with path compression, so
    A -> B -> C collapses to a single A for all three rather than leaving C
    pointing at an intermediate that itself points elsewhere.
    """

    def __init__(self, config: DedupConfig) -> None:
        self.config = config
        self._order: list[str] = []
        self._minhashes: dict[str, MinHash] = {}
        self._position: dict[str, int] = {}
        self._direct_dup_of: dict[str, str | None] = {}
        self._lsh: MinHashLSH | None = (
            MinHashLSH(threshold=config.jaccard_threshold, num_perm=config.minhash_num_perm)
            if config.enabled
            else None
        )

    def add(self, chunk_uid: str, text: str) -> None:
        """Register one chunk. Must be called exactly once per ``chunk_uid``.

        Raises:
            ValueError: ``chunk_uid`` was already added. A silent overwrite
                would let a caller's bug (re-adding a chunk, e.g. after a
                retried batch) quietly corrupt the position bookkeeping that
                "keep the earliest occurrence" depends on.
        """
        if chunk_uid in self._direct_dup_of:
            raise ValueError(f"chunk_uid added more than once: {chunk_uid!r}")

        self._order.append(chunk_uid)

        if not self.config.enabled:
            # True no-op: never touch the LSH index (there isn't one), never
            # compute a signature. Every chunk is its own original.
            self._direct_dup_of[chunk_uid] = None
            return

        words = _normalized_words(text)
        shingles = _shingles(words, self.config.shingle_size)
        if not shingles:
            self._direct_dup_of[chunk_uid] = None
            return

        mh = MinHash(num_perm=self.config.minhash_num_perm)
        for shingle in shingles:
            mh.update(shingle.encode("utf-8"))

        lsh = self._lsh
        assert lsh is not None  # config.enabled implies an index was built

        earliest_target: str | None = None
        earliest_position = len(self._order)  # sentinel: beyond any real position
        for candidate in lsh.query(mh):
            # LSH banding is a probabilistic *candidate* filter; the actual
            # accept/reject decision against the configured threshold is
            # always this exact recomputation, so behaviour at the threshold
            # boundary depends only on the MinHash Jaccard estimate, never on
            # LSH's internal band/row parameters.
            if mh.jaccard(self._minhashes[candidate]) < self.config.jaccard_threshold:
                continue
            position = self._position[candidate]
            if position < earliest_position:
                earliest_position = position
                earliest_target = candidate

        self._direct_dup_of[chunk_uid] = earliest_target
        self._minhashes[chunk_uid] = mh
        self._position[chunk_uid] = len(self._order) - 1
        lsh.insert(chunk_uid, mh)

    def resolve(self) -> dict[str, str | None]:
        """Resolve every added chunk to its ultimate ``dup_of`` target.

        Returns a dict keyed in insertion order (never dict/set iteration
        order of internal state) mapping each ``chunk_uid`` to the earliest
        chunk in its near-duplicate chain, or ``None`` if it is an original.
        Pure: calling this more than once returns the same result and
        mutates nothing.
        """
        # root_of[x] is the chunk_uid of x's ultimate root -- x itself if x
        # is an original. That self-pointing case is deliberately distinct
        # from the *reported* value: only at the end is "root equals self"
        # translated to the public ``None``, so a node partway through a
        # chain can still be told "your root is chunk X" even when X turns
        # out to be its own root.
        root_of: dict[str, str] = {}
        for chunk_uid in self._order:
            if chunk_uid not in root_of:
                self._resolve_root(chunk_uid, root_of)
        return {
            chunk_uid: None if root_of[chunk_uid] == chunk_uid else root_of[chunk_uid]
            for chunk_uid in self._order
        }

    def _resolve_root(self, chunk_uid: str, root_of: dict[str, str]) -> str:
        """Follow direct ``dup_of`` pointers to the chain's root, iteratively.

        Iterative with path compression rather than recursive: a long version
        chain (v1..v20 of one paper, each a near-duplicate of the last) must
        not risk the recursion limit. ``_direct_dup_of[x]`` always points to a
        strictly earlier-inserted chunk (see ``add``), so this is guaranteed
        to terminate -- there is no cycle to guard against.
        """
        path: list[str] = []
        node = chunk_uid
        while node not in root_of:
            direct = self._direct_dup_of.get(node)
            if direct is None:
                root_of[node] = node  # original: its own root
                break
            path.append(node)
            node = direct
        root = root_of[node]
        for visited in path:
            root_of[visited] = root
        return root
