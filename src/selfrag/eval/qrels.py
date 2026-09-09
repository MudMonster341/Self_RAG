"""Append-only, versioned storage for relevance judgements (qrels).

Why span-based and why versioned, together: a qrel is a judgement over a
document character span (``selfrag.schema.Qrel``), never a chunk id, so that
re-chunking a document -- the first ablation axis anyone runs -- never
invalidates a single judgement. But judgements themselves are *not*
immutable truth: a human reviewer corrects a grade, a pooled judgement turns
out wrong, a new relevant passage is found. Those corrections must be
recorded, not merged silently into history, so that:

  1. every score ever reported can be traced to the exact qrels version that
     produced it (``RunManifest.qrels_version`` / ``rescore``'s
     ``qrels_version`` argument), and
  2. a correction to the qrels never *edits* a past experiment's inputs out
     from under it -- old runs are re-scored against the new version
     explicitly, not silently re-interpreted.

Each version file is a full, self-contained snapshot of all judgements as of
that version (not a delta), which is what makes ``diff`` between two
versions meaningful: a judgement can be added, removed, or regraded between
v_a and v_b, and ``diff`` reports all three. The one invariant enforced in
code is that a version file, once written, is never opened for writing
again.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from selfrag.schema import Qrel

_VERSION_RE = re.compile(r"^v(\d+)\.parquet$")

# Explicit schema so round-tripping never depends on pyarrow's type inference
# guessing wrong on an all-null column (e.g. every judged_at being absent).
_QREL_SCHEMA = pa.schema(
    [
        ("qrels_version", pa.int64()),
        ("query_id", pa.string()),
        ("doc_id", pa.string()),
        ("char_start", pa.int64()),
        ("char_end", pa.int64()),
        ("grade", pa.int64()),
        ("judge", pa.string()),
        # Stored as ISO-8601 text, not pyarrow's timestamp type, so that
        # timezone-awareness survives the round trip exactly as pydantic
        # parsed it (pyarrow's timestamp columns are tz-naive by default and
        # would silently strip that information).
        ("judged_at", pa.string()),
    ]
)


class QrelValidationError(ValueError):
    """A batch of qrels violates a write-time invariant (duplicate key, etc.)."""


class QrelVersionExistsError(FileExistsError):
    """Refused write: this qrels version file already exists on disk.

    Qrels are append-only. If this fires, the caller asked to write a
    version number that has already been committed -- the fix is to append
    a *new* version, never to overwrite this one.
    """


QrelKey = tuple[str, str, int, int]  # (query_id, doc_id, char_start, char_end)


def _key(q: Qrel) -> QrelKey:
    return (q.query_id, q.doc_id, q.char_start, q.char_end)


@dataclass(frozen=True)
class QrelVersionMeta:
    """Sidecar metadata written next to every version's Parquet file."""

    qrels_version: int
    created_at: str
    note: str
    n_queries: int
    n_judgements: int
    judges: list[str] = field(default_factory=list)
    parent_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "qrels_version": self.qrels_version,
            "created_at": self.created_at,
            "note": self.note,
            "n_queries": self.n_queries,
            "n_judgements": self.n_judgements,
            "judges": self.judges,
            "parent_version": self.parent_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QrelVersionMeta:
        return cls(
            qrels_version=d["qrels_version"],
            created_at=d["created_at"],
            note=d["note"],
            n_queries=d["n_queries"],
            n_judgements=d["n_judgements"],
            judges=list(d.get("judges", [])),
            parent_version=d.get("parent_version"),
        )


@dataclass(frozen=True)
class QrelDiff:
    """Summary of how qrels changed between two versions.

    ``added``/``removed`` are judgement keys (query_id, doc_id, char_start,
    char_end); ``regraded`` are the same keys paired with (grade_in_a,
    grade_in_b) for keys present in both versions with a different grade.
    """

    v_a: int
    v_b: int
    added: list[QrelKey]
    removed: list[QrelKey]
    regraded: list[tuple[QrelKey, int, int]]
    n_unchanged: int


def _qrels_to_table(qrels: Sequence[Qrel]) -> pa.Table:
    records = [
        {
            "qrels_version": q.qrels_version,
            "query_id": q.query_id,
            "doc_id": q.doc_id,
            "char_start": q.char_start,
            "char_end": q.char_end,
            "grade": q.grade,
            "judge": q.judge,
            "judged_at": q.judged_at.isoformat() if q.judged_at is not None else None,
        }
        for q in qrels
    ]
    return pa.Table.from_pylist(records, schema=_QREL_SCHEMA)


def _table_to_qrels(table: pa.Table) -> list[Qrel]:
    return [Qrel(**row) for row in table.to_pylist()]


class QrelStore:
    """Reads and writes versioned qrels snapshots under ``root``.

    ``root`` holds one Parquet file per version (``v{N}.parquet``) plus a
    JSON sidecar (``v{N}.meta.json``). Versions are assigned automatically by
    ``append_version`` as ``latest_version() + 1`` (or 1 for the first
    write) -- callers never choose a version number, which is what makes the
    "never overwrite an existing version" guarantee enforceable purely by
    checking whether the target path already exists.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[int, list[Qrel]] = {}

    def _parquet_path(self, version: int) -> Path:
        return self.root / f"v{version}.parquet"

    def _meta_path(self, version: int) -> Path:
        return self.root / f"v{version}.meta.json"

    def _all_versions(self) -> list[int]:
        versions = []
        for p in self.root.glob("v*.parquet"):
            m = _VERSION_RE.match(p.name)
            if m:
                versions.append(int(m.group(1)))
        return sorted(versions)

    def latest_version(self) -> int | None:
        """The most recently written version number, or ``None`` if the store is empty."""
        versions = self._all_versions()
        return versions[-1] if versions else None

    def load(self, version: int) -> list[Qrel]:
        """Load the full snapshot of qrels for one version."""
        if version in self._cache:
            return list(self._cache[version])
        path = self._parquet_path(version)
        if not path.exists():
            raise FileNotFoundError(f"qrels version {version} not found at {path}")
        qrels = _table_to_qrels(pq.read_table(path))
        self._cache[version] = qrels
        return list(qrels)

    def load_meta(self, version: int) -> QrelVersionMeta:
        path = self._meta_path(version)
        if not path.exists():
            raise FileNotFoundError(f"qrels version {version} has no sidecar metadata at {path}")
        return QrelVersionMeta.from_dict(orjson.loads(path.read_bytes()))

    def append_version(self, qrels: Sequence[Qrel], note: str, judges: Sequence[str] | None = None) -> int:
        """Write a new qrels version as a full snapshot. Never overwrites.

        ``qrels`` must be the *complete* set of judgements for the new
        version (typically the previous version plus additions and
        corrections), not just an incremental delta -- see the module
        docstring for why full snapshots are what makes ``diff`` meaningful.

        The incoming ``Qrel.qrels_version`` values are ignored and
        overwritten with the version number this call assigns; callers
        should not need to know the next version number in advance.

        Raises:
            ValueError: if ``qrels`` is empty, or contains a duplicate
                (query_id, doc_id, char_start, char_end) key.
            QrelVersionExistsError: if the target version file already
                exists on disk (should only happen if something wrote to
                the store outside this API).
        """
        if not qrels:
            raise ValueError("cannot append an empty qrels version")

        seen: set[QrelKey] = set()
        stamped: list[Qrel] = []
        parent = self.latest_version()
        new_version = 1 if parent is None else parent + 1

        for q in qrels:
            key = _key(q)
            if key in seen:
                raise QrelValidationError(f"duplicate judgement in this version for key {key}")
            seen.add(key)
            # char span emptiness/grade range are already enforced by the
            # Qrel model's own validators; re-checking here would be
            # redundant with pydantic, so we trust the model.
            stamped.append(q.model_copy(update={"qrels_version": new_version}))

        # Check BOTH target paths before writing either one. Writing the
        # Parquet file first and only then discovering the metadata sidecar
        # collides would leave a dangling data file with no metadata --
        # itself a silent corruption of the append-only guarantee this
        # method exists to enforce.
        path = self._parquet_path(new_version)
        meta_path = self._meta_path(new_version)
        if path.exists():
            raise QrelVersionExistsError(
                f"qrels version {new_version} already exists at {path} -- "
                "qrels are append-only, refusing to overwrite"
            )
        if meta_path.exists():
            raise QrelVersionExistsError(
                f"qrels version {new_version} metadata already exists at {meta_path} -- "
                "qrels are append-only, refusing to overwrite"
            )

        distinct_judges = judges if judges is not None else sorted({q.judge for q in stamped})
        meta = QrelVersionMeta(
            qrels_version=new_version,
            created_at=datetime.now(UTC).isoformat(),
            note=note,
            n_queries=len({q.query_id for q in stamped}),
            n_judgements=len(stamped),
            judges=sorted(distinct_judges),
            parent_version=parent,
        )

        pq.write_table(_qrels_to_table(stamped), path)
        meta_path.write_bytes(orjson.dumps(meta.to_dict(), option=orjson.OPT_INDENT_2))

        self._cache[new_version] = stamped
        return new_version

    def diff(self, v_a: int, v_b: int) -> QrelDiff:
        """Summarise how qrels changed between two versions."""
        a = {_key(q): q for q in self.load(v_a)}
        b = {_key(q): q for q in self.load(v_b)}
        added = sorted(set(b) - set(a))
        removed = sorted(set(a) - set(b))
        common = set(a) & set(b)
        regraded = sorted(
            ((k, a[k].grade, b[k].grade) for k in common if a[k].grade != b[k].grade),
            key=lambda item: item[0],
        )
        n_unchanged = len(common) - len(regraded)
        return QrelDiff(
            v_a=v_a, v_b=v_b, added=added, removed=removed, regraded=regraded, n_unchanged=n_unchanged
        )

    def for_query(self, version: int, query_id: str) -> list[Qrel]:
        """All judgements for one query, at one version."""
        return [q for q in self.load(version) if q.query_id == query_id]

    def queries_with_judgements(self, version: int) -> set[str]:
        """Every query_id that has at least one judgement at this version."""
        return {q.query_id for q in self.load(version)}

    def group_by_query(self, version: int) -> dict[str, list[Qrel]]:
        """All judgements at one version, grouped by query_id.

        A convenience for scoring code (``runfile.rescore``) that needs this
        grouping for every query in a run, where calling ``for_query`` once
        per query would re-scan the whole version on every call.
        """
        out: dict[str, list[Qrel]] = {}
        for q in self.load(version):
            out.setdefault(q.query_id, []).append(q)
        return out
