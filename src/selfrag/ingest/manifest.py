"""The corpus manifest: the exact list of documents in one corpus snapshot.

This is a different "machine truth" from the DuckDB experiment ledger
(``selfrag.ledger``: ``runs`` / ``run_metrics`` / ``qrels``). The ledger
answers "which config won and by how much"; the manifest answers "which
documents are actually in this corpus, and what state is each one in" --
a question that has to be answerable before any run exists, and one the
ledger's schema was never designed to hold. Keeping them apart means
acquisition can be re-run, retried, and dead-lettered freely without ever
touching ``ledger.py``.

Backed by a single Parquet file, loaded fully into memory and rewritten
wholesale on every ``save()``. That is a deliberate scale bet, not laziness:
CLAUDE.md sizes the dev corpus at 25-40k *chunks*, i.e. at most a few
thousand *documents* -- a full read-modify-write at that size is
milliseconds, and it is what makes the idempotency guarantee trivial to
reason about and to test: entries live in one dict keyed by ``doc_id``, so
there is no code path that could ever produce two rows for the same
document.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from selfrag.ids import config_hash


class ManifestStatus(StrEnum):
    PENDING = "pending"
    ACQUIRED = "acquired"
    PARSED = "parsed"
    FAILED = "failed"
    TOMBSTONED = "tombstoned"


class ManifestEntry(BaseModel):
    """One document's row in the manifest.

    ``doc_id`` is the version-stripped arXiv base id -- the same key
    ``selfrag.schema.Document.doc_id`` uses -- so a manifest row and the
    ``Document`` it eventually produces always agree on identity. ``version``
    records *which* version was actually fetched, independent of that.
    """

    doc_id: str
    version: int | None = None
    source_url: str = ""
    content_hash: str | None = None
    acquired_at: datetime | None = None
    parser_used: str | None = None
    status: ManifestStatus = ManifestStatus.PENDING
    failure_reason: str | None = None


def _empty_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("doc_id", pa.string()),
            pa.field("version", pa.int64()),
            pa.field("source_url", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("acquired_at", pa.string()),
            pa.field("parser_used", pa.string()),
            pa.field("status", pa.string()),
            pa.field("failure_reason", pa.string()),
        ]
    )


def _write_manifest(path: Path, entries: list[ManifestEntry]) -> None:
    rows = [e.model_dump(mode="json") for e in entries]
    table = pa.Table.from_pylist(rows, schema=_empty_schema() if not rows else None)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


def _read_manifest(path: Path) -> dict[str, ManifestEntry]:
    table = pq.read_table(path)
    entries = (ManifestEntry.model_validate(row) for row in table.to_pylist())
    return {e.doc_id: e for e in entries}


class Manifest:
    """The reproducibility record of which documents are in a corpus snapshot."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, ManifestEntry] = _read_manifest(path) if path.exists() else {}

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[ManifestEntry]:
        return iter(self._entries.values())

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._entries

    def get(self, doc_id: str) -> ManifestEntry | None:
        return self._entries.get(doc_id)

    def needs_acquisition(self, doc_id: str) -> bool:
        """False iff ``doc_id`` is already durably acquired, parsed, or tombstoned.

        This is the check acquisition is expected to make before downloading
        anything, which is what makes re-running acquisition over an
        existing manifest idempotent: a document already past ``pending``
        (successfully, or deliberately removed) is never re-fetched just
        because the acquisition step ran again. ``failed`` still needs work
        -- that is the whole point of recording failures rather than
        dropping them.
        """
        entry = self._entries.get(doc_id)
        if entry is None:
            return True
        return entry.status not in (
            ManifestStatus.ACQUIRED,
            ManifestStatus.PARSED,
            ManifestStatus.TOMBSTONED,
        )

    def seed_pending(self, doc_id: str, source_url: str, *, version: int | None = None) -> bool:
        """Idempotently register ``doc_id`` as a target of this corpus.

        Returns True only if this created a new row. Never overwrites an
        entry that has already made progress past ``pending`` -- reseeding
        a manifest from a freshly re-fetched candidate-document list must
        not regress an already-acquired document back to square one, and
        must not create a second row for it either.
        """
        if doc_id in self._entries:
            return False
        self._entries[doc_id] = ManifestEntry(doc_id=doc_id, version=version, source_url=source_url)
        return True

    def upsert(self, entry: ManifestEntry) -> bool:
        """Insert or replace the row for ``entry.doc_id``. Returns True iff new."""
        is_new = entry.doc_id not in self._entries
        self._entries[entry.doc_id] = entry
        return is_new

    def mark_acquired(
        self,
        doc_id: str,
        *,
        source_url: str,
        content_hash: str,
        version: int | None = None,
        acquired_at: datetime | None = None,
    ) -> None:
        """Record a successful download. Safe to call again for the same doc_id.

        Calling this twice for the same ``doc_id`` (the idempotent-re-run
        case) updates the one existing row in place rather than adding a
        second -- storage is a dict keyed by ``doc_id``, so there is no
        other outcome.
        """
        existing = self._entries.get(doc_id)
        self._entries[doc_id] = ManifestEntry(
            doc_id=doc_id,
            version=version if version is not None else (existing.version if existing else None),
            source_url=source_url,
            content_hash=content_hash,
            acquired_at=acquired_at or datetime.now(UTC),
            parser_used=existing.parser_used if existing else None,
            status=ManifestStatus.ACQUIRED,
            failure_reason=None,
        )

    def mark_parsed(self, doc_id: str, *, parser_used: str) -> None:
        existing = self._entries.get(doc_id)
        if existing is None:
            raise KeyError(f"cannot mark {doc_id!r} parsed -- it was never acquired")
        self._entries[doc_id] = existing.model_copy(
            update={"status": ManifestStatus.PARSED, "parser_used": parser_used, "failure_reason": None}
        )

    def mark_failed(self, doc_id: str, reason: str, *, version: int | None = None, source_url: str = "") -> None:
        """Dead-letter ``doc_id``: recorded with its failure reason, never dropped."""
        existing = self._entries.get(doc_id)
        if existing is None:
            self._entries[doc_id] = ManifestEntry(
                doc_id=doc_id,
                version=version,
                source_url=source_url,
                status=ManifestStatus.FAILED,
                failure_reason=reason,
            )
        else:
            self._entries[doc_id] = existing.model_copy(
                update={"status": ManifestStatus.FAILED, "failure_reason": reason}
            )

    def mark_tombstoned(self, doc_id: str) -> None:
        existing = self._entries.get(doc_id)
        if existing is None:
            raise KeyError(f"cannot tombstone {doc_id!r} -- it is not in the manifest")
        self._entries[doc_id] = existing.model_copy(
            update={"status": ManifestStatus.TOMBSTONED, "failure_reason": None}
        )

    def snapshot_id(self) -> str:
        """Content-addressed id for the *acquired* subset of this manifest.

        Only ``acquired``/``parsed`` rows count: a snapshot is what is
        actually in the corpus, not what is merely queued or dead-lettered.
        Hashes ``(doc_id, version, content_hash)`` triples, sorted by
        ``doc_id`` so insertion order never affects the id. This is what
        lets ``RunManifest.corpus_snapshot`` mean something concrete instead
        of being a label nobody can verify.
        """
        live = sorted(
            (
                {"doc_id": e.doc_id, "version": e.version, "content_hash": e.content_hash}
                for e in self._entries.values()
                if e.status in (ManifestStatus.ACQUIRED, ManifestStatus.PARSED)
            ),
            key=lambda d: (d["doc_id"], d["version"] or 0),
        )
        return config_hash({"docs": live})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _write_manifest(self.path, list(self._entries.values()))
