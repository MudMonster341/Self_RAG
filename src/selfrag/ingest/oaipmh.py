"""Streaming OAI-PMH metadata harvest from arXiv.

arXiv's full metadata is 4-5 GB and this machine has 7.7 GB total (CLAUDE.md).
A harvest therefore never holds more than one OAI-PMH *page* (bounded by the
server, not by us -- arXiv paginates ``ListRecords`` internally) plus the
caller's own row buffer in memory. Buffered rows are flushed to a Parquet
*part file* once at least ``row_group_size`` of them have accumulated, at the
next page boundary -- never mid-page, which matters for the resumability
story below.

**Resumability.** A full harvest is hours of wall-clock at one request every
three seconds, so it must survive being interrupted. The resumption token
handed back with each page is the server's cursor into the harvest; the
state file persists that token together with how many rows have been
*durably flushed* so far, and only ever after a flush actually lands on disk
(via an atomic rename, so a crash mid-write can never leave a corrupt part
file). The consequence of persisting only at flush points -- rather than
after every page -- is that resuming after a crash re-fetches and re-parses
up to one row-group's worth of already-seen pages (bounded, cheap: at
1000 records/page and a 50k row-group, at most ~50 wasted requests, i.e.
~2.5 minutes). The alternative, persisting after every page, would mean
splitting Parquet row groups at page boundaries instead of at
``row_group_size`` -- undoing the very batching this module exists to do.
Restarting the whole harvest from ``from_date`` on every interruption would
be far worse than either.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from selfrag.ids import canonical_arxiv_id
from selfrag.ingest.arxiv_client import ArxivClient
from selfrag.paths import raw_dir

# arXiv moved its OAI-PMH endpoint: the long-documented
# http://export.arxiv.org/oai2 now 301-redirects here. We point at the current
# host directly rather than relying on the redirect, because a redirect costs an
# extra round trip on every page and, at one request per three seconds, a
# multi-hour harvest pays that twice over.
#
# Found by a live smoke test, not by the unit suite: every transport-level test
# here uses a mock, and a mock cannot notice that the real endpoint has moved.
# See ERRORS.md ERR-0003.
OAI_PMH_URL = "https://oaipmh.arxiv.org/oai"

_HARVEST_SCHEMA = pa.schema(
    [
        pa.field("base_id", pa.string()),
        pa.field("version", pa.int32()),
        pa.field("oai_identifier", pa.string()),
        pa.field("status", pa.string()),
        pa.field("title", pa.string()),
        pa.field("abstract", pa.string()),
        pa.field("authors", pa.list_(pa.string())),
        pa.field("categories", pa.list_(pa.string())),
        pa.field("created_at", pa.string()),
        pa.field("updated_at", pa.string()),
        pa.field("license", pa.string()),
        pa.field("doi", pa.string()),
        pa.field("datestamp", pa.string()),
        pa.field("harvested_at", pa.string()),
    ]
)


class OAIPMHError(Exception):
    """A real OAI-PMH protocol error (never raised for ``noRecordsMatch``)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"OAI-PMH error [{code}]: {message}")


class HarvestResult(BaseModel):
    rows_written: int
    tombstones: int
    parts_written: int
    completed: bool


class _HarvestState(BaseModel):
    """Everything needed to resume an interrupted harvest, and nothing else."""

    resumption_token: str | None
    rows_written: int
    tombstones: int
    parts_written: int
    completed: bool
    set_spec: str
    metadata_prefix: str
    from_date: str | None
    until_date: str | None


def _local(tag: str) -> str:
    """Strip the ``{namespace}`` prefix ElementTree keeps on every tag name."""
    return tag.rsplit("}", 1)[-1]


def _iter_local(root: ET.Element, name: str) -> Iterator[ET.Element]:
    for el in root.iter():
        if _local(el.tag) == name:
            yield el


def _child_text(parent: ET.Element, name: str) -> str | None:
    for child in parent:
        if _local(child.tag) == name:
            text = (child.text or "").strip()
            return text or None
    return None


def _parse_authors(authors_el: ET.Element) -> list[str]:
    out: list[str] = []
    for author_el in authors_el:
        if _local(author_el.tag) != "author":
            continue
        keyname = _child_text(author_el, "keyname") or ""
        forenames = _child_text(author_el, "forenames") or ""
        full = f"{forenames} {keyname}".strip()
        if full:
            out.append(full)
    return out


def _parse_header(header_el: ET.Element) -> tuple[str, str, bool]:
    identifier = _child_text(header_el, "identifier") or ""
    datestamp = _child_text(header_el, "datestamp") or ""
    deleted = header_el.get("status") == "deleted"
    return identifier, datestamp, deleted


def _base_id_and_version(raw_id: str) -> tuple[str, int | None]:
    try:
        return canonical_arxiv_id(raw_id)
    except ValueError:
        # An id arXiv's own OAI feed emitted that we cannot parse is a real
        # anomaly, but the harvest of everything else must not die on it --
        # record it as-is rather than raising, so it is visible (a base_id
        # that fails canonical_arxiv_id downstream) instead of silently
        # dropped.
        return raw_id, None


def _parse_record(record_el: ET.Element) -> dict:
    header_el = next((c for c in record_el if _local(c.tag) == "header"), None)
    metadata_el = next((c for c in record_el if _local(c.tag) == "metadata"), None)
    if header_el is None:
        raise OAIPMHError("record", "record is missing a <header> element")

    identifier, datestamp, deleted = _parse_header(header_el)
    oai_raw_id = identifier.rsplit(":", 1)[-1] if identifier else ""
    harvested_at = datetime.now(UTC).isoformat()

    if deleted or metadata_el is None:
        base_id, version = _base_id_and_version(oai_raw_id) if oai_raw_id else ("", None)
        return {
            "base_id": base_id,
            "version": version,
            "oai_identifier": identifier,
            "status": "deleted",
            "title": None,
            "abstract": None,
            "authors": [],
            "categories": [],
            "created_at": None,
            "updated_at": None,
            "license": None,
            "doi": None,
            "datestamp": datestamp,
            "harvested_at": harvested_at,
        }

    arxiv_el = next((c for c in metadata_el if _local(c.tag) == "arXiv"), None)
    if arxiv_el is None:
        raise OAIPMHError("record", f"record {identifier!r} has no <arXiv> metadata block")

    raw_id = _child_text(arxiv_el, "id") or oai_raw_id
    base_id, version = _base_id_and_version(raw_id)

    title = _child_text(arxiv_el, "title")
    title = " ".join(title.split()) if title else None
    abstract = _child_text(arxiv_el, "abstract")
    categories_raw = _child_text(arxiv_el, "categories") or ""
    categories = categories_raw.split() if categories_raw else []
    authors_el = next((c for c in arxiv_el if _local(c.tag) == "authors"), None)
    authors = _parse_authors(authors_el) if authors_el is not None else []

    return {
        "base_id": base_id,
        "version": version,
        "oai_identifier": identifier,
        "status": "active",
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "categories": categories,
        "created_at": _child_text(arxiv_el, "created"),
        "updated_at": _child_text(arxiv_el, "updated"),
        "license": _child_text(arxiv_el, "license"),
        "doi": _child_text(arxiv_el, "doi"),
        "datestamp": datestamp,
        "harvested_at": harvested_at,
    }


def _iter_oai_pages(
    client: ArxivClient, base_url: str, first_params: dict[str, str | None]
) -> Iterator[tuple[list[dict], str | None]]:
    """Yield ``(records_in_page, next_resumption_token)`` once per OAI-PMH page.

    A generator, not a list: each page is fetched only when the caller asks
    for the next one, and its parsed XML is discarded before that happens --
    at no point does this hold more than one page in memory. ``next token
    is None`` signals the harvest is complete (either the server omitted
    ``resumptionToken`` or handed back an empty one, both of which mean "no
    more pages" per the OAI-PMH spec).
    """
    params: dict[str, str | None] = dict(first_params)
    while True:
        response = client.get(base_url, params=params)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        error_el = next(_iter_local(root, "error"), None)
        if error_el is not None:
            code = error_el.get("code", "")
            if code == "noRecordsMatch":
                return
            raise OAIPMHError(code, (error_el.text or "").strip())

        page_records = [_parse_record(el) for el in _iter_local(root, "record")]

        token_el = next(_iter_local(root, "resumptionToken"), None)
        token_text = (token_el.text or "").strip() if token_el is not None else ""
        next_token = token_text or None

        yield page_records, next_token

        if next_token is None:
            return
        params = {"verb": "ListRecords", "resumptionToken": next_token}


def _rows_to_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=_HARVEST_SCHEMA)


def _flush_part(dest_dir: Path, part_index: int, buffer: list[dict]) -> int:
    """Write one self-contained, independently-valid Parquet part file.

    Written to a ``.tmp`` path and atomically renamed into place so that a
    crash mid-write can never leave ``part-NNNNN.parquet`` half-written --
    anything a later reader (or a resumed harvest) finds under that final
    name is guaranteed complete.
    """
    table = _rows_to_table(buffer)
    part_path = dest_dir / f"part-{part_index:05d}.parquet"
    tmp_path = part_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp_path)
    tmp_path.replace(part_path)
    return part_index + 1


def _state_path_for(dest_dir: Path) -> Path:
    return dest_dir / "state.json"


def _load_state(state_path: Path) -> _HarvestState | None:
    if not state_path.exists():
        return None
    return _HarvestState.model_validate_json(state_path.read_text(encoding="utf-8"))


def _save_state(state_path: Path, state: _HarvestState) -> None:
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(state_path)


def _check_state_matches(
    state: _HarvestState,
    set_spec: str,
    metadata_prefix: str,
    from_date: str | None,
    until_date: str | None,
) -> None:
    wanted = (set_spec, metadata_prefix, from_date, until_date)
    got = (state.set_spec, state.metadata_prefix, state.from_date, state.until_date)
    if wanted != got:
        raise ValueError(
            f"existing harvest state at this path was for a different scope "
            f"(set={state.set_spec!r}, metadataPrefix={state.metadata_prefix!r}, "
            f"from={state.from_date!r}, until={state.until_date!r}); wanted "
            f"(set={set_spec!r}, metadataPrefix={metadata_prefix!r}, from={from_date!r}, "
            f"until={until_date!r}). Use a different dest_dir/state_path, or delete the "
            f"stale state file to restart this scope from scratch."
        )


def harvest_oai_pmh(
    client: ArxivClient,
    *,
    set_spec: str = "cs",
    metadata_prefix: str = "arXiv",
    from_date: str | None = None,
    until_date: str | None = None,
    row_group_size: int = 50_000,
    dest_dir: Path | None = None,
    state_path: Path | None = None,
    base_url: str = OAI_PMH_URL,
) -> HarvestResult:
    """Harvest arXiv metadata for one OAI-PMH set into Parquet part files.

    Resumable: if ``state_path`` (default: ``dest_dir/state.json``) already
    records an in-progress harvest for the same scope (set/prefix/date
    bounds), harvesting continues from its resumption token rather than
    restarting. If it records a *completed* harvest, this is a cheap no-op.
    A mismatched scope at an existing state path raises rather than silently
    harvesting the wrong thing under someone else's bookkeeping.

    Returns:
        HarvestResult with total rows written (deleted-record tombstones
        included), tombstone count, part-file count, and whether the
        resumption token was fully exhausted (vs. this call simply reaching
        the end of an already-paginated-elsewhere token chain -- in
        practice this function always runs to completion or raises, so
        ``completed`` is True on any normal return).
    """
    dest_dir = dest_dir or (raw_dir() / "oai_metadata" / set_spec)
    state_path = state_path or _state_path_for(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    state = _load_state(state_path)
    if state is not None:
        _check_state_matches(state, set_spec, metadata_prefix, from_date, until_date)
        if state.completed:
            return HarvestResult(
                rows_written=state.rows_written,
                tombstones=state.tombstones,
                parts_written=state.parts_written,
                completed=True,
            )
        first_params: dict[str, str | None] = {"verb": "ListRecords", "resumptionToken": state.resumption_token}
        rows_written, tombstones, part_index = state.rows_written, state.tombstones, state.parts_written
    else:
        first_params = {"verb": "ListRecords", "metadataPrefix": metadata_prefix, "set": set_spec}
        if from_date:
            first_params["from"] = from_date
        if until_date:
            first_params["until"] = until_date
        rows_written = tombstones = part_index = 0

    buffer: list[dict] = []
    completed = False
    any_page = False

    for page_records, next_token in _iter_oai_pages(client, base_url, first_params):
        any_page = True
        buffer.extend(page_records)
        tombstones += sum(1 for r in page_records if r["status"] == "deleted")
        completed = next_token is None

        if len(buffer) >= row_group_size or completed:
            if buffer:
                part_index = _flush_part(dest_dir, part_index, buffer)
                rows_written += len(buffer)
                buffer = []
            _save_state(
                state_path,
                _HarvestState(
                    resumption_token=next_token,
                    rows_written=rows_written,
                    tombstones=tombstones,
                    parts_written=part_index,
                    completed=completed,
                    set_spec=set_spec,
                    metadata_prefix=metadata_prefix,
                    from_date=from_date,
                    until_date=until_date,
                ),
            )

    if not any_page:
        # noRecordsMatch on the very first page: nothing to harvest, but a
        # real, completed result -- not an error, and not "try again later."
        completed = True
        _save_state(
            state_path,
            _HarvestState(
                resumption_token=None,
                rows_written=rows_written,
                tombstones=tombstones,
                parts_written=part_index,
                completed=True,
                set_spec=set_spec,
                metadata_prefix=metadata_prefix,
                from_date=from_date,
                until_date=until_date,
            ),
        )

    return HarvestResult(
        rows_written=rows_written, tombstones=tombstones, parts_written=part_index, completed=completed
    )
