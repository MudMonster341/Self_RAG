"""Tests for selfrag.ingest.oaipmh.

Every test drives ``harvest_oai_pmh`` against ``httpx.MockTransport`` with
hand-built OAI-PMH XML pages -- nothing here touches the network, and the
shared rate limiter is faked (autouse fixture below) so pagination tests
that make several requests do not spend real seconds waiting between them.
"""

from __future__ import annotations

import inspect

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from selfrag.ingest import arxiv_client as _ac
from selfrag.ingest.arxiv_client import ArxivClient
from selfrag.ingest.oaipmh import (
    OAI_PMH_URL,
    HarvestResult,
    OAIPMHError,
    _HarvestState,
    _iter_oai_pages,
    _save_state,
    harvest_oai_pmh,
)

OAI_NS = "http://www.openarchives.org/OAI/2.0/"


@pytest.fixture(autouse=True)
def fast_shared_limiter(monkeypatch):
    state = {"now": 0.0}

    def _clock() -> float:
        return state["now"]

    def _sleep(seconds: float) -> None:
        state["now"] += seconds

    monkeypatch.setattr(_ac._SHARED_LIMITER, "_clock", _clock)
    monkeypatch.setattr(_ac._SHARED_LIMITER, "_sleep", _sleep)
    monkeypatch.setattr(_ac._SHARED_LIMITER, "_last_call", None)


def _record_xml(arxiv_id: str, title: str, *, created: str = "2024-01-15", categories: str = "cs.IR") -> str:
    return f"""
    <record>
      <header>
        <identifier>oai:arXiv.org:{arxiv_id}</identifier>
        <datestamp>{created}</datestamp>
        <setSpec>cs</setSpec>
      </header>
      <metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>{arxiv_id}</id>
          <created>{created}</created>
          <title>{title}</title>
          <authors>
            <author><keyname>Doe</keyname><forenames>Jane</forenames></author>
          </authors>
          <categories>{categories}</categories>
          <abstract>An abstract about {title}.</abstract>
        </arXiv>
      </metadata>
    </record>
    """


def _deleted_record_xml(arxiv_id: str, *, datestamp: str = "2024-02-01") -> str:
    return f"""
    <record>
      <header status="deleted">
        <identifier>oai:arXiv.org:{arxiv_id}</identifier>
        <datestamp>{datestamp}</datestamp>
      </header>
    </record>
    """


def _list_records_page(records_xml: str, *, resumption_token: str | None) -> str:
    """``resumption_token=None`` omits the element entirely (end of harvest);
    ``""`` emits an empty element (also end of harvest, but the "present but
    empty" spelling arXiv itself uses on the final page)."""
    token_el = ""
    if resumption_token is not None:
        token_el = f'<resumptionToken cursor="0" completeListSize="0">{resumption_token}</resumptionToken>'
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <OAI-PMH xmlns="{OAI_NS}">
      <responseDate>2024-01-01T00:00:00Z</responseDate>
      <request verb="ListRecords">{OAI_PMH_URL}</request>
      <ListRecords>
        {records_xml}
        {token_el}
      </ListRecords>
    </OAI-PMH>
    """


def _no_records_match_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <OAI-PMH xmlns="{OAI_NS}">
      <responseDate>2024-01-01T00:00:00Z</responseDate>
      <request verb="ListRecords">{OAI_PMH_URL}</request>
      <error code="noRecordsMatch">no matching records</error>
    </OAI-PMH>
    """


def _error_xml(code: str, message: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <OAI-PMH xmlns="{OAI_NS}">
      <responseDate>2024-01-01T00:00:00Z</responseDate>
      <request verb="ListRecords">{OAI_PMH_URL}</request>
      <error code="{code}">{message}</error>
    </OAI-PMH>
    """


def _client_for(handler) -> ArxivClient:
    return ArxivClient(transport=httpx.MockTransport(handler), retry_sleep=lambda _s: None)


class TestPagination:
    def test_harvest_follows_resumption_tokens_across_multiple_pages(self, tmp_path):
        page1 = _list_records_page(_record_xml("2401.00001", "First"), resumption_token="TOK1")
        page2 = _list_records_page(_record_xml("2401.00002", "Second"), resumption_token="")
        requests_seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests_seen.append(dict(request.url.params))
            return httpx.Response(200, text=page1 if len(requests_seen) == 1 else page2)

        client = _client_for(handler)
        dest_dir = tmp_path / "cs"
        result = harvest_oai_pmh(client, set_spec="cs", dest_dir=dest_dir, state_path=dest_dir / "state.json")

        assert result.rows_written == 2
        assert result.completed is True
        assert len(requests_seen) == 2
        assert requests_seen[0]["set"] == "cs"
        assert requests_seen[1] == {"verb": "ListRecords", "resumptionToken": "TOK1"}

        parts = sorted(dest_dir.glob("part-*.parquet"))
        assert len(parts) == 1
        table = pq.read_table(parts[0])
        assert table.num_rows == 2
        assert set(table.column("base_id").to_pylist()) == {"2401.00001", "2401.00002"}

    def test_from_and_until_are_sent_on_the_first_request_only(self, tmp_path):
        requests_seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests_seen.append(dict(request.url.params))
            return httpx.Response(200, text=_no_records_match_xml())

        client = _client_for(handler)
        dest_dir = tmp_path / "cs"
        harvest_oai_pmh(
            client,
            set_spec="cs",
            dest_dir=dest_dir,
            state_path=dest_dir / "state.json",
            from_date="2024-01-01",
            until_date="2024-01-31",
        )
        assert requests_seen[0]["from"] == "2024-01-01"
        assert requests_seen[0]["until"] == "2024-01-31"


class TestDeletedRecords:
    def test_deleted_record_becomes_a_tombstone_row_not_a_dropped_one(self, tmp_path):
        page = _list_records_page(
            _record_xml("2401.00001", "Alive") + _deleted_record_xml("2401.00002"),
            resumption_token=None,
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=page)

        client = _client_for(handler)
        dest_dir = tmp_path / "cs"
        result = harvest_oai_pmh(client, set_spec="cs", dest_dir=dest_dir, state_path=dest_dir / "state.json")

        assert result.rows_written == 2
        assert result.tombstones == 1

        table = pq.read_table(sorted(dest_dir.glob("part-*.parquet"))[0])
        rows = table.to_pylist()
        deleted_rows = [r for r in rows if r["status"] == "deleted"]
        assert len(deleted_rows) == 1
        assert deleted_rows[0]["base_id"] == "2401.00002"
        assert deleted_rows[0]["title"] is None


class TestNoRecordsMatch:
    def test_is_a_clean_zero_row_completion_not_an_error(self, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=_no_records_match_xml())

        client = _client_for(handler)
        dest_dir = tmp_path / "cs"
        result = harvest_oai_pmh(client, set_spec="cs", dest_dir=dest_dir, state_path=dest_dir / "state.json")
        assert result == HarvestResult(rows_written=0, tombstones=0, parts_written=0, completed=True)
        assert list(dest_dir.glob("part-*.parquet")) == []


class TestProtocolErrors:
    def test_a_real_error_code_raises_oaipmh_error(self, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=_error_xml("badResumptionToken", "token expired"))

        client = _client_for(handler)
        dest_dir = tmp_path / "cs"
        with pytest.raises(OAIPMHError) as exc_info:
            harvest_oai_pmh(client, set_spec="cs", dest_dir=dest_dir, state_path=dest_dir / "state.json")
        assert exc_info.value.code == "badResumptionToken"


class TestStreamingAndRowGroups:
    def test_pages_flush_into_multiple_part_files_at_the_row_group_boundary(self, tmp_path):
        # 3 pages of 2 records each = 6 total. row_group_size=3 means the
        # buffer only crosses the threshold *between* pages: it flushes
        # once 4 rows are buffered (after page 2), then flushes the
        # remaining 2 at completion (after page 3) -- never splitting a
        # page across two part files.
        pages = [
            _list_records_page(
                _record_xml("2401.00001", "A") + _record_xml("2401.00002", "B"), resumption_token="TOK1"
            ),
            _list_records_page(
                _record_xml("2401.00003", "C") + _record_xml("2401.00004", "D"), resumption_token="TOK2"
            ),
            _list_records_page(
                _record_xml("2401.00005", "E") + _record_xml("2401.00006", "F"), resumption_token=""
            ),
        ]
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            idx = call_count["n"]
            call_count["n"] += 1
            return httpx.Response(200, text=pages[idx])

        client = _client_for(handler)
        dest_dir = tmp_path / "cs"
        result = harvest_oai_pmh(
            client, set_spec="cs", dest_dir=dest_dir, state_path=dest_dir / "state.json", row_group_size=3
        )

        assert result.rows_written == 6
        assert call_count["n"] == 3

        parts = sorted(dest_dir.glob("part-*.parquet"))
        assert len(parts) == 2
        row_counts = [pq.read_table(p).num_rows for p in parts]
        assert row_counts == [4, 2]

    def test_iter_oai_pages_is_a_generator_that_fetches_lazily(self):
        """Proves pages are fetched on demand, never all at once: exactly as
        many HTTP calls happen as pages exist, and none happen before the
        caller asks for that page."""
        pages = [
            _list_records_page(_record_xml("2401.00001", "A"), resumption_token="TOK1"),
            _list_records_page(_record_xml("2401.00002", "B"), resumption_token=None),
        ]
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            idx = call_count["n"]
            call_count["n"] += 1
            return httpx.Response(200, text=pages[idx])

        client = _client_for(handler)
        gen = _iter_oai_pages(
            client, OAI_PMH_URL, {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": "cs"}
        )
        assert inspect.isgenerator(gen)
        assert call_count["n"] == 0  # nothing fetched before the first next()

        first_records, token = next(gen)
        assert call_count["n"] == 1
        assert token == "TOK1"
        assert len(first_records) == 1

        second_records, token2 = next(gen)
        assert call_count["n"] == 2
        assert token2 is None

        with pytest.raises(StopIteration):
            next(gen)
        assert call_count["n"] == 2  # never fetched a nonexistent third page


class TestResumability:
    def test_resumes_from_the_saved_token_instead_of_restarting(self, tmp_path):
        dest_dir = tmp_path / "cs"
        state_path = dest_dir / "state.json"
        dest_dir.mkdir(parents=True)

        _save_state(
            state_path,
            _HarvestState(
                resumption_token="RESUME_HERE",
                rows_written=5,
                tombstones=1,
                parts_written=1,
                completed=False,
                set_spec="cs",
                metadata_prefix="arXiv",
                from_date=None,
                until_date=None,
            ),
        )
        # a part file from "before the crash" that a resumed harvest must
        # neither touch nor duplicate
        pq.write_table(pa.table({"base_id": ["already-there"]}), dest_dir / "part-00000.parquet")

        requests_seen: list[dict] = []
        page = _list_records_page(_record_xml("2401.00099", "Resumed"), resumption_token=None)

        def handler(request: httpx.Request) -> httpx.Response:
            requests_seen.append(dict(request.url.params))
            return httpx.Response(200, text=page)

        client = _client_for(handler)
        result = harvest_oai_pmh(client, set_spec="cs", dest_dir=dest_dir, state_path=state_path)

        assert requests_seen == [{"verb": "ListRecords", "resumptionToken": "RESUME_HERE"}]
        assert result.rows_written == 6  # 5 already-durable + 1 newly harvested
        assert result.tombstones == 1
        assert result.parts_written == 2
        assert result.completed is True

    def test_completed_state_is_a_cheap_no_op_and_makes_no_request(self, tmp_path):
        dest_dir = tmp_path / "cs"
        state_path = dest_dir / "state.json"
        dest_dir.mkdir(parents=True)
        _save_state(
            state_path,
            _HarvestState(
                resumption_token=None,
                rows_written=42,
                tombstones=0,
                parts_written=1,
                completed=True,
                set_spec="cs",
                metadata_prefix="arXiv",
                from_date=None,
                until_date=None,
            ),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("must not request anything for an already-completed harvest")

        client = _client_for(handler)
        result = harvest_oai_pmh(client, set_spec="cs", dest_dir=dest_dir, state_path=state_path)
        assert result.rows_written == 42
        assert result.completed is True

    def test_mismatched_scope_at_an_existing_state_path_raises_before_any_request(self, tmp_path):
        dest_dir = tmp_path / "cs"
        state_path = dest_dir / "state.json"
        dest_dir.mkdir(parents=True)
        _save_state(
            state_path,
            _HarvestState(
                resumption_token="T",
                rows_written=0,
                tombstones=0,
                parts_written=0,
                completed=False,
                set_spec="math",  # different set than this call asks for
                metadata_prefix="arXiv",
                from_date=None,
                until_date=None,
            ),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("must validate scope before making any request")

        client = _client_for(handler)
        with pytest.raises(ValueError, match="different scope"):
            harvest_oai_pmh(client, set_spec="cs", dest_dir=dest_dir, state_path=state_path)
