"""Tests for selfrag.ingest.pipeline -- the Phase 1 orchestrator.

Every test drives ``run_ingest`` through ``FixtureSource``, a small
in-memory ``DocumentSource`` implementation defined in this file. It never
imports ``httpx``/``ArxivClient``/anything network-capable, so a network
call is structurally unreachable from anything in this module -- see
``pipeline``'s own module docstring on why that is the design, not an
accident of these tests.

The Phase 1 exit criterion itself (re-ingesting the same corpus adds zero
new chunk ids) is proven end to end in
``tests/integration/test_ingest_idempotency.py``; this file covers the
individual behaviours that make that criterion true: dead-lettering,
resumability from every manifest status, the dedup pass, the span
assertion, parser selection, and ``--dry-run``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
import pytest

from selfrag.ingest import pipeline
from selfrag.ingest.dedup import DedupConfig
from selfrag.ingest.manifest import Manifest, ManifestStatus
from selfrag.ledger import Ledger
from selfrag.schema import Chunk

# ---------------------------------------------------------------------------
# Fixtures shared by every test in this file.
# ---------------------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points SELFRAG_DATA_DIR at a fresh tmp_path -- canonical.py has no
    injectable override, so every pipeline test needs this regardless of
    whether it looks like it touches canonical text directly."""
    monkeypatch.setenv("SELFRAG_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def ledger(tmp_path: Path):
    with Ledger(tmp_path / "ledger.duckdb") as led:
        yield led


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "manifest.parquet"


@dataclass
class FixtureSource:
    """Test double for ``pipeline.DocumentSource``. Serves pre-built local
    files; never touches the network. A doc_id not registered in either
    dict raises, so a test's fixture set is exactly what it declares."""

    latex_docs: dict[str, tuple[Path, str]] = field(default_factory=dict)
    pdf_docs: dict[str, Path] = field(default_factory=dict)
    acquire_calls: list[str] = field(default_factory=list)

    def acquire(self, doc_id: str, dest_root: Path) -> pipeline.AcquiredDocument:
        self.acquire_calls.append(doc_id)
        if doc_id in self.latex_docs:
            latex_dir, main_file = self.latex_docs[doc_id]
            return pipeline.AcquiredDocument(
                source_url=f"https://arxiv.org/abs/{doc_id}",
                content_hash=f"hash-{doc_id}",
                latex_dir=latex_dir,
                main_tex_file=main_file,
            )
        if doc_id in self.pdf_docs:
            return pipeline.AcquiredDocument(
                source_url=f"https://arxiv.org/abs/{doc_id}",
                content_hash=f"hash-{doc_id}",
                pdf_path=self.pdf_docs[doc_id],
            )
        raise RuntimeError(f"FixtureSource has no fixture registered for {doc_id!r}")

    def acquire_pdf_fallback(self, doc_id: str, dest_root: Path) -> Path:
        if doc_id in self.pdf_docs:
            return self.pdf_docs[doc_id]
        raise RuntimeError(f"FixtureSource has no pdf fallback for {doc_id!r}")


def _write_tex(tmp_path: Path, name: str, content: str) -> tuple[Path, str]:
    d = tmp_path / name
    d.mkdir()
    (d / "main.tex").write_text(content, encoding="utf-8")
    return d, "main.tex"


def _make_pdf(tmp_path: Path, name: str, blocks: list[tuple[str, int]]) -> Path:
    """A minimal single-page, real PDF -- same technique as test_pdf_fallback.py."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72.0
    for text, fontsize in blocks:
        page.insert_text((72, y), text, fontsize=fontsize)
        y += max(30, fontsize * 2)
    path = tmp_path / name
    doc.save(str(path))
    doc.close()
    return path


_SHARED_PARAGRAPH = (
    "Retrieval augmented generation combines a retriever with a generator to answer "
    "questions using external evidence documents that are retrieved at query time "
    "from a large corpus."
)

# docA and docB share an identical opening (title + Introduction section) and
# diverge afterward -- crafted so a small fixed chunk_size (100 chars, no
# overlap) produces byte-identical leading chunks between the two documents,
# exercising ChunkDeduplicator the way a quoted related-work paragraph would
# in a real corpus (see dedup.py's own module docstring).
_DOC_A_TEX = rf"""\documentclass{{article}}
\title{{Paper A}}
\begin{{document}}
\section{{Introduction}}
{_SHARED_PARAGRAPH}

\section{{Method}}
Paper A proposes a fixed chunking strategy with character based windows and measures recall at various cutoff values on a held out development set of queries.
\end{{document}}
"""

_DOC_B_TEX = rf"""\documentclass{{article}}
\title{{Paper B}}
\begin{{document}}
\section{{Introduction}}
{_SHARED_PARAGRAPH}

\section{{Results}}
Paper B reports results on three benchmark datasets and compares against several strong baselines including dense and sparse retrieval methods combined via late fusion.
\end{{document}}
"""

_DOC_SIMPLE_TEX = r"""\documentclass{article}
\title{A Simple Paper}
\begin{document}
\section{Introduction}
This is a short, self-contained document used where the test only needs one clean, parseable document and does not care about its exact content.
\end{document}
"""


def _basic_config(doc_ids: list[str], **overrides) -> pipeline.IngestConfig:
    defaults: dict = dict(corpus_name="test", doc_ids=doc_ids)
    defaults.update(overrides)
    return pipeline.IngestConfig(**defaults)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_ingests_independent_documents(self, data_dir, ledger, manifest_path, tmp_path):
        source = FixtureSource(
            latex_docs={
                "docX": _write_tex(tmp_path, "x", _DOC_SIMPLE_TEX),
                "docY": _write_tex(tmp_path, "y", _DOC_A_TEX),
            }
        )
        config = _basic_config(["docX", "docY"])

        report = pipeline.run_ingest(config, source=source, ledger=ledger, manifest_path=manifest_path)

        assert report.n_seen == 2
        assert report.n_parsed == 2
        assert report.n_failed == 0
        assert report.n_chunks_created > 0
        assert report.parse_quality is not None
        assert len(ledger.live_documents()) == 2
        assert ledger.count_chunks() == report.n_chunks_created

    def test_document_row_records_which_parser_ran(self, data_dir, ledger, manifest_path, tmp_path):
        source = FixtureSource(latex_docs={"docX": _write_tex(tmp_path, "x", _DOC_SIMPLE_TEX)})
        pipeline.run_ingest(_basic_config(["docX"]), source=source, ledger=ledger, manifest_path=manifest_path)

        doc = ledger.get_document("docX")
        assert doc.parser_id == "latex"
        assert doc.title == "A Simple Paper"


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


class TestDedup:
    def test_near_duplicate_chunks_are_collapsed(self, data_dir, ledger, manifest_path, tmp_path):
        source = FixtureSource(
            latex_docs={
                "docA": _write_tex(tmp_path, "a", _DOC_A_TEX),
                "docB": _write_tex(tmp_path, "b", _DOC_B_TEX),
            }
        )
        config = _basic_config(
            ["docA", "docB"],
            chunker={"name": "fixed", "config": {"chunk_size": 100, "overlap": 0}},
            dedup=DedupConfig(jaccard_threshold=0.85, shingle_size=5),
        )

        report = pipeline.run_ingest(config, source=source, ledger=ledger, manifest_path=manifest_path)

        assert report.n_chunks_duplicate > 0

        b_chunk_ids = ledger.get_chunk_ids(doc_id="docB")
        dup_rows = ledger._conn.execute(
            "SELECT chunk_uid, dup_of FROM chunks WHERE doc_id = 'docB' AND dup_of IS NOT NULL"
        ).fetchall()
        assert dup_rows, "expected at least one of docB's chunks to point at an earlier duplicate"
        for chunk_uid, dup_of in dup_rows:
            assert chunk_uid in b_chunk_ids
            assert dup_of in ledger.get_chunk_ids(doc_id="docA")


# ---------------------------------------------------------------------------
# Dead-letter
# ---------------------------------------------------------------------------


class TestDeadLetter:
    def test_a_failing_document_is_recorded_and_the_run_continues(
        self, data_dir, ledger, manifest_path, tmp_path
    ):
        source = FixtureSource(
            latex_docs={
                "docGood": _write_tex(tmp_path, "good", _DOC_SIMPLE_TEX),
                # deliberately unparseable: main_tex_file points at a file
                # that does not exist under latex_dir, so parse_latex_source
                # raises LatexSourceError immediately.
                "docBad": (tmp_path / "bad", "does-not-exist.tex"),
            }
        )
        (tmp_path / "bad").mkdir()

        report = pipeline.run_ingest(
            _basic_config(["docGood", "docBad"]), source=source, ledger=ledger, manifest_path=manifest_path
        )

        assert report.n_parsed == 1
        assert report.n_failed == 1
        assert len(report.failures) == 1
        failure = report.failures[0]
        assert failure.doc_id == "docBad"
        assert failure.stage == "parse"

        reloaded = Manifest(manifest_path)
        assert reloaded.get("docBad").status == ManifestStatus.FAILED
        assert reloaded.get("docBad").failure_reason
        assert reloaded.get("docGood").status == ManifestStatus.PARSED

        # the good document was still fully ingested despite the other's failure
        assert len(ledger.live_documents()) == 1
        assert ledger.get_document("docGood") is not None

    def test_failures_are_also_recorded_in_the_ledger_errors_table(
        self, data_dir, ledger, manifest_path, tmp_path
    ):
        source = FixtureSource(latex_docs={"docBad": (tmp_path / "bad", "missing.tex")})
        (tmp_path / "bad").mkdir()

        pipeline.run_ingest(_basic_config(["docBad"]), source=source, ledger=ledger, manifest_path=manifest_path)

        rows = ledger._conn.execute("SELECT stage, message FROM errors WHERE stage = 'parse'").fetchall()
        assert len(rows) == 1
        assert "docBad" in rows[0][1]


# ---------------------------------------------------------------------------
# A retried, now-fixed document produces new chunk ids; untouched
# documents keep theirs -- the required "changed source document" test.
# Canonical text is write-once (freeze_canonical refuses to re-freeze a
# doc_id with different text), so this models a "changed document" the only
# way this architecture allows: a document that failed before any canonical
# text was ever frozen for it, fixed, and retried.
# ---------------------------------------------------------------------------


class TestRetryAfterFix:
    def test_fixed_document_gets_new_ids_unchanged_document_keeps_its_own(
        self, data_dir, ledger, manifest_path, tmp_path
    ):
        good_dir = _write_tex(tmp_path, "good", _DOC_SIMPLE_TEX)
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()

        source = FixtureSource(latex_docs={"docGood": good_dir, "docBroken": (bad_dir, "missing.tex")})
        config = _basic_config(["docGood", "docBroken"])

        first = pipeline.run_ingest(config, source=source, ledger=ledger, manifest_path=manifest_path)
        assert first.n_failed == 1
        good_ids_after_run1 = ledger.get_chunk_ids(doc_id="docGood")
        assert ledger.get_chunk_ids(doc_id="docBroken") == set()

        # "fix" docBroken: point the source at real, parseable content.
        source.latex_docs["docBroken"] = _write_tex(tmp_path, "fixed", _DOC_SIMPLE_TEX)

        second = pipeline.run_ingest(config, source=source, ledger=ledger, manifest_path=manifest_path)
        assert second.n_failed == 0

        assert ledger.get_chunk_ids(doc_id="docGood") == good_ids_after_run1
        fixed_ids = ledger.get_chunk_ids(doc_id="docBroken")
        assert fixed_ids  # the previously-failed document now has chunks
        assert fixed_ids.isdisjoint(good_ids_after_run1)


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------


class TestResumeFromManifestStatus:
    def test_already_parsed_document_is_not_reacquired(self, data_dir, ledger, manifest_path, tmp_path):
        source = FixtureSource(latex_docs={"docX": _write_tex(tmp_path, "x", _DOC_SIMPLE_TEX)})
        config = _basic_config(["docX"])

        pipeline.run_ingest(config, source=source, ledger=ledger, manifest_path=manifest_path)
        assert source.acquire_calls == ["docX"]

        pipeline.run_ingest(config, source=source, ledger=ledger, manifest_path=manifest_path)
        assert source.acquire_calls == ["docX"]  # not called a second time

    def test_acquired_but_not_parsed_document_resumes_without_reacquiring(
        self, data_dir, ledger, manifest_path, tmp_path
    ):
        """Simulates a process that died between acquire and parse on a
        previous run: the manifest says ACQUIRED, the raw source is already
        on disk under the deterministic dest_root/doc_id layout, and the
        source has no fixture registered for this doc_id at all -- if the
        pipeline called `source.acquire()` again it would raise."""
        dest_root = pipeline.default_dest_root()
        doc_dir = dest_root / "docResume"
        doc_dir.mkdir(parents=True)
        (doc_dir / "main.tex").write_text(_DOC_SIMPLE_TEX, encoding="utf-8")

        m = Manifest(manifest_path)
        m.seed_pending("docResume", "https://arxiv.org/abs/docResume")
        m.mark_acquired("docResume", source_url="https://arxiv.org/abs/docResume", content_hash="preexisting")
        m.save()

        source = FixtureSource()  # deliberately empty: acquire() would raise
        report = pipeline.run_ingest(
            _basic_config(["docResume"]), source=source, ledger=ledger, manifest_path=manifest_path
        )

        assert source.acquire_calls == []
        assert report.n_failed == 0
        assert report.n_parsed == 1
        assert ledger.get_document("docResume").parser_id == "latex"

    def test_manifest_tombstoned_document_is_skipped_not_reacquired(
        self, data_dir, ledger, manifest_path, tmp_path
    ):
        source = FixtureSource(latex_docs={"docX": _write_tex(tmp_path, "x", _DOC_SIMPLE_TEX)})
        config = _basic_config(["docX"])
        pipeline.run_ingest(config, source=source, ledger=ledger, manifest_path=manifest_path)

        m = Manifest(manifest_path)
        m.mark_tombstoned("docX")
        m.save()

        report = pipeline.run_ingest(config, source=source, ledger=ledger, manifest_path=manifest_path)

        assert source.acquire_calls == ["docX"]  # only the first run ever acquired it
        assert report.n_skipped == 1
        assert report.n_parsed == 0
        assert report.n_failed == 0


# ---------------------------------------------------------------------------
# The span assertion
# ---------------------------------------------------------------------------


class TestSpanAssertion:
    def test_matching_span_does_not_raise(self, data_dir):
        from selfrag.ingest import canonical
        from selfrag.ingest.latex import ParsedDocument

        canonical.freeze_canonical("docX", ParsedDocument(text="Hello world.", parser_id="latex"))
        chunk = Chunk(
            chunk_uid="c1",
            doc_id="docX",
            chunker_config_id="fixed@aaaaaaaa",
            char_start=0,
            char_end=5,
            text="Hello",
        )
        pipeline._assert_chunk_span_matches("docX", chunk)  # must not raise

    def test_mismatched_span_raises_span_integrity_error(self, data_dir):
        from selfrag.ingest import canonical
        from selfrag.ingest.latex import ParsedDocument

        canonical.freeze_canonical("docX", ParsedDocument(text="Hello world.", parser_id="latex"))
        # deliberately wrong: char_start/char_end resolve to "world" but the
        # chunk claims to hold "Hello"
        bad_chunk = Chunk(
            chunk_uid="c1",
            doc_id="docX",
            chunker_config_id="fixed@aaaaaaaa",
            char_start=6,
            char_end=11,
            text="Hello",
        )
        with pytest.raises(pipeline.SpanIntegrityError):
            pipeline._assert_chunk_span_matches("docX", bad_chunk)


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_writes_nothing(self, data_dir, manifest_path):
        config = _basic_config(["docA", "docB"])

        report = pipeline.run_ingest(config, manifest_path=manifest_path, dry_run=True)

        assert report.dry_run is True
        assert report.n_seen == 2
        assert not manifest_path.exists()  # Manifest.save() was never called

    def test_requires_neither_source_nor_ledger(self, data_dir, manifest_path):
        # must not raise even though source/ledger are both omitted
        pipeline.run_ingest(_basic_config(["docA"]), manifest_path=manifest_path, dry_run=True)

    def test_reports_already_done_documents_as_skipped(
        self, data_dir, ledger, manifest_path, tmp_path
    ):
        source = FixtureSource(latex_docs={"docX": _write_tex(tmp_path, "x", _DOC_SIMPLE_TEX)})
        config = _basic_config(["docX"])
        pipeline.run_ingest(config, source=source, ledger=ledger, manifest_path=manifest_path)

        report = pipeline.run_ingest(config, manifest_path=manifest_path, dry_run=True)
        assert report.n_skipped == 1


class TestRunIngestGuard:
    def test_raises_when_source_or_ledger_missing_and_not_dry_run(self, data_dir, manifest_path, ledger):
        with pytest.raises(ValueError):
            pipeline.run_ingest(_basic_config(["docA"]), ledger=ledger, manifest_path=manifest_path)
        with pytest.raises(ValueError):
            pipeline.run_ingest(
                _basic_config(["docA"]), source=FixtureSource(), manifest_path=manifest_path
            )


# ---------------------------------------------------------------------------
# PDF fallback / parser selection
# ---------------------------------------------------------------------------


class TestPdfFallback:
    def test_pdf_only_document_is_parsed_with_pdf_fallback(
        self, data_dir, ledger, manifest_path, tmp_path
    ):
        pdf_path = _make_pdf(tmp_path, "doc.pdf", [("A PDF-Only Paper", 20), ("Some body text.", 11)])
        source = FixtureSource(pdf_docs={"docPdf": pdf_path})

        report = pipeline.run_ingest(
            _basic_config(["docPdf"]), source=source, ledger=ledger, manifest_path=manifest_path
        )

        assert report.n_failed == 0
        doc = ledger.get_document("docPdf")
        assert doc.parser_id == "pdf_fallback"

    def test_latex_parse_failure_falls_back_to_pdf(self, data_dir, ledger, manifest_path, tmp_path):
        pdf_path = _make_pdf(tmp_path, "doc.pdf", [("Fallback Paper", 20), ("Body text here.", 11)])
        bad_dir = tmp_path / "badlatex"
        bad_dir.mkdir()
        source = FixtureSource(latex_docs={"docHybrid": (bad_dir, "missing.tex")}, pdf_docs={"docHybrid": pdf_path})

        report = pipeline.run_ingest(
            _basic_config(["docHybrid"]), source=source, ledger=ledger, manifest_path=manifest_path
        )

        assert report.n_failed == 0
        assert ledger.get_document("docHybrid").parser_id == "pdf_fallback"


# ---------------------------------------------------------------------------
# Standalone quality reporting
# ---------------------------------------------------------------------------


class TestComputeQualityReport:
    def test_returns_none_for_empty_ledger(self, data_dir, ledger):
        assert pipeline.compute_quality_report(ledger) is None

    def test_aggregates_across_live_documents(self, data_dir, ledger, manifest_path, tmp_path):
        source = FixtureSource(
            latex_docs={
                "docX": _write_tex(tmp_path, "x", _DOC_SIMPLE_TEX),
                "docY": _write_tex(tmp_path, "y", _DOC_A_TEX),
            }
        )
        pipeline.run_ingest(
            _basic_config(["docX", "docY"]), source=source, ledger=ledger, manifest_path=manifest_path
        )

        report = pipeline.compute_quality_report(ledger)
        assert report is not None
        assert "header_footer_contamination_rate" in report

    def test_filters_by_parser_id(self, data_dir, ledger, manifest_path, tmp_path):
        pdf_path = _make_pdf(tmp_path, "doc.pdf", [("PDF Paper", 20), ("Body.", 11)])
        source = FixtureSource(
            latex_docs={"docX": _write_tex(tmp_path, "x", _DOC_SIMPLE_TEX)},
            pdf_docs={"docPdf": pdf_path},
        )
        pipeline.run_ingest(
            _basic_config(["docX", "docPdf"]), source=source, ledger=ledger, manifest_path=manifest_path
        )

        assert pipeline.compute_quality_report(ledger, parser_id="latex") is not None
        assert pipeline.compute_quality_report(ledger, parser_id="pdf_fallback") is not None
        assert pipeline.compute_quality_report(ledger, parser_id="no_such_parser") is None


# ---------------------------------------------------------------------------
# Config validation flows through the registry, fast, before any document
# is touched.
# ---------------------------------------------------------------------------


class TestChunkerValidation:
    def test_unknown_chunker_name_raises_before_touching_any_document(
        self, data_dir, ledger, manifest_path, tmp_path
    ):
        source = FixtureSource(latex_docs={"docX": _write_tex(tmp_path, "x", _DOC_SIMPLE_TEX)})
        config = _basic_config(["docX"], chunker={"name": "no_such_chunker", "config": {}})

        with pytest.raises(KeyError):
            pipeline.run_ingest(config, source=source, ledger=ledger, manifest_path=manifest_path)

        assert source.acquire_calls == []  # failed before acquisition ever started
