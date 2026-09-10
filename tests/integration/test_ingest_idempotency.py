"""The Phase 1 exit criterion, proven end to end, without network.

CLAUDE.md's Phase 1 exit criterion, verbatim: **running ingest twice over
the same documents must produce ZERO new chunk ids.** This test ingests a
small fixture corpus -- two near-duplicate documents (to exercise dedup)
and one deliberately unparseable document (to exercise the dead-letter
path) -- through ``selfrag.ingest.pipeline.run_ingest`` twice, against the
same manifest and the same ledger, and checks every angle of "nothing new
happened the second time":

- the chunk id *set* is identical across both runs;
- the *row count* in the ledger's ``chunks`` table is not merely close to
  that set's size, it equals it exactly -- proof upserts replaced rather
  than duplicated;
- every document's canonical text hash is identical across both runs;
- the document row count is unchanged;
- the on-disk manifest gains zero new rows on the second pass;
- the fixture ``DocumentSource`` was never asked to re-acquire a document
  it already successfully acquired on the first pass.

No real HTTP client, no real arXiv endpoint, no real filesystem outside
``tmp_path`` -- the whole point of ``pipeline.DocumentSource`` being an
injectable abstraction (see ``pipeline``'s module docstring) is that this
kind of test can exist at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from selfrag.ingest import pipeline
from selfrag.ingest.manifest import Manifest, ManifestStatus
from selfrag.ledger import Ledger


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SELFRAG_DATA_DIR", str(tmp_path))
    return tmp_path


@dataclass
class FixtureSource:
    """No-network DocumentSource -- see tests/unit/test_pipeline.py for the
    identical helper; duplicated here (rather than imported) so this
    integration test stays self-contained and readable on its own."""

    latex_docs: dict[str, tuple[Path, str]] = field(default_factory=dict)
    acquire_calls: list[str] = field(default_factory=list)

    def acquire(self, doc_id: str, dest_root: Path) -> pipeline.AcquiredDocument:
        self.acquire_calls.append(doc_id)
        if doc_id not in self.latex_docs:
            raise RuntimeError(f"no fixture registered for {doc_id!r}")
        latex_dir, main_file = self.latex_docs[doc_id]
        return pipeline.AcquiredDocument(
            source_url=f"https://arxiv.org/abs/{doc_id}",
            content_hash=f"hash-{doc_id}",
            latex_dir=latex_dir,
            main_tex_file=main_file,
        )

    def acquire_pdf_fallback(self, doc_id: str, dest_root: Path) -> Path:
        raise RuntimeError(f"no PDF fallback available for {doc_id!r} in this fixture corpus")


def _write_tex(tmp_path: Path, name: str, content: str) -> tuple[Path, str]:
    d = tmp_path / name
    d.mkdir()
    (d / "main.tex").write_text(content, encoding="utf-8")
    return d, "main.tex"


_SHARED_PARAGRAPH = (
    "Dense retrieval encodes queries and documents into a shared embedding "
    "space and ranks candidates by vector similarity, in contrast to sparse "
    "lexical matching methods such as BM25."
)

# docA and docB share an identical opening paragraph and diverge afterward,
# modeling a related-work passage quoted verbatim across two papers -- see
# dedup.py's own module docstring for exactly this scenario.
_DOC_A_TEX = rf"""\documentclass{{article}}
\title{{Fixture Paper A}}
\begin{{document}}
\section{{Related Work}}
{_SHARED_PARAGRAPH}

\section{{Contribution}}
This fixture paper A contributes a synthetic benchmark used only to exercise the ingest pipeline's tests, never a real result.
\end{{document}}
"""

_DOC_B_TEX = rf"""\documentclass{{article}}
\title{{Fixture Paper B}}
\begin{{document}}
\section{{Related Work}}
{_SHARED_PARAGRAPH}

\section{{Contribution}}
This fixture paper B contributes a different synthetic benchmark, distinct from paper A's, again only for exercising the ingest pipeline's own tests.
\end{{document}}
"""


def _build_corpus(tmp_path: Path) -> tuple[FixtureSource, pipeline.IngestConfig]:
    source = FixtureSource(
        latex_docs={
            "fixtureA": _write_tex(tmp_path, "a", _DOC_A_TEX),
            "fixtureB": _write_tex(tmp_path, "b", _DOC_B_TEX),
            # deliberately unparseable: no main.tex was ever written under
            # this directory, so parse_latex_source raises LatexSourceError.
            "fixtureBad": (tmp_path / "bad", "main.tex"),
        }
    )
    (tmp_path / "bad").mkdir()
    config = pipeline.IngestConfig(
        corpus_name="idempotency-fixture",
        doc_ids=["fixtureA", "fixtureB", "fixtureBad"],
        chunker={"name": "fixed", "config": {"chunk_size": 100, "overlap": 0}},
    )
    return source, config


class TestIngestTwiceProducesZeroNewChunkIds:
    def test_full_idempotency_across_two_runs(self, data_dir, tmp_path):
        manifest_path = tmp_path / "manifest.parquet"
        ledger_path = tmp_path / "ledger.duckdb"
        source, config = _build_corpus(tmp_path)

        with Ledger(ledger_path) as ledger:
            first_report = pipeline.run_ingest(
                config, source=source, ledger=ledger, manifest_path=manifest_path
            )

            assert first_report.n_parsed == 2  # fixtureA, fixtureB
            assert first_report.n_failed == 1  # fixtureBad

            chunk_ids_after_first = ledger.get_chunk_ids()
            chunk_count_after_first = ledger.count_chunks()
            docs_after_first = {d.doc_id: d for d in ledger.live_documents()}
            hashes_after_first = {doc_id: d.doc_text_sha256 for doc_id, d in docs_after_first.items()}

            assert chunk_ids_after_first  # the happy-path documents produced real chunks
            assert chunk_count_after_first == len(chunk_ids_after_first)  # zero row-level duplication
            assert set(docs_after_first) == {"fixtureA", "fixtureB"}

            # --- second, identical pass -------------------------------------------------
            second_report = pipeline.run_ingest(
                config, source=source, ledger=ledger, manifest_path=manifest_path
            )

            chunk_ids_after_second = ledger.get_chunk_ids()
            chunk_count_after_second = ledger.count_chunks()
            docs_after_second = {d.doc_id: d for d in ledger.live_documents()}
            hashes_after_second = {doc_id: d.doc_text_sha256 for doc_id, d in docs_after_second.items()}

        # The exit criterion itself: the id SET is identical.
        assert chunk_ids_after_second == chunk_ids_after_first

        # No duplicate ROWS: row count still equals the (unchanged) id-set size.
        assert chunk_count_after_second == chunk_count_after_first == len(chunk_ids_after_first)

        # Canonical text hashes are byte-identical across both runs.
        assert hashes_after_second == hashes_after_first

        # Document rows: same set, same count -- no duplicates, nothing lost.
        assert set(docs_after_second) == set(docs_after_first)
        assert len(docs_after_second) == len(docs_after_first)

        # The second run still tried (and still failed) the dead-lettered
        # document -- FAILED status is retried every run by design -- but
        # produced no chunks or document rows for it either time.
        assert second_report.n_failed == 1
        assert "fixtureBad" not in docs_after_second

        # The manifest itself gained zero new rows.
        reloaded_manifest = Manifest(manifest_path)
        assert len(reloaded_manifest) == 3
        assert reloaded_manifest.get("fixtureA").status == ManifestStatus.PARSED
        assert reloaded_manifest.get("fixtureB").status == ManifestStatus.PARSED
        assert reloaded_manifest.get("fixtureBad").status == ManifestStatus.FAILED

        # Acquisition itself was never repeated for the documents that
        # already succeeded -- only the perpetually-failing one is retried.
        assert source.acquire_calls.count("fixtureA") == 1
        assert source.acquire_calls.count("fixtureB") == 1
        assert source.acquire_calls.count("fixtureBad") == 2  # retried both runs, by design

    def test_third_pass_is_still_a_no_op_for_chunk_ids(self, data_dir, tmp_path):
        """Idempotency has to hold indefinitely, not just once -- a second
        no-op pass proving stability is a weaker claim than a third one
        that starts from an already-twice-converged state."""
        manifest_path = tmp_path / "manifest.parquet"
        ledger_path = tmp_path / "ledger.duckdb"
        source, config = _build_corpus(tmp_path)

        with Ledger(ledger_path) as ledger:
            pipeline.run_ingest(config, source=source, ledger=ledger, manifest_path=manifest_path)
            pipeline.run_ingest(config, source=source, ledger=ledger, manifest_path=manifest_path)
            ids_after_two = ledger.get_chunk_ids()

            pipeline.run_ingest(config, source=source, ledger=ledger, manifest_path=manifest_path)
            ids_after_three = ledger.get_chunk_ids()

        assert ids_after_three == ids_after_two
