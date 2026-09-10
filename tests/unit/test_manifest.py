"""Tests for selfrag.ingest.manifest.

The property every test here ultimately protects: re-running acquisition
over the same manifest never creates a duplicate row and never silently
loses a document, whether it succeeded, failed, or was deliberately
tombstoned.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from selfrag.ingest.manifest import Manifest, ManifestEntry, ManifestStatus


class TestSeedIdempotency:
    def test_seeding_the_same_doc_id_twice_adds_zero_new_rows(self, tmp_path):
        manifest = Manifest(tmp_path / "manifest.parquet")
        created_first = manifest.seed_pending("2401.00001", "https://arxiv.org/abs/2401.00001")
        created_second = manifest.seed_pending("2401.00001", "https://arxiv.org/abs/2401.00001")
        assert created_first is True
        assert created_second is False
        assert len(manifest) == 1

    def test_reseeding_does_not_regress_an_already_acquired_document(self, tmp_path):
        manifest = Manifest(tmp_path / "manifest.parquet")
        manifest.seed_pending("2401.00001", "https://arxiv.org/abs/2401.00001")
        manifest.mark_acquired("2401.00001", source_url="https://arxiv.org/abs/2401.00001", content_hash="abc")
        manifest.seed_pending("2401.00001", "https://arxiv.org/abs/2401.00001")
        assert manifest.get("2401.00001").status == ManifestStatus.ACQUIRED
        assert len(manifest) == 1


class TestAcquisitionIdempotency:
    def test_marking_acquired_twice_updates_in_place_not_a_second_row(self, tmp_path):
        manifest = Manifest(tmp_path / "manifest.parquet")
        manifest.mark_acquired("2401.00001", source_url="u", content_hash="h1")
        manifest.mark_acquired("2401.00001", source_url="u", content_hash="h1")
        assert len(manifest) == 1
        assert manifest.get("2401.00001").content_hash == "h1"

    def test_needs_acquisition_is_false_once_acquired(self, tmp_path):
        manifest = Manifest(tmp_path / "manifest.parquet")
        assert manifest.needs_acquisition("2401.00001") is True
        manifest.mark_acquired("2401.00001", source_url="u", content_hash="h1")
        assert manifest.needs_acquisition("2401.00001") is False

    def test_needs_acquisition_is_true_for_a_failed_document(self, tmp_path):
        manifest = Manifest(tmp_path / "manifest.parquet")
        manifest.mark_failed("2401.00001", "network timeout")
        assert manifest.needs_acquisition("2401.00001") is True

    def test_rerunning_acquisition_over_the_same_manifest_adds_zero_rows(self, tmp_path):
        """The scenario the task calls out directly: acquisition runs twice
        over the same on-disk manifest and must not re-download or duplicate."""
        path = tmp_path / "manifest.parquet"
        doc_ids = [f"2401.0000{i}" for i in range(5)]

        def run_acquisition_pass() -> tuple[int, list[str]]:
            manifest = Manifest(path)
            downloaded = []
            for doc_id in doc_ids:
                manifest.seed_pending(doc_id, f"https://arxiv.org/abs/{doc_id}")
                if manifest.needs_acquisition(doc_id):
                    downloaded.append(doc_id)
                    manifest.mark_acquired(
                        doc_id, source_url=f"https://arxiv.org/abs/{doc_id}", content_hash=f"hash-{doc_id}"
                    )
            manifest.save()
            return len(manifest), downloaded

        first_len, first_downloaded = run_acquisition_pass()
        second_len, second_downloaded = run_acquisition_pass()

        assert first_len == 5
        assert second_len == 5
        assert first_downloaded == doc_ids
        assert second_downloaded == []  # nothing re-downloaded the second time


class TestDeadLetter:
    def test_failed_document_is_recorded_with_its_reason_not_dropped(self, tmp_path):
        manifest = Manifest(tmp_path / "manifest.parquet")
        manifest.seed_pending("2401.00001", "u")
        manifest.mark_failed("2401.00001", "parser raised UnicodeDecodeError")
        entry = manifest.get("2401.00001")
        assert entry.status == ManifestStatus.FAILED
        assert entry.failure_reason == "parser raised UnicodeDecodeError"
        assert len(manifest) == 1

    def test_failed_document_survives_a_save_and_reload(self, tmp_path):
        path = tmp_path / "manifest.parquet"
        manifest = Manifest(path)
        manifest.mark_failed("2401.00001", "boom")
        manifest.save()

        reloaded = Manifest(path)
        entry = reloaded.get("2401.00001")
        assert entry is not None
        assert entry.status == ManifestStatus.FAILED
        assert entry.failure_reason == "boom"


class TestStatusTransitions:
    def test_full_lifecycle_pending_to_acquired_to_parsed(self, tmp_path):
        manifest = Manifest(tmp_path / "manifest.parquet")
        manifest.seed_pending("2401.00001", "u")
        assert manifest.get("2401.00001").status == ManifestStatus.PENDING

        manifest.mark_acquired("2401.00001", source_url="u", content_hash="h")
        assert manifest.get("2401.00001").status == ManifestStatus.ACQUIRED

        manifest.mark_parsed("2401.00001", parser_used="latex")
        entry = manifest.get("2401.00001")
        assert entry.status == ManifestStatus.PARSED
        assert entry.parser_used == "latex"
        assert len(manifest) == 1

    def test_mark_parsed_without_prior_acquisition_raises(self, tmp_path):
        manifest = Manifest(tmp_path / "manifest.parquet")
        with pytest.raises(KeyError):
            manifest.mark_parsed("2401.99999", parser_used="latex")

    def test_mark_tombstoned_without_existing_entry_raises(self, tmp_path):
        manifest = Manifest(tmp_path / "manifest.parquet")
        with pytest.raises(KeyError):
            manifest.mark_tombstoned("2401.99999")

    def test_tombstone_is_a_deliberate_removal_not_a_deletion(self, tmp_path):
        manifest = Manifest(tmp_path / "manifest.parquet")
        manifest.mark_acquired("2401.00001", source_url="u", content_hash="h")
        manifest.mark_tombstoned("2401.00001")
        entry = manifest.get("2401.00001")
        assert entry.status == ManifestStatus.TOMBSTONED
        assert len(manifest) == 1  # still present, just marked removed
        assert manifest.needs_acquisition("2401.00001") is False


class TestPersistence:
    def test_save_and_reload_round_trips_all_fields(self, tmp_path):
        path = tmp_path / "manifest.parquet"
        manifest = Manifest(path)
        acquired_at = datetime(2024, 1, 15, 12, 30, tzinfo=UTC)
        manifest.upsert(
            ManifestEntry(
                doc_id="2401.00001",
                version=2,
                source_url="https://arxiv.org/abs/2401.00001v2",
                content_hash="deadbeef",
                acquired_at=acquired_at,
                parser_used="latex",
                status=ManifestStatus.PARSED,
                failure_reason=None,
            )
        )
        manifest.save()

        reloaded = Manifest(path)
        entry = reloaded.get("2401.00001")
        assert entry.version == 2
        assert entry.source_url == "https://arxiv.org/abs/2401.00001v2"
        assert entry.content_hash == "deadbeef"
        assert entry.acquired_at == acquired_at
        assert entry.parser_used == "latex"
        assert entry.status == ManifestStatus.PARSED

    def test_empty_manifest_saves_and_reloads_cleanly(self, tmp_path):
        path = tmp_path / "manifest.parquet"
        manifest = Manifest(path)
        manifest.save()
        reloaded = Manifest(path)
        assert len(reloaded) == 0

    def test_upsert_reports_whether_the_row_was_new(self, tmp_path):
        manifest = Manifest(tmp_path / "manifest.parquet")
        is_new = manifest.upsert(ManifestEntry(doc_id="2401.00001"))
        is_new_again = manifest.upsert(ManifestEntry(doc_id="2401.00001", status=ManifestStatus.ACQUIRED))
        assert is_new is True
        assert is_new_again is False
        assert len(manifest) == 1


class TestSnapshotId:
    def test_snapshot_id_is_stable_across_reload(self, tmp_path):
        path = tmp_path / "manifest.parquet"
        manifest = Manifest(path)
        manifest.mark_acquired("2401.00001", source_url="u", content_hash="h1")
        manifest.mark_acquired("2401.00002", source_url="u", content_hash="h2")
        first_id = manifest.snapshot_id()
        manifest.save()

        reloaded = Manifest(path)
        assert reloaded.snapshot_id() == first_id

    def test_snapshot_id_is_independent_of_insertion_order(self, tmp_path):
        a = Manifest(tmp_path / "a.parquet")
        a.mark_acquired("2401.00001", source_url="u", content_hash="h1")
        a.mark_acquired("2401.00002", source_url="u", content_hash="h2")

        b = Manifest(tmp_path / "b.parquet")
        b.mark_acquired("2401.00002", source_url="u", content_hash="h2")
        b.mark_acquired("2401.00001", source_url="u", content_hash="h1")

        assert a.snapshot_id() == b.snapshot_id()

    def test_snapshot_id_changes_when_acquired_content_changes(self, tmp_path):
        manifest = Manifest(tmp_path / "manifest.parquet")
        manifest.mark_acquired("2401.00001", source_url="u", content_hash="h1")
        before = manifest.snapshot_id()
        manifest.mark_acquired("2401.00001", source_url="u", content_hash="h2-revised")
        after = manifest.snapshot_id()
        assert before != after

    def test_snapshot_id_ignores_pending_and_failed_documents(self, tmp_path):
        manifest = Manifest(tmp_path / "manifest.parquet")
        manifest.mark_acquired("2401.00001", source_url="u", content_hash="h1")
        baseline = manifest.snapshot_id()

        manifest.seed_pending("2401.00002", "u")
        manifest.mark_failed("2401.00003", "boom")
        assert manifest.snapshot_id() == baseline
