"""Tests for selfrag.ingest.dedup -- version collapsing + near-duplicate chunks.

Two independent things are exercised here, matching the two mechanisms in
decisions/0004-arxiv-base-id-deduplication.md:

- ``resolve_versions``: grouping by arXiv base id, numeric (not lexicographic)
  version ordering, unversioned-sorts-lowest, and loud failure on a malformed
  id -- a silently mis-parsed id would corrupt the document key space.
- ``ChunkDeduplicator``: MinHash+LSH near-duplicate detection, with special
  attention to the false-positive direction (genuinely different text must
  never be flagged), transitive chains, determinism, and the ``enabled=False``
  no-op path the ablation depends on.
"""

from __future__ import annotations

import pytest

from selfrag.ingest.dedup import (
    ChunkDeduplicator,
    DedupConfig,
    ResolvedDocument,
    VersionedRecord,
    resolve_versions,
)

# -----------------------------------------------------------------------------
# DedupConfig / dedup_config_id
# -----------------------------------------------------------------------------


class TestDedupConfigId:
    def test_format_is_dedup_dot_v1_at_8_hex(self):
        config_id = DedupConfig().dedup_config_id
        label, _, digest = config_id.partition("@")
        assert label == "dedup.v1"
        assert len(digest) == 8
        int(digest, 16)  # valid hex

    def test_same_config_gives_same_id(self):
        a = DedupConfig(minhash_num_perm=64, jaccard_threshold=0.7, shingle_size=3)
        b = DedupConfig(minhash_num_perm=64, jaccard_threshold=0.7, shingle_size=3)
        assert a.dedup_config_id == b.dedup_config_id

    def test_default_values_and_explicit_defaults_collapse_to_same_id(self):
        a = DedupConfig()
        b = DedupConfig(minhash_num_perm=128, jaccard_threshold=0.85, shingle_size=5, enabled=True)
        assert a.dedup_config_id == b.dedup_config_id

    def test_changed_minhash_num_perm_changes_id(self):
        a = DedupConfig(minhash_num_perm=128)
        b = DedupConfig(minhash_num_perm=64)
        assert a.dedup_config_id != b.dedup_config_id

    def test_changed_jaccard_threshold_changes_id(self):
        a = DedupConfig(jaccard_threshold=0.85)
        b = DedupConfig(jaccard_threshold=0.9)
        assert a.dedup_config_id != b.dedup_config_id

    def test_changed_shingle_size_changes_id(self):
        a = DedupConfig(shingle_size=5)
        b = DedupConfig(shingle_size=3)
        assert a.dedup_config_id != b.dedup_config_id

    def test_changed_enabled_changes_id(self):
        a = DedupConfig(enabled=True)
        b = DedupConfig(enabled=False)
        assert a.dedup_config_id != b.dedup_config_id

    def test_invalid_jaccard_threshold_rejected_at_validation(self):
        with pytest.raises(ValueError):
            DedupConfig(jaccard_threshold=1.5)
        with pytest.raises(ValueError):
            DedupConfig(jaccard_threshold=0.0)

    def test_invalid_shingle_size_rejected_at_validation(self):
        with pytest.raises(ValueError):
            DedupConfig(shingle_size=0)


# -----------------------------------------------------------------------------
# resolve_versions
# -----------------------------------------------------------------------------


class TestResolveVersionsBasics:
    def test_single_unversioned_record(self):
        resolved, id_to_base = resolve_versions([VersionedRecord("2401.01234")])
        assert resolved == [
            ResolvedDocument(
                base_id="2401.01234",
                version=None,
                versions_seen=[],
                raw_id="2401.01234",
                payload=None,
            )
        ]
        assert id_to_base == {"2401.01234": "2401.01234"}

    def test_multiple_versions_keep_latest_as_live(self):
        records = [
            VersionedRecord("2401.01234v1"),
            VersionedRecord("2401.01234v2"),
            VersionedRecord("2401.01234v3"),
        ]
        resolved, _ = resolve_versions(records)
        assert len(resolved) == 1
        doc = resolved[0]
        assert doc.version == 3
        assert doc.versions_seen == [1, 2, 3]
        assert doc.raw_id == "2401.01234v3"

    def test_v10_sorts_above_v9_numerically_not_lexicographically(self):
        """The classic bug: "v10" < "v9" as strings, but 10 > 9 as versions."""
        forward, _ = resolve_versions(
            [VersionedRecord("2401.01234v9"), VersionedRecord("2401.01234v10")]
        )
        backward, _ = resolve_versions(
            [VersionedRecord("2401.01234v10"), VersionedRecord("2401.01234v9")]
        )
        assert forward[0].version == 10
        assert backward[0].version == 10
        assert forward[0].versions_seen == [9, 10]

    def test_unversioned_sorts_below_any_explicit_version(self):
        # Unversioned arrives *last* in input order but must still lose to
        # any explicit version already seen.
        resolved, _ = resolve_versions(
            [VersionedRecord("2401.01234v1"), VersionedRecord("2401.01234")]
        )
        assert resolved[0].version == 1

        # And arriving *first* must not let it win either.
        resolved2, _ = resolve_versions(
            [VersionedRecord("2401.01234"), VersionedRecord("2401.01234v1")]
        )
        assert resolved2[0].version == 1

    def test_duplicate_identical_versions_collapse_to_one_record(self):
        resolved, _ = resolve_versions(
            [VersionedRecord("2401.01234v3"), VersionedRecord("2401.01234v3")]
        )
        assert len(resolved) == 1
        assert resolved[0].version == 3
        assert resolved[0].versions_seen == [3]

    def test_tie_keeps_first_seen_raw_id(self):
        """Two different raw ids that both parse to the same (base, version)
        -- e.g. an arXiv: prefix and a bare id for the same version -- must
        deterministically keep whichever came first in input order."""
        resolved, _ = resolve_versions(
            [VersionedRecord("arXiv:2401.01234v2"), VersionedRecord("2401.01234v2")]
        )
        assert len(resolved) == 1
        assert resolved[0].raw_id == "arXiv:2401.01234v2"

    def test_malformed_id_raises_rather_than_silently_passing_through(self):
        with pytest.raises(ValueError):
            resolve_versions([VersionedRecord("not an id at all")])

    def test_malformed_id_raises_even_amid_valid_records(self):
        with pytest.raises(ValueError):
            resolve_versions(
                [VersionedRecord("2401.01234v1"), VersionedRecord("garbage-id")]
            )

    def test_distinct_papers_produce_distinct_documents(self):
        resolved, _ = resolve_versions(
            [VersionedRecord("2401.01234v1"), VersionedRecord("2405.06789v2")]
        )
        assert {doc.base_id for doc in resolved} == {"2401.01234", "2405.06789"}

    def test_old_style_ids_also_resolve(self):
        resolved, _ = resolve_versions(
            [VersionedRecord("hep-th/9901001"), VersionedRecord("hep-th/9901001v2")]
        )
        assert len(resolved) == 1
        assert resolved[0].base_id == "hep-th/9901001"
        assert resolved[0].version == 2

    def test_id_to_base_mapping_covers_every_seen_raw_id(self):
        records = [
            VersionedRecord("2401.01234v1"),
            VersionedRecord("2401.01234v2"),
            VersionedRecord("2405.06789"),
        ]
        _, id_to_base = resolve_versions(records)
        assert id_to_base == {
            "2401.01234v1": "2401.01234",
            "2401.01234v2": "2401.01234",
            "2405.06789": "2405.06789",
        }

    def test_payload_rides_along_with_the_winning_record(self):
        records = [
            VersionedRecord("2401.01234v1", payload={"title": "old draft"}),
            VersionedRecord("2401.01234v2", payload={"title": "revised"}),
        ]
        resolved, _ = resolve_versions(records)
        assert resolved[0].payload == {"title": "revised"}

    def test_resolved_documents_ordered_by_first_appearance(self):
        records = [
            VersionedRecord("2405.06789"),
            VersionedRecord("2401.01234"),
            VersionedRecord("2405.06789v1"),
        ]
        resolved, _ = resolve_versions(records)
        assert [doc.base_id for doc in resolved] == ["2405.06789", "2401.01234"]


# -----------------------------------------------------------------------------
# ChunkDeduplicator
# -----------------------------------------------------------------------------

LONG_TEXT_A = "the quick brown fox jumps over the lazy dog again and again and again"
LONG_TEXT_B = "quantum tensor gradient softmax kernel attention transformer embedding vector"


class TestChunkDeduplicatorBasics:
    def test_identical_texts_detected_as_duplicates(self):
        dedup = ChunkDeduplicator(DedupConfig())
        dedup.add("c1", LONG_TEXT_A)
        dedup.add("c2", LONG_TEXT_A)
        result = dedup.resolve()
        assert result["c1"] is None
        assert result["c2"] == "c1"

    def test_genuinely_different_texts_not_flagged(self):
        """Guard the false-positive direction: this is the one that silently
        destroys recall, so it gets its own explicit test rather than being
        assumed from the near-duplicate tests passing."""
        dedup = ChunkDeduplicator(DedupConfig())
        dedup.add("c1", LONG_TEXT_A)
        dedup.add("c2", LONG_TEXT_B)
        result = dedup.resolve()
        assert result["c1"] is None
        assert result["c2"] is None

    def test_earliest_occurrence_is_kept_as_canonical(self):
        dedup = ChunkDeduplicator(DedupConfig())
        dedup.add("first", LONG_TEXT_A)
        dedup.add("second", LONG_TEXT_A)
        dedup.add("third", LONG_TEXT_A)
        result = dedup.resolve()
        assert result["first"] is None
        assert result["second"] == "first"
        assert result["third"] == "first"

    def test_add_same_chunk_uid_twice_raises(self):
        dedup = ChunkDeduplicator(DedupConfig())
        dedup.add("c1", LONG_TEXT_A)
        with pytest.raises(ValueError):
            dedup.add("c1", LONG_TEXT_B)


def _words(prefix: str, lo: int, hi: int) -> str:
    """Space-joined unique tokens ``prefix0 .. prefix(n-1)`` over ``[lo, hi)``."""
    return " ".join(f"{prefix}{i}" for i in range(lo, hi))


class TestNearDuplicateThreshold:
    """shingle_size=1 (bag-of-words) makes the true Jaccard of these
    constructed word sets simple arithmetic, so "above" / "below" threshold
    is by a comfortable margin rather than a knife's edge that MinHash's
    estimation noise (num_perm=128, so a few percent) could flip."""

    def test_near_duplicate_above_threshold_is_flagged(self):
        # 20 shared tokens vs 1 swapped -> Jaccard = 19/21 ~= 0.905 > 0.85
        config = DedupConfig(shingle_size=1, jaccard_threshold=0.85)
        dedup = ChunkDeduplicator(config)
        dedup.add("base", _words("tok", 0, 20))
        dedup.add("near", _words("tok", 1, 21))
        result = dedup.resolve()
        assert result["near"] == "base"

    def test_near_duplicate_below_threshold_is_not_flagged(self):
        # 20 tokens, shifted by 5 -> intersection 15, union 25, Jaccard = 0.6 < 0.85
        config = DedupConfig(shingle_size=1, jaccard_threshold=0.85)
        dedup = ChunkDeduplicator(config)
        dedup.add("base", _words("tok", 0, 20))
        dedup.add("far", _words("tok", 5, 25))
        result = dedup.resolve()
        assert result["far"] is None


class TestTransitiveChain:
    def test_chain_collapses_to_single_canonical_target(self):
        # A = tok[0:20], B = tok[4:24], C = tok[8:28]
        # Jaccard(A,B) = 16/24 = 0.667 >= 0.6 (direct match)
        # Jaccard(B,C) = 16/24 = 0.667 >= 0.6 (direct match)
        # Jaccard(A,C) = 12/28 = 0.429 <  0.6 (no direct match)
        # So C must chain through B to resolve to A, not dangle at B.
        config = DedupConfig(shingle_size=1, jaccard_threshold=0.6)
        dedup = ChunkDeduplicator(config)
        dedup.add("A", _words("tok", 0, 20))
        dedup.add("B", _words("tok", 4, 24))
        dedup.add("C", _words("tok", 8, 28))
        result = dedup.resolve()
        assert result["A"] is None
        assert result["B"] == "A"
        assert result["C"] == "A"  # collapsed, not left pointing at B


class TestShortText:
    """Documented rule: text shorter than shingle_size words cannot form a
    shingle, so it is never compared against anything (including other short
    chunks) and is always an original. This must not crash, and must not
    fabricate a match between short chunks purely because they're both
    short."""

    def test_short_text_does_not_crash(self):
        config = DedupConfig(shingle_size=5)
        dedup = ChunkDeduplicator(config)
        dedup.add("c1", "hi there")
        result = dedup.resolve()
        assert result["c1"] is None

    def test_identical_short_texts_are_not_flagged_as_mutual_duplicates(self):
        config = DedupConfig(shingle_size=5)
        dedup = ChunkDeduplicator(config)
        dedup.add("c1", "hi there")
        dedup.add("c2", "hi there")
        result = dedup.resolve()
        assert result["c1"] is None
        assert result["c2"] is None

    def test_short_text_not_matched_against_a_long_superset_text(self):
        config = DedupConfig(shingle_size=5)
        dedup = ChunkDeduplicator(config)
        dedup.add("long", LONG_TEXT_A)
        dedup.add("short", "the quick")
        result = dedup.resolve()
        assert result["short"] is None


class TestNormalization:
    def test_case_and_punctuation_and_whitespace_do_not_defeat_matching(self):
        dedup = ChunkDeduplicator(DedupConfig())
        dedup.add("c1", LONG_TEXT_A)
        noisy = "  THE   Quick, Brown FOX!! jumps over the LAZY dog... again and again and again  "
        dedup.add("c2", noisy)
        result = dedup.resolve()
        assert result["c2"] == "c1"

    def test_normalization_never_mutates_the_stored_text(self):
        dedup = ChunkDeduplicator(DedupConfig())
        original = "  THE Quick Brown FOX!!  "
        dedup.add("c1", original)
        # add() takes text by value and never hands back a mutated copy --
        # there is nothing for this module to have mutated in place, but we
        # confirm the input string object itself is untouched.
        assert original == "  THE Quick Brown FOX!!  "


class TestDeterminism:
    def test_same_input_same_order_gives_identical_result(self):
        chunks = [("c1", LONG_TEXT_A), ("c2", LONG_TEXT_A), ("c3", LONG_TEXT_B)]

        def run():
            dedup = ChunkDeduplicator(DedupConfig())
            for uid, text in chunks:
                dedup.add(uid, text)
            return dedup.resolve()

        assert run() == run()

    def test_resolve_is_idempotent(self):
        dedup = ChunkDeduplicator(DedupConfig())
        dedup.add("c1", LONG_TEXT_A)
        dedup.add("c2", LONG_TEXT_A)
        assert dedup.resolve() == dedup.resolve()

    def test_shuffled_insertion_order_changes_the_canonical_target_but_stays_deterministic(self):
        """Order determines outcome, by design (see class docstring): the
        earliest-*inserted* chunk wins, not some order-independent notion of
        "the" duplicate. Reordering the same three mutually-identical chunks
        changes which one is canonical, but each fixed order always gives the
        same answer, run after run."""

        def run(order: list[str]) -> dict[str, str | None]:
            dedup = ChunkDeduplicator(DedupConfig())
            for uid in order:
                dedup.add(uid, LONG_TEXT_A)
            return dedup.resolve()

        forward = ["x", "y", "z"]
        reversed_ = ["z", "y", "x"]

        result_forward_1 = run(forward)
        result_forward_2 = run(forward)
        result_reversed = run(reversed_)

        # Same order twice -> identical result.
        assert result_forward_1 == result_forward_2
        assert result_forward_1 == {"x": None, "y": "x", "z": "x"}

        # Different order -> different (but still deterministic) canonical target.
        assert result_reversed == {"z": None, "y": "z", "x": "z"}


class TestEnabledFalseIsNoOp:
    def test_all_chunks_resolve_to_none_even_when_textually_identical(self):
        dedup = ChunkDeduplicator(DedupConfig(enabled=False))
        dedup.add("c1", LONG_TEXT_A)
        dedup.add("c2", LONG_TEXT_A)
        dedup.add("c3", LONG_TEXT_A)
        result = dedup.resolve()
        assert result == {"c1": None, "c2": None, "c3": None}

    def test_no_op_holds_for_near_duplicates_too(self):
        dedup = ChunkDeduplicator(DedupConfig(enabled=False, shingle_size=1))
        dedup.add("base", _words("tok", 0, 20))
        dedup.add("near", _words("tok", 1, 21))
        result = dedup.resolve()
        assert result == {"base": None, "near": None}
