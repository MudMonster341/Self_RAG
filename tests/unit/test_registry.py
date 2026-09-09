"""Tests for selfrag.registry.

Covers the two things that make the registry trustworthy as the thing a
run_id is derived from: config_id stability (same config -> same id,
changed config -> different id) and clear, actionable errors for unknown
components and invalid configs. Also exercises the two shipped reference
components (fixed chunker, passthrough query transform) as real
implementations, not just registrations.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from selfrag import registry
from selfrag.registry import Component, register


class TestRegistryLookup:
    def test_list_kinds_includes_shipped_kinds(self):
        assert "chunker" in registry.list_kinds()
        assert "query_transform" in registry.list_kinds()

    def test_list_components_for_known_kind(self):
        assert "fixed" in registry.list_components("chunker")
        assert "passthrough" in registry.list_components("query_transform")

    def test_list_components_for_unknown_kind_is_empty_not_an_error(self):
        assert registry.list_components("no_such_kind") == []

    def test_get_known_component(self):
        cls = registry.get("chunker", "fixed")
        assert cls.name == "fixed"
        assert cls.kind == "chunker"

    def test_get_unknown_kind_raises_with_valid_kinds_listed(self):
        with pytest.raises(KeyError) as exc_info:
            registry.get("no_such_kind", "whatever")
        message = str(exc_info.value)
        assert "no_such_kind" in message
        assert "chunker" in message  # names a valid kind

    def test_get_unknown_name_raises_with_valid_names_listed(self):
        with pytest.raises(KeyError) as exc_info:
            registry.get("chunker", "no_such_component")
        message = str(exc_info.value)
        assert "no_such_component" in message
        assert "fixed" in message  # names the valid component


class TestRegisterDecorator:
    def test_register_rejects_unknown_kind(self):
        class DummyConfig(BaseModel):
            pass

        with pytest.raises(ValueError, match="unknown component kind"):
            @register("not_a_real_kind", "dummy_unknown_kind_component")
            class Dummy(Component):
                config_model = DummyConfig

    def test_register_rejects_missing_config_model(self):
        with pytest.raises(TypeError, match="config_model"):
            @register("chunker", "dummy_missing_config_model")
            class Dummy(Component):
                pass

    def test_register_rejects_duplicate_name(self):
        class DummyConfig(BaseModel):
            pass

        @register("chunker", "dummy_dup_test_component")
        class First(Component):
            config_model = DummyConfig

        with pytest.raises(ValueError, match="already registered"):
            @register("chunker", "dummy_dup_test_component")
            class Second(Component):
                config_model = DummyConfig

    def test_registered_component_sets_kind_and_name(self):
        class DummyConfig(BaseModel):
            value: int = 1

        @register("verifier", "dummy_kind_name_test_component")
        class Dummy(Component):
            config_model = DummyConfig

        assert Dummy.kind == "verifier"
        assert Dummy.name == "dummy_kind_name_test_component"


class TestConfigIdStability:
    def test_same_config_gives_same_id(self):
        a = registry.build_from_dict("chunker", {"name": "fixed", "config": {"chunk_size": 200, "overlap": 20}})
        b = registry.build_from_dict("chunker", {"name": "fixed", "config": {"chunk_size": 200, "overlap": 20}})
        assert a.config_id == b.config_id

    def test_changed_config_gives_different_id(self):
        a = registry.build_from_dict("chunker", {"name": "fixed", "config": {"chunk_size": 200, "overlap": 20}})
        b = registry.build_from_dict("chunker", {"name": "fixed", "config": {"chunk_size": 300, "overlap": 20}})
        assert a.config_id != b.config_id

    def test_changed_overlap_alone_gives_different_id(self):
        a = registry.build_from_dict("chunker", {"name": "fixed", "config": {"chunk_size": 200, "overlap": 20}})
        b = registry.build_from_dict("chunker", {"name": "fixed", "config": {"chunk_size": 200, "overlap": 30}})
        assert a.config_id != b.config_id

    def test_config_id_format_is_name_dot_version_at_hash(self):
        component = registry.build_from_dict("chunker", {"name": "fixed", "config": {}})
        label, _, digest = component.config_id.partition("@")
        assert label == "fixed.v1"
        assert len(digest) == 8
        int(digest, 16)  # valid hex

    def test_version_bump_changes_the_id_even_with_identical_config(self):
        """The bug this prevents is the nastiest one the id scheme can have.

        config_id is fed into chunk_uid. If a chunker's implementation changes
        but its config does not, every chunk would keep its old id while now
        covering different text: two index generations silently sharing ids,
        stale vectors surviving a re-index, tombstones never firing, and
        blue/green diffs reporting no change.
        """

        class Cfg(BaseModel):
            size: int = 10

        @register("chunker", "versioned_probe_v1")
        class ProbeV1(Component):
            version = 1
            config_model = Cfg

        @register("chunker", "versioned_probe_v2")
        class ProbeV2(Component):
            version = 2
            config_model = Cfg

        cfg = Cfg()
        # Same config object, same field values -- only the declared version differs.
        v1_digest = ProbeV1(cfg).config_id.partition("@")[2]
        v2_digest = ProbeV2(cfg).config_id.partition("@")[2]
        assert v1_digest != v2_digest, "version must participate in the hash, not just the label"

    def test_version_is_visible_without_resolving_the_digest(self):
        """A human reading a ledger row should see which implementation ran."""

        class Cfg(BaseModel):
            size: int = 10

        @register("chunker", "versioned_probe_visible")
        class Probe(Component):
            version = 7
            config_model = Cfg

        assert Probe(Cfg()).config_id.startswith("versioned_probe_visible.v7@")

    def test_default_config_values_are_explicit_in_the_id(self):
        """Omitting a field in the YAML and spelling out its default must
        produce the same id -- the hash is over the *validated* config."""
        explicit = registry.build_from_dict(
            "chunker", {"name": "fixed", "config": {"chunk_size": 512, "overlap": 64}}
        )
        implicit = registry.build_from_dict("chunker", {"name": "fixed", "config": {}})
        assert explicit.config_id == implicit.config_id


class TestBuildFromDict:
    def test_missing_name_key_raises_value_error(self):
        with pytest.raises(ValueError, match="name"):
            registry.build_from_dict("chunker", {"config": {}})

    def test_unknown_component_name_raises_key_error(self):
        with pytest.raises(KeyError):
            registry.build_from_dict("chunker", {"name": "does_not_exist"})

    def test_invalid_config_raises_value_error_naming_the_component(self):
        with pytest.raises(ValueError) as exc_info:
            registry.build_from_dict(
                "chunker", {"name": "fixed", "config": {"chunk_size": 10, "overlap": 10}}
            )
        message = str(exc_info.value)
        assert "chunker.fixed" in message

    def test_invalid_config_type_raises_value_error(self):
        with pytest.raises(ValueError):
            registry.build_from_dict("chunker", {"name": "fixed", "config": {"chunk_size": "not a number"}})


class TestFixedChunker:
    def test_produces_non_overlapping_by_stride_offsets(self):
        component = registry.build_from_dict(
            "chunker", {"name": "fixed", "config": {"chunk_size": 10, "overlap": 3}}
        )
        text = "abcdefghijklmnopqrstuvwxyz"  # 26 chars
        chunks = component.chunk("doc1", text)

        stride = 10 - 3
        for i in range(1, len(chunks)):
            assert chunks[i].char_start - chunks[i - 1].char_start == stride

    def test_chunks_round_trip_through_source_text(self):
        component = registry.build_from_dict(
            "chunker", {"name": "fixed", "config": {"chunk_size": 10, "overlap": 3}}
        )
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = component.chunk("doc1", text)
        for chunk in chunks:
            assert chunk.text == text[chunk.char_start : chunk.char_end]

    def test_full_coverage_no_gaps(self):
        component = registry.build_from_dict(
            "chunker", {"name": "fixed", "config": {"chunk_size": 10, "overlap": 3}}
        )
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = component.chunk("doc1", text)
        assert chunks[0].char_start == 0
        assert chunks[-1].char_end == len(text)
        # every character index is covered by at least one chunk
        covered = set()
        for chunk in chunks:
            covered.update(range(chunk.char_start, chunk.char_end))
        assert covered == set(range(len(text)))

    def test_last_chunk_is_clipped_not_padded(self):
        component = registry.build_from_dict(
            "chunker", {"name": "fixed", "config": {"chunk_size": 10, "overlap": 0}}
        )
        text = "a" * 25  # 25 chars: windows [0,10) [10,20) [20,25)
        chunks = component.chunk("doc1", text)
        assert [c.char_end - c.char_start for c in chunks] == [10, 10, 5]

    def test_short_text_produces_single_chunk(self):
        component = registry.build_from_dict(
            "chunker", {"name": "fixed", "config": {"chunk_size": 512, "overlap": 64}}
        )
        chunks = component.chunk("doc1", "short text")
        assert len(chunks) == 1
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == len("short text")

    def test_empty_text_raises(self):
        component = registry.build_from_dict("chunker", {"name": "fixed", "config": {}})
        with pytest.raises(ValueError):
            component.chunk("doc1", "")

    def test_chunk_uid_matches_selfrag_ids_chunk_uid(self):
        from selfrag.ids import chunk_uid

        component = registry.build_from_dict(
            "chunker", {"name": "fixed", "config": {"chunk_size": 10, "overlap": 0}}
        )
        chunks = component.chunk("doc1", "abcdefghijklmnop")
        for chunk in chunks:
            expected = chunk_uid("doc1", component.config_id, chunk.char_start, chunk.char_end)
            assert chunk.chunk_uid == expected

    def test_invalid_overlap_config_rejected_at_validation(self):
        with pytest.raises(ValidationError):
            from selfrag.registry import FixedChunkerConfig

            FixedChunkerConfig(chunk_size=10, overlap=10)


class TestPassthroughQueryTransform:
    def test_returns_query_unchanged_as_single_element_list(self):
        component = registry.build_from_dict("query_transform", {"name": "passthrough", "config": {}})
        assert component.transform("what is retrieval augmented generation?") == [
            "what is retrieval augmented generation?"
        ]

    def test_config_id_is_stable_and_has_no_config_fields(self):
        a = registry.build_from_dict("query_transform", {"name": "passthrough", "config": {}})
        b = registry.build_from_dict("query_transform", {"name": "passthrough"})
        assert a.config_id == b.config_id


class TestComponentIsAbstractBase:
    def test_component_subclasses_must_implement_kind_specific_methods(self):
        """Chunker/QueryTransform declare @abstractmethod stage entry points;
        a subclass that skips them cannot be instantiated."""
        from selfrag.registry import Chunker

        class IncompleteConfig(BaseModel):
            pass

        class Incomplete(Chunker):
            config_model = IncompleteConfig

        with pytest.raises(TypeError):
            Incomplete(IncompleteConfig())
