"""Component registry.

Every pipeline stage -- parser, chunker, embedder, sparse retriever, fusion,
reranker, query transform, generator, verifier -- is registered here under a
``(kind, name)`` pair together with a Pydantic config schema. That is what
lets a whole pipeline be described completely by a YAML file (see
``selfrag.cli config validate`` and ``configs/baseline.yaml``) and hashed
into a single ``run_id``: nothing about a component's behaviour is allowed to
live outside its declared config, because anything that does becomes an
untracked variable in every experiment that uses it.

``Component.config_id`` (``"<name>@<8-hex-char-hash>"``) is the load-bearing
string in this module. It flows directly into ``selfrag.ids.chunk_uid`` as
the ``chunker_config_id`` argument and into ``RunManifest`` as
``chunker_config_id`` / ``query_transform_id`` / etc. Two consequences follow
from that placement:

1. It must be *stable*: the same component name with the same config values,
   hashed the same way (``selfrag.ids.config_hash``, which canonicalises key
   order and int/float spelling), must always produce the same id -- on this
   machine, next week, on a teammate's machine.
2. It must be *sensitive*: changing any config value must change the id.
   Otherwise two chunkers with different chunk sizes would mint colliding
   ``chunk_uid``s, and a run comparison would silently compare incompatible
   chunk boundaries under one shared identity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, Field, ValidationError, model_validator

from selfrag.ids import chunk_uid, config_hash
from selfrag.schema import Chunk


class ComponentKind(StrEnum):
    """The nine pipeline-stage kinds a run manifest can reference.

    Fixed and small on purpose: a typo'd kind should fail loudly at
    registration time (see ``register``) rather than silently create a new,
    never-queried bucket in the registry.
    """

    PARSER = "parser"
    CHUNKER = "chunker"
    EMBEDDER = "embedder"
    SPARSE_RETRIEVER = "sparse_retriever"
    FUSION = "fusion"
    RERANKER = "reranker"
    QUERY_TRANSFORM = "query_transform"
    GENERATOR = "generator"
    VERIFIER = "verifier"


class Component(ABC):
    """Base class for every registered pipeline-stage implementation.

    A component is exactly the quadruple ``(kind, name, version, config)`` --
    nothing else about it is allowed to affect a run's identity. ``config_id``
    is implemented once, here, rather than per-subclass, so that guarantee
    cannot be broken by a component author overriding the formula.

    ``version`` exists because config values alone are not sufficient identity.
    ``config_id`` is fed into ``chunk_uid``; if a chunker's *implementation*
    changes while its config stays byte-identical, every chunk would keep its
    old id while covering different text. Two index generations would then
    silently share ids -- stale vectors surviving a re-index, tombstones not
    firing, and blue/green diffs reporting no change. That is exactly the
    failure class the id scheme exists to prevent, so any behaviour change to a
    component MUST bump its ``version``.
    """

    kind: ClassVar[str] = ""
    name: ClassVar[str] = ""
    version: ClassVar[int] = 1
    config_model: ClassVar[type[BaseModel]]

    def __init__(self, config: BaseModel) -> None:
        if not isinstance(config, self.config_model):
            raise TypeError(
                f"{type(self).__name__} expects a {self.config_model.__name__} instance, "
                f"got {type(config).__name__}"
            )
        self.config = config

    @property
    def config_id(self) -> str:
        """Short deterministic id: ``"<registered name>.v<version>@<8 hex>"``.

        The hash covers the component's *validated* config (``model_dump``
        after Pydantic has applied defaults and coercions), not the raw
        input dict, so two configs that only differ in which fields were
        left to their defaults still collapse to the same id.

        ``version`` is carried in the hash *and* rendered in the visible
        prefix. In the hash so that a version bump invalidates derived ids
        (chunk_uids, cache keys); in the prefix so that a human reading a
        ledger row can see which implementation produced a result without
        having to resolve the digest.
        """
        payload = {
            "name": self.name,
            "version": self.version,
            "config": self.config.model_dump(mode="json"),
        }
        digest = config_hash(payload)[:8]
        return f"{self.name}.v{self.version}@{digest}"


_Registry = dict[str, dict[str, type[Component]]]
_REGISTRY: _Registry = {}


def register(kind: str, name: str):
    """Class decorator: register a ``Component`` subclass under ``(kind, name)``.

    Raises:
        ValueError: ``kind`` is not one of ``ComponentKind``, or ``(kind,
            name)`` is already registered (a silent overwrite would let two
            components share an identity string without anyone noticing).
        TypeError: the class does not declare ``config_model`` as a
            ``BaseModel`` subclass -- every component must be validated by a
            real schema, never by convention.
    """

    valid_kinds = {k.value for k in ComponentKind}

    def decorator(cls: type[Component]) -> type[Component]:
        if kind not in valid_kinds:
            raise ValueError(f"unknown component kind {kind!r}; valid kinds: {sorted(valid_kinds)}")

        config_model = getattr(cls, "config_model", None)
        if not (isinstance(config_model, type) and issubclass(config_model, BaseModel)):
            raise TypeError(
                f"component {cls.__name__!r} must declare config_model as a "
                "pydantic.BaseModel subclass"
            )

        bucket = _REGISTRY.setdefault(kind, {})
        if name in bucket:
            raise ValueError(
                f"component {kind}.{name} is already registered (by {bucket[name].__name__})"
            )

        cls.kind = kind
        cls.name = name
        bucket[name] = cls
        return cls

    return decorator


def get(kind: str, name: str) -> type[Component]:
    """Look up a registered component class.

    Raises:
        KeyError: unknown kind or name, with the valid options for that kind
            spelled out in the message -- this is the error a mistyped
            pipeline YAML surfaces, so it has to be actionable on its own.
    """
    bucket = _REGISTRY.get(kind)
    if bucket is None:
        raise KeyError(f"unknown component kind {kind!r}; valid kinds: {list_kinds()}")
    try:
        return bucket[name]
    except KeyError:
        raise KeyError(
            f"unknown component {name!r} for kind {kind!r}; valid names: {list_components(kind)}"
        ) from None


def list_kinds() -> list[str]:
    """Kinds that currently have at least one registered component."""
    return sorted(_REGISTRY)


def list_components(kind: str) -> list[str]:
    """Registered component names for ``kind`` (empty list if none/unknown)."""
    return sorted(_REGISTRY.get(kind, {}))


def build_from_dict(kind: str, spec: dict[str, Any]) -> Component:
    """Validate ``spec`` against the named component's schema and construct it.

    ``spec`` is the ``{"name": ..., "config": {...}}`` shape used in pipeline
    YAML. This is the single choke point through which every stage of a
    pipeline config passes, so it is also the single place that has to raise
    a clear, actionable error for both failure modes: an unknown component
    name, and a structurally-known-but-invalid config for a real one.

    Raises:
        ValueError: ``spec`` has no ``"name"`` key, or the config fails
            validation against the component's ``config_model``.
        KeyError: unknown kind or component name (see ``get``).
    """
    if "name" not in spec:
        raise ValueError(f"component spec for kind={kind!r} is missing required key 'name'")
    name = spec["name"]
    cls = get(kind, name)
    raw_config = spec.get("config") or {}
    try:
        config = cls.config_model.model_validate(raw_config)
    except ValidationError as exc:
        raise ValueError(f"invalid config for {kind}.{name}: {exc}") from exc
    return cls(config)


# --------------------------------------------------------------------------
# Reference components.
#
# These exist so the registry is exercised end to end by something that
# actually works, not just by test doubles: a real chunker that produces
# real, round-trippable Chunk objects, and a real (if intentionally trivial)
# query transform. Every later chunker/query-transform implementation is
# just another (kind, name) pair alongside these two.
# --------------------------------------------------------------------------


class Chunker(Component, ABC):
    """Base for chunking components: canonical document text -> ``list[Chunk]``."""

    @abstractmethod
    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        pass


class FixedChunkerConfig(BaseModel):
    """Fixed-width character windows with optional overlap.

    ``overlap`` must be strictly smaller than ``chunk_size`` or the window
    would never advance and chunking would not terminate.
    """

    chunk_size: int = Field(512, gt=0, description="window width, in characters")
    overlap: int = Field(64, ge=0, description="characters shared between consecutive windows")

    @model_validator(mode="after")
    def _check_overlap(self) -> FixedChunkerConfig:
        if self.overlap >= self.chunk_size:
            raise ValueError(f"overlap ({self.overlap}) must be smaller than chunk_size ({self.chunk_size})")
        return self


@register(ComponentKind.CHUNKER.value, "fixed")
class FixedChunker(Chunker):
    """Reference chunker: non-overlapping-by-stride fixed character windows.

    Windows are ``chunk_size`` characters wide and advance by
    ``stride = chunk_size - overlap`` each step, so consecutive windows share
    exactly ``overlap`` characters and the union of all windows covers
    ``text[0:len(text)]`` with no gap. The last window is clipped to
    ``len(text)`` rather than padded, so ``chunk.text`` always round-trips
    exactly through ``text[chunk.char_start:chunk.char_end]``.
    """

    config_model = FixedChunkerConfig

    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        if not text:
            raise ValueError("cannot chunk empty text")

        size = self.config.chunk_size
        stride = size - self.config.overlap
        n = len(text)
        config_id = self.config_id

        out: list[Chunk] = []
        start = 0
        while start < n:
            end = min(start + size, n)
            out.append(
                Chunk(
                    chunk_uid=chunk_uid(doc_id, config_id, start, end),
                    doc_id=doc_id,
                    chunker_config_id=config_id,
                    char_start=start,
                    char_end=end,
                    text=text[start:end],
                )
            )
            if end == n:
                break
            start += stride
        return out


class QueryTransform(Component, ABC):
    """Base for query-transform components: one query in, one-or-more variants out.

    The list return type (rather than a single string) is the contract that
    lets later multi-query-expansion transforms slot in without changing
    the interface; passthrough is simply the one-element case.
    """

    @abstractmethod
    def transform(self, query: str) -> list[str]:
        pass


class PassthroughConfig(BaseModel):
    """Passthrough takes no configuration.

    A dedicated empty schema -- rather than making the transform stage
    optional/``None`` in a pipeline spec -- keeps "no transform" a first-class,
    hashable, registry-validated choice like every other stage.
    """


@register(ComponentKind.QUERY_TRANSFORM.value, "passthrough")
class PassthroughQueryTransform(QueryTransform):
    """Reference query transform: returns the query unchanged."""

    config_model = PassthroughConfig

    def transform(self, query: str) -> list[str]:
        return [query]
