"""Core data models.

Field-level notes explain *why* a field exists wherever the reason is not
obvious, because several of these fields look redundant and are not.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Namespace(StrEnum):
    """Retrieval namespaces are never silently mixed.

    "papers" is the external corpus; "project" is this project's own decision
    log. Mixing them at retrieval time lets a stale agent-written experiment log
    answer a question about the literature, and lets the system manufacture
    consensus by citing restatements of its own earlier claims.
    """

    PAPERS = "papers"
    PROJECT = "project"


class Modality(StrEnum):
    TEXT = "text"
    FIGURE = "figure"
    TABLE = "table"
    PAGE_IMAGE = "page_image"


class Authorship(StrEnum):
    HUMAN = "human"
    AGENT = "agent"


class QueryStratum(StrEnum):
    """Answerable strata plus the four unanswerable strata.

    UNANSWERABLE_COUNTERFACTUAL is the important one: an answerable query whose
    supporting document has been removed from the index, everything else held
    constant. It is the only stratum that produces controlled near-miss
    unanswerables, and it supports a paired comparison against its answerable
    twin.
    """

    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    COMPARATIVE = "comparative"
    DEFINITIONAL = "definitional"
    NEGATION = "negation"
    UNANSWERABLE_COUNTERFACTUAL = "unanswerable_counterfactual"
    UNANSWERABLE_FALSE_PREMISE = "unanswerable_false_premise"
    UNANSWERABLE_TEMPORAL = "unanswerable_temporal"
    UNANSWERABLE_OFF_TOPIC = "unanswerable_off_topic"

    @property
    def is_unanswerable(self) -> bool:
        return self.value.startswith("unanswerable_")


class QuerySource(StrEnum):
    HUMAN = "human"
    SYNTHETIC = "synthetic"
    DOGFOOD = "dogfood"


class Split(StrEnum):
    """DEV is tuned against freely. TEST is opened exactly once, at the end."""

    DEV = "dev"
    TEST = "test"


class Document(BaseModel):
    doc_id: str  # arXiv base id, version-stripped
    namespace: Namespace
    source_url: str
    title: str = ""
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    updated_at: datetime | None = None

    latest_version: int | None = None
    versions_seen: list[int] = Field(default_factory=list)

    license: str = ""
    doc_text_sha256: str = ""  # hash of frozen canonical text; all offsets index into it
    parser_id: str = ""  # which parser produced the canonical text -- a ladder axis
    char_len: int = 0

    # Access control. The index stores only this key; the permission decision is
    # made at query time against the source of truth, because permissions drift
    # after ingest and the index never notices.
    acl_key: str = "public"

    # Project-namespace lineage. Used to collapse N agent-authored restatements
    # of one claim down to the weight of one piece of evidence at ranking time.
    authored_by: Authorship = Authorship.HUMAN
    derived_from: list[str] = Field(default_factory=list)
    superseded_by: str | None = None
    asserted_at: datetime | None = None

    ingest_run_id: str = ""
    tombstoned_at: datetime | None = None

    @property
    def is_live(self) -> bool:
        return self.tombstoned_at is None


class Chunk(BaseModel):
    chunk_uid: str
    doc_id: str
    chunker_config_id: str
    char_start: int
    char_end: int
    text: str

    token_count: int = 0
    section_path: str = ""  # e.g. "3 Method > 3.2 Retrieval" -- free from LaTeX
    raw_text_sha256: str = ""  # dedup + embedding-cache key; never part of chunk_uid

    # Contextual-retrieval output. Kept out of the identity function so it can be
    # regenerated without invalidating chunk ids or embedding-cache lineage.
    context_prefix: str | None = None
    context_model_id: str | None = None
    context_prompt_sha: str | None = None

    # Multimodal locators. Nullable now, populated in Phase 10. They exist today
    # because adding them later means rewriting every chunk id and every qrel.
    modality: Modality = Modality.TEXT
    page_no: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    asset_uri: str | None = None
    parent_chunk_uid: str | None = None

    dup_of: str | None = None  # near-duplicate resolution target

    @model_validator(mode="after")
    def _check_span(self) -> Chunk:
        if self.char_end <= self.char_start:
            raise ValueError(f"empty or inverted span [{self.char_start}, {self.char_end})")
        return self

    def embedding_text(self) -> str:
        """Exactly what gets embedded -- prefix included when present.

        Retrieval sees this. Relevance judges never do (see selfrag.eval.qrels).
        """
        if self.context_prefix:
            return f"{self.context_prefix}\n\n{self.text}"
        return self.text


class Query(BaseModel):
    query_id: str
    text: str
    namespace: Namespace
    stratum: QueryStratum
    source: QuerySource
    split: Split

    generator_model: str | None = None  # must differ from the HyDE/generation vendor
    removed_doc_id: str | None = None  # counterfactual stratum: doc pulled from the index
    paired_query_id: str | None = None  # its answerable twin

    created_at: datetime | None = None


class Qrel(BaseModel):
    """A relevance judgement over a document character span, never a chunk.

    Chunking is ablation axis #1. A qrel keyed to a chunk id would be
    invalidated by every chunker change, i.e. by the first experiment we run.
    """

    qrels_version: int
    query_id: str
    doc_id: str
    char_start: int
    char_end: int
    grade: int = Field(ge=0, le=3)
    judge: str  # "human:mustafa" | "pooled:<model_id>" | ...
    judged_at: datetime | None = None

    @model_validator(mode="after")
    def _check_span(self) -> Qrel:
        if self.char_end <= self.char_start:
            raise ValueError(f"empty or inverted gold span [{self.char_start}, {self.char_end})")
        return self


class Constraints(BaseModel):
    """Pre-registered hard limits. A config that violates one is disqualified
    regardless of quality score -- otherwise the ladder happily selects a winner
    that cannot be served."""

    max_p95_latency_ms: float = 800.0
    max_peak_rss_mb: float = 2560.0
    max_cost_per_query_usd: float = 0.02


class RunManifest(BaseModel):
    """Everything required to reproduce a run, hashed into run_id.

    embedder_snapshot_date is not redundant with embedder_id: hosted embedding
    models change silently under a stable name, so the name alone does not
    identify the model that produced a vector.
    """

    corpus_snapshot: str
    dedup_config_id: str
    parser_id: str
    chunker_config_id: str

    embedder_id: str
    embedder_snapshot_date: str
    embedding_dim: int
    quantization: str = "none"

    index_params: dict = Field(default_factory=dict)
    prefilter: bool = True  # postfiltering an ANN result set silently changes recall@k

    sparse_id: str | None = None
    fusion_id: str | None = None
    reranker_id: str | None = None
    rerank_top_k: int | None = None
    query_transform_id: str | None = None

    generator_id: str | None = None
    generator_snapshot_date: str | None = None
    prompt_sha: str | None = None
    judge_id: str | None = None

    qrels_version: int
    split: Split
    seed: int = 0
    constraints: Constraints = Field(default_factory=Constraints)

    code_git_sha: str = ""
    notes: str = ""

    def run_id(self) -> str:
        from selfrag.ids import config_hash

        return config_hash(self.model_dump(mode="json"))
