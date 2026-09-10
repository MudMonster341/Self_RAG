"""The ``selfrag`` command-line app.

Every command here talks only to modules that already exist and already
work (``paths``, ``registry``, ``ledger``) -- nothing is stubbed out for a
future phase. A command that cannot yet do anything real is not included
rather than included as a placeholder.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from pathlib import Path
from typing import Annotated

import psutil
import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from selfrag import paths, registry
from selfrag.ingest import pipeline
from selfrag.ingest.arxiv_client import ArxivClient
from selfrag.ingest.manifest import Manifest
from selfrag.ledger import Ledger
from selfrag.schema import RunManifest

app = typer.Typer(
    name="selfrag",
    help="A production-grade, falsifiable RAG system over a self-referential research corpus.",
    no_args_is_help=True,
)
registry_app = typer.Typer(help="Inspect the component registry.", no_args_is_help=True)
ledger_app = typer.Typer(help="Inspect the experiment ledger.", no_args_is_help=True)
config_app = typer.Typer(help="Validate pipeline configuration YAML.", no_args_is_help=True)
ingest_app = typer.Typer(help="Corpus ingest: acquire, parse, chunk, persist.", no_args_is_help=True)
app.add_typer(registry_app, name="registry")
app.add_typer(ledger_app, name="ledger")
app.add_typer(config_app, name="config")
app.add_typer(ingest_app, name="ingest")

# arXiv's own parser_id constants (selfrag.ingest.latex.PARSER_ID /
# selfrag.ingest.pdf_fallback.PARSER_ID), duplicated here only as CLI-facing
# shorthand so `--parser latex|pdf` reads naturally; the actual values
# compared against ledger rows always come from those modules.
_PARSER_ID_ALIASES = {"latex": "latex", "pdf": "pdf_fallback"}

console = Console()
error_console = Console(stderr=True, style="bold red")

_MIN_AVAILABLE_RAM_GB = 3.0
_MIN_FREE_DISK_PCT = 5.0

# Representative import for each optional extra group in pyproject.toml,
# used by `doctor` to report which extras are actually installed rather than
# just which the pyproject *declares*.
_OPTIONAL_EXTRAS = {
    "retrieval": "bm25s",
    "parse": "pymupdf4llm",
    "serve": "fastapi",
    "dev": "pytest",
}

# Maps a registry component "kind" onto the RunManifest field its config_id
# feeds. "verifier" has no field on RunManifest -- see selfrag.schema -- so
# a verifier stage is validated and reported but not merged into the
# manifest.
_STAGE_TO_MANIFEST_FIELD = {
    "parser": "parser_id",
    "chunker": "chunker_config_id",
    "embedder": "embedder_id",
    "sparse_retriever": "sparse_id",
    "fusion": "fusion_id",
    "reranker": "reranker_id",
    "query_transform": "query_transform_id",
    "generator": "generator_id",
}


def _fail(message: str) -> typer.Exit:
    error_console.print(message)
    return typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the installed selfrag version."""
    from selfrag import __version__

    console.print(f"selfrag {__version__}")


@app.command()
def doctor() -> None:
    """Report on the environment this process is running in.

    The project is RAM-constrained end to end (local embedding models,
    local ANN indexes, no server to offload to), so the two checks that
    actually cause a warning here are available RAM and free disk on the
    data volume -- everything else is informational.
    """
    table = Table(title="selfrag doctor", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")

    table.add_row("python", platform.python_version())
    table.add_row("platform", platform.platform())
    table.add_row("cpu count", str(os.cpu_count()))

    vm = psutil.virtual_memory()
    total_gb = vm.total / 1024**3
    available_gb = vm.available / 1024**3
    table.add_row("RAM total", f"{total_gb:.2f} GB")
    table.add_row("RAM available", f"{available_gb:.2f} GB")

    data_dir = paths.data_dir()
    disk = psutil.disk_usage(str(data_dir))
    free_gb = disk.free / 1024**3
    free_pct = 100.0 - disk.percent
    table.add_row("data dir", str(data_dir))
    table.add_row("disk free (data dir volume)", f"{free_gb:.2f} GB ({free_pct:.1f}%)")

    ledger_exists = paths.ledger_path().exists()
    table.add_row("ledger", str(paths.ledger_path()) + (" (exists)" if ledger_exists else " (not yet created)"))

    ort_threads = os.environ.get("SELFRAG_ORT_THREADS")
    table.add_row("SELFRAG_ORT_THREADS", ort_threads if ort_threads else "(not set)")

    for extra, module_name in _OPTIONAL_EXTRAS.items():
        installed = importlib.util.find_spec(module_name) is not None
        table.add_row(f"extra: {extra}", "installed" if installed else "not installed")

    console.print(table)

    warnings: list[str] = []
    if available_gb < _MIN_AVAILABLE_RAM_GB:
        warnings.append(
            f"available RAM is {available_gb:.2f} GB, below the {_MIN_AVAILABLE_RAM_GB:.0f} GB "
            "this project assumes is free for local embedding/index workloads."
        )
    if free_pct < _MIN_FREE_DISK_PCT:
        warnings.append(
            f"the data dir's volume has only {free_pct:.1f}% free space "
            f"({free_gb:.2f} GB) -- indexes and caches may not fit."
        )

    if warnings:
        console.print()
        for w in warnings:
            console.print(f"[yellow]warning:[/yellow] {w}")


@registry_app.command("list")
def registry_list(
    kind: Annotated[str | None, typer.Option(help="restrict to one component kind")] = None,
) -> None:
    """List registered components and their config schemas."""
    if kind is not None and kind not in registry.list_kinds():
        raise _fail(
            f"unknown component kind {kind!r}; valid kinds with registered components: "
            f"{registry.list_kinds()}"
        )

    kinds = [kind] if kind is not None else registry.list_kinds()
    if not kinds:
        console.print("[yellow]no components registered[/yellow]")
        return

    for k in kinds:
        table = Table(title=f"kind: {k}")
        table.add_column("name", style="bold")
        table.add_column("config fields")
        for name in registry.list_components(k):
            cls = registry.get(k, name)
            fields = ", ".join(cls.config_model.model_fields)
            table.add_row(name, fields or "(none)")
        console.print(table)


@ledger_app.command("init")
def ledger_init() -> None:
    """Create (or open) the experiment ledger, applying schema migrations."""
    ledger_file = paths.ledger_path()
    already_existed = ledger_file.exists()
    with Ledger(ledger_file):
        pass
    verb = "opened existing" if already_existed else "created new"
    console.print(f"[green]{verb} ledger[/green] at {ledger_file}")


@ledger_app.command("runs")
def ledger_runs(
    limit: Annotated[int, typer.Option(help="maximum number of runs to show")] = 20,
) -> None:
    """List the most recent runs."""
    ledger_file = paths.ledger_path()
    if not ledger_file.exists():
        raise _fail(f"no ledger at {ledger_file}; run `selfrag ledger init` first")

    with Ledger(ledger_file) as ledger:
        records = ledger.list_runs(limit=limit)

    if not records:
        console.print("[yellow]no runs recorded yet[/yellow]")
        return

    table = Table(title=f"last {len(records)} run(s)")
    for col in ("run_id", "status", "started_at", "finished_at", "p95_ms", "cost_usd"):
        table.add_column(col)
    for r in records:
        table.add_row(
            r.run_id,
            r.status,
            str(r.started_at) if r.started_at else "",
            str(r.finished_at) if r.finished_at else "",
            f"{r.p95_ms:.1f}" if r.p95_ms is not None else "",
            f"{r.cost_usd:.5f}" if r.cost_usd is not None else "",
        )
    console.print(table)


@ledger_app.command("show")
def ledger_show(run_id: str) -> None:
    """Show full detail for one run."""
    ledger_file = paths.ledger_path()
    if not ledger_file.exists():
        raise _fail(f"no ledger at {ledger_file}; run `selfrag ledger init` first")

    with Ledger(ledger_file) as ledger:
        try:
            record = ledger.get_run(run_id)
        except KeyError as exc:
            raise _fail(str(exc)) from exc
        violations = ledger.constraint_violations(run_id)

    table = Table(title=f"run {run_id}", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("status", record.status)
    table.add_row("git sha", record.git_sha or "(none)")
    table.add_row("started_at", str(record.started_at))
    table.add_row("finished_at", str(record.finished_at))
    table.add_row("p50_ms", str(record.p50_ms))
    table.add_row("p95_ms", str(record.p95_ms))
    table.add_row("peak_rss_mb", str(record.peak_rss_mb))
    table.add_row("cost_usd", str(record.cost_usd))
    table.add_row("notes", record.notes or "(none)")
    table.add_row("metrics", str(record.metrics))
    console.print(table)

    if violations:
        console.print("[yellow]constraint violations:[/yellow]")
        for v in violations:
            console.print(f"  - {v.constraint}: actual={v.actual} > limit={v.limit}")
    else:
        console.print("[green]no constraint violations[/green]")


@config_app.command("validate")
def config_validate(
    path: Annotated[Path, typer.Argument(help="pipeline YAML file")],
) -> None:
    """Validate a pipeline YAML against the registry and print the resolved run_id.

    Each entry under ``stages:`` is built through the registry (so an
    unknown component name or an invalid config is caught here, not at run
    time); every other top-level key is passed straight through as a
    ``RunManifest`` field. See ``configs/baseline.yaml`` for the shape this
    expects.
    """
    if not path.is_file():
        raise _fail(f"no such file: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise _fail(f"failed to parse YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise _fail("top-level YAML document must be a mapping")

    manifest_fields = dict(raw)
    stages = manifest_fields.pop("stages", {}) or {}
    if not isinstance(stages, dict):
        raise _fail("'stages' must be a mapping of kind -> {name, config}")

    table = Table(title=f"resolved stages for {path}")
    table.add_column("kind", style="bold")
    table.add_column("component")
    table.add_column("config_id")

    for kind, spec in stages.items():
        if not isinstance(spec, dict):
            raise _fail(f"stage {kind!r} must be a mapping with a 'name' key, got {spec!r}")
        try:
            component = registry.build_from_dict(kind, spec)
        except (KeyError, ValueError) as exc:
            raise _fail(f"stage {kind!r} is invalid: {exc}") from exc

        table.add_row(kind, spec.get("name", ""), component.config_id)

        field_name = _STAGE_TO_MANIFEST_FIELD.get(kind)
        if field_name:
            manifest_fields[field_name] = component.config_id

    console.print(table)

    try:
        manifest = RunManifest.model_validate(manifest_fields)
    except ValidationError as exc:
        raise _fail(f"resulting manifest is invalid: {exc}") from exc

    run_id = manifest.run_id()
    console.print(f"\n[bold green]run_id[/bold green]: {run_id}")


def _load_ingest_config(path: Path) -> pipeline.IngestConfig:
    if not path.is_file():
        raise _fail(f"no such file: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise _fail(f"failed to parse YAML: {exc}") from exc
    try:
        return pipeline.IngestConfig.model_validate(raw or {})
    except ValidationError as exc:
        raise _fail(f"invalid ingest config: {exc}") from exc


def _print_ingest_report(report: pipeline.IngestReport) -> None:
    table = Table(title="ingest run" + (" (dry run)" if report.dry_run else ""), show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("ingest_run_id", report.ingest_run_id or "(none -- dry run)")
    table.add_row("corpus_snapshot", report.corpus_snapshot)
    table.add_row("documents seen", str(report.n_seen))
    table.add_row("documents acquired", str(report.n_acquired))
    table.add_row("documents parsed", str(report.n_parsed))
    table.add_row("documents skipped (already done)", str(report.n_skipped))
    table.add_row("documents failed", str(report.n_failed))
    table.add_row("chunks created", str(report.n_chunks_created))
    table.add_row("chunks marked duplicate", str(report.n_chunks_duplicate))
    table.add_row("elapsed", f"{report.elapsed_seconds:.2f}s")
    console.print(table)

    if report.parse_quality:
        quality_table = Table(title="aggregate parse quality")
        quality_table.add_column("metric", style="bold")
        quality_table.add_column("mean")
        for k, v in report.parse_quality.items():
            quality_table.add_row(k, f"{v:.4f}")
        console.print(quality_table)

    if report.failures:
        console.print("\n[yellow]failures:[/yellow]")
        for f in report.failures:
            console.print(f"  - {f.doc_id} [{f.stage}]: {f.reason}")


@ingest_app.command("run")
def ingest_run(
    config: Annotated[Path, typer.Option("--config", help="ingest pipeline YAML")],
    limit: Annotated[int | None, typer.Option(help="process at most N documents")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="report what would happen; write nothing")
    ] = False,
) -> None:
    """Run the full ingest pipeline: acquire, parse, chunk, dedup, persist."""
    ingest_config = _load_ingest_config(config)
    manifest_path = pipeline.manifest_path_for(ingest_config.corpus_name)

    if dry_run:
        report = pipeline.run_ingest(ingest_config, manifest_path=manifest_path, limit=limit, dry_run=True)
        _print_ingest_report(report)
        return

    ledger_file = paths.ledger_path()
    with ArxivClient() as client, Ledger(ledger_file) as ledger:
        source = pipeline.ArxivDocumentSource(client, max_extracted_bytes=ingest_config.max_extracted_bytes)
        report = pipeline.run_ingest(
            ingest_config, source=source, ledger=ledger, manifest_path=manifest_path, limit=limit
        )

    _print_ingest_report(report)
    if report.n_failed:
        raise typer.Exit(code=1)


@ingest_app.command("status")
def ingest_status(
    corpus_name: Annotated[str, typer.Option(help="corpus name, matching an ingest config's corpus_name")] = "dev",
) -> None:
    """Manifest counts by status, chunk/document totals, and the corpus snapshot id."""
    manifest_path = pipeline.manifest_path_for(corpus_name)
    manifest = Manifest(manifest_path)

    counts: dict[str, int] = {}
    for entry in manifest:
        counts[entry.status.value] = counts.get(entry.status.value, 0) + 1

    table = Table(title=f"ingest status: {corpus_name}", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("manifest path", str(manifest_path))
    table.add_row("total documents (manifest)", str(len(manifest)))
    for status_name in ("pending", "acquired", "parsed", "failed", "tombstoned"):
        table.add_row(f"  {status_name}", str(counts.get(status_name, 0)))
    table.add_row("corpus_snapshot", manifest.snapshot_id())

    ledger_file = paths.ledger_path()
    if ledger_file.exists():
        with Ledger(ledger_file) as ledger:
            table.add_row("documents (ledger, live)", str(len(ledger.live_documents())))
            table.add_row("chunks (ledger)", str(ledger.count_chunks()))
    else:
        table.add_row("ledger", "(not yet created)")

    console.print(table)


@ingest_app.command("quality")
def ingest_quality(
    parser: Annotated[str | None, typer.Option(help="restrict to one parser: latex|pdf")] = None,
) -> None:
    """Aggregate, deterministic parse-quality report -- how LaTeX-vs-PDF gets decided from data."""
    if parser is not None and parser not in _PARSER_ID_ALIASES:
        raise _fail(f"unknown --parser {parser!r}; valid choices: {sorted(_PARSER_ID_ALIASES)}")

    ledger_file = paths.ledger_path()
    if not ledger_file.exists():
        raise _fail(f"no ledger at {ledger_file}; run `selfrag ingest run` first")

    parser_id = _PARSER_ID_ALIASES.get(parser) if parser else None
    with Ledger(ledger_file) as ledger:
        report = pipeline.compute_quality_report(ledger, parser_id=parser_id)

    if report is None:
        console.print("[yellow]no documents to report on[/yellow]")
        return

    table = Table(title=f"parse quality{f' ({parser})' if parser else ''}")
    table.add_column("metric", style="bold")
    table.add_column("mean")
    for k, v in report.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
