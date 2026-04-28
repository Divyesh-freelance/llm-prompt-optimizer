"""
CLI for LLM Prompt Optimizer.

Usage:
    llm-prompt-optimizer optimize "debug EMA mismatch"
    llm-prompt-optimizer classify "debug EMA mismatch"
    llm-prompt-optimizer validate --raw "..." --optimized "..."
    llm-prompt-optimizer detect-drift --raw "..." --optimized "..."
    llm-prompt-optimizer estimate-cost "debug EMA mismatch"
    llm-prompt-optimizer benchmark
    llm-prompt-optimizer serve --transport stdio
    llm-prompt-optimizer serve --transport http --port 8765
    llm-prompt-optimizer install-rules
    llm-prompt-optimizer uninstall-rules
    llm-prompt-optimizer rules-status
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer

from llm_prompt_optimizer import Optimizer, OptimizerConfig
from llm_prompt_optimizer.models.prompt import RawPrompt

app = typer.Typer(
    name="llm-prompt-optimizer",
    help="Deterministic prompt optimization middleware for AI coding agents.",
    add_completion=False,
)

_optimizer: Optional[Optimizer] = None


def get_optimizer() -> Optimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = Optimizer(config=OptimizerConfig.from_env())
    return _optimizer


@app.command("optimize")
def optimize(
    prompt: str = typer.Argument(..., help="The raw prompt to optimize."),
    strict: bool = typer.Option(False, "--strict", help="Enable strict scope mode."),
    repo_root: Optional[str] = typer.Option(None, "--repo-root", help="Repository root path."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
):
    """Optimize a prompt and print the result."""
    opt = get_optimizer()
    result = opt.optimize(prompt=prompt, strict_mode=strict, repo_root=repo_root)
    if json_output:
        out = {
            "optimized_text": result.optimized_prompt.text,
            "token_estimate": result.optimized_prompt.token_estimate,
            "semantic_similarity": result.optimized_prompt.semantic_similarity,
            "compression_ratio": result.optimized_prompt.compression_ratio,
            "success": result.success,
            "errors": result.errors,
            "warnings": result.warnings,
        }
        typer.echo(json.dumps(out, indent=2))
    else:
        typer.echo("\n" + "=" * 60)
        typer.echo("OPTIMIZED PROMPT")
        typer.echo("=" * 60)
        typer.echo(result.optimized_prompt.text)
        typer.echo("\n" + "-" * 60)
        typer.echo(f"Tokens: {result.optimized_prompt.token_estimate}")
        typer.echo(f"Similarity: {result.optimized_prompt.semantic_similarity:.3f}")
        typer.echo(f"Compression: {result.optimized_prompt.compression_ratio:.2f}x")
        typer.echo(f"Success: {result.success}")
        if result.errors:
            typer.echo(f"Errors: {result.errors}", err=True)


@app.command("classify")
def classify(
    prompt: str = typer.Argument(..., help="Prompt to classify."),
    json_output: bool = typer.Option(False, "--json"),
):
    """Classify a prompt into task categories."""
    opt = get_optimizer()
    cls = opt.classify(prompt)
    if json_output:
        typer.echo(json.dumps({
            "category": cls.primary_category.value,
            "confidence": cls.primary_confidence,
            "complexity": cls.complexity_score,
            "has_stacktrace": cls.has_stacktrace,
            "extracted_files": cls.extracted_file_paths,
        }, indent=2))
    else:
        typer.echo(f"Category:    {cls.primary_category.value}")
        typer.echo(f"Confidence:  {cls.primary_confidence:.3f}")
        typer.echo(f"Complexity:  {cls.complexity_score:.3f}")
        typer.echo(f"Stacktrace:  {cls.has_stacktrace}")
        typer.echo(f"Files:       {cls.extracted_file_paths}")
        typer.echo(f"Symbols:     {cls.extracted_symbols}")


@app.command("validate")
def validate(
    raw: str = typer.Option(..., "--raw", help="Original raw prompt."),
    optimized: str = typer.Option(..., "--optimized", help="Optimized prompt to validate."),
    json_output: bool = typer.Option(False, "--json"),
):
    """Validate semantic similarity between raw and optimized prompt."""
    opt = get_optimizer()
    result = opt.validate(raw, optimized)
    if json_output:
        typer.echo(json.dumps({
            "passed": result.passed,
            "similarity": result.semantic_similarity,
            "threshold": result.threshold,
            "failure_reason": result.failure_reason,
        }, indent=2))
    else:
        typer.echo(f"Passed:     {result.passed}")
        typer.echo(f"Similarity: {result.semantic_similarity:.4f}")
        typer.echo(f"Threshold:  {result.threshold}")
        if result.failure_reason:
            typer.echo(f"Reason:     {result.failure_reason}", err=True)
    raise typer.Exit(0 if result.passed else 1)


@app.command("detect-drift")
def detect_drift(
    raw: str = typer.Option(..., "--raw"),
    optimized: str = typer.Option(..., "--optimized"),
    json_output: bool = typer.Option(False, "--json"),
):
    """Detect semantic drift between raw and optimized prompts."""
    opt = get_optimizer()
    report = opt.detect_drift(raw, optimized)
    if json_output:
        typer.echo(json.dumps({
            "is_clean": report.is_clean,
            "severity": report.overall_severity,
            "blocked": report.blocked,
            "events": [
                {"type": d.drift_type, "severity": d.severity, "description": d.description}
                for d in report.drifts_detected
            ],
        }, indent=2))
    else:
        typer.echo(f"Clean:    {report.is_clean}")
        typer.echo(f"Severity: {report.overall_severity}")
        typer.echo(f"Blocked:  {report.blocked}")
        for d in report.drifts_detected:
            typer.echo(f"  [{d.severity.upper()}] {d.drift_type}: {d.description}")


@app.command("estimate-cost")
def estimate_cost(
    prompt: str = typer.Argument(...),
    json_output: bool = typer.Option(False, "--json"),
):
    """Estimate token cost of a prompt."""
    opt = get_optimizer()
    cost = opt.estimate_cost(prompt)
    if json_output:
        typer.echo(json.dumps(cost, indent=2))
    else:
        typer.echo(f"Estimated tokens: {cost['estimated_tokens']}")
        typer.echo(f"Characters:       {cost['approx_chars']}")


@app.command("benchmark")
def benchmark(
    json_output: bool = typer.Option(False, "--json"),
):
    """Run the built-in benchmark suite."""
    opt = get_optimizer()
    report = opt.benchmark()
    summary = report.summarize()
    if json_output:
        typer.echo(json.dumps(summary, indent=2))
    else:
        typer.echo(f"\nBenchmark Results ({summary['total']} cases)")
        typer.echo("-" * 40)
        typer.echo(f"Passed:        {summary['passed']}/{summary['total']}")
        typer.echo(f"Pass rate:     {summary['pass_rate']:.1%}")
        typer.echo(f"Avg score:     {summary['avg_overall_score']:.3f}")
        typer.echo(f"Avg similarity:{summary['avg_similarity']:.3f}")
        typer.echo(f"Avg reduction: {summary['avg_token_reduction_pct']:.1f}%")
        for r in report.results:
            status = "✓" if r.passed else "✗"
            typer.echo(f"  {status} {r.case_name:<30} score={r.overall_score:.3f}")


@app.command("serve")
def serve(
    transport: str = typer.Option("stdio", "--transport", help="stdio | http | websocket | ipc"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
):
    """Start the MCP server."""
    import asyncio
    from llm_prompt_optimizer.mcp_server.server import MCPServer

    server = MCPServer()

    if transport == "stdio":
        typer.echo("Starting MCP server (stdio)...", err=True)
        server.run_stdio_sync()
    elif transport == "http":
        typer.echo(f"Starting MCP HTTP server on {host}:{port}...", err=True)
        asyncio.run(server.run_http(host=host, port=port))
    elif transport == "websocket":
        typer.echo(f"Starting MCP WebSocket server on ws://{host}:{port}...", err=True)
        asyncio.run(server.run_websocket(host=host, port=port))
    elif transport == "ipc":
        typer.echo("Starting MCP IPC server...", err=True)
        from llm_prompt_optimizer.mcp_server.transport.ipc import run as ipc_run
        asyncio.run(ipc_run())
    else:
        typer.echo(f"Unknown transport: {transport}", err=True)
        raise typer.Exit(1)


# ── Host-rule install / uninstall ─────────────────────────────────────────────

@app.command("install-rules")
def install_rules(
    host: List[str] = typer.Option(
        [],
        "--host",
        help=(
            "Target host(s): claude-code, cursor, continue. "
            "Repeat to select multiple. Omit to auto-detect installed hosts."
        ),
    ),
    scope: str = typer.Option(
        "user-global",
        "--scope",
        help="user-global (default, safe) or project (writes into --project-root).",
    ),
    project_root: Optional[str] = typer.Option(
        None,
        "--project-root",
        help="Required when --scope=project. Rule files written here are typically committed to git.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Skip the MCP-registration health check. Use only if you know what you're doing.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be written without touching the filesystem.",
    ),
    json_output: bool = typer.Option(False, "--json"),
):
    """
    Install agent rule blocks so coding prompts auto-route through optimize_prompt.

    Each block is wrapped in BEGIN/END sentinels and includes fail-open
    fallback wording: if the MCP server is unavailable, the agent proceeds
    normally with the user's original message.
    """
    from llm_prompt_optimizer import __version__
    from llm_prompt_optimizer.installer import install as do_install

    pr = Path(project_root).expanduser().resolve() if project_root else None
    try:
        result = do_install(
            hosts=list(host) or None,
            scope=scope,
            project_root=pr,
            force=force,
            dry_run=dry_run,
            package_version=__version__,
        )
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if json_output:
        typer.echo(json.dumps(result.as_dict(), indent=2))
        raise typer.Exit(0 if result.health_ok or force else 1)

    if not result.health_ok and not force:
        typer.echo(f"Health check failed: {result.health_reason}", err=True)
        typer.echo("No files written. Re-run with --force to override.", err=True)
        raise typer.Exit(1)

    if result.written:
        typer.echo("Wrote rule block to:")
        for p in result.written:
            typer.echo(f"  {p}")
    if result.skipped:
        typer.echo("\nSkipped:")
        for p, reason in result.skipped:
            typer.echo(f"  {p} — {reason}")
    if not result.written and not result.skipped:
        typer.echo("No targets matched. Try --host claude-code (or cursor / continue).")
    if dry_run:
        typer.echo("\n(dry run — nothing was actually written)")
    else:
        typer.echo("\nTo undo: llm-prompt-optimizer uninstall-rules")


@app.command("uninstall-rules")
def uninstall_rules(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be removed without touching the filesystem.",
    ),
    purge_unknown: bool = typer.Option(
        False,
        "--purge-unknown",
        help=(
            "Also scan standard host paths for orphaned BEGIN/END blocks "
            "(e.g. left behind by a manual edit) and clean them too."
        ),
    ),
    json_output: bool = typer.Option(False, "--json"),
):
    """
    Remove rule blocks installed by `install-rules`.

    Surgical: only the BEGIN/END-bracketed region is removed; user content
    above and below is preserved. Idempotent: running this twice is a no-op.
    Files that the installer created (and that are now empty) are deleted.
    """
    from llm_prompt_optimizer.installer import uninstall as do_uninstall

    result = do_uninstall(dry_run=dry_run, purge_unknown=purge_unknown)

    if json_output:
        typer.echo(json.dumps(result.as_dict(), indent=2))
        raise typer.Exit(0)

    if result.cleaned:
        typer.echo("Cleaned (block removed, file kept):")
        for p in result.cleaned:
            typer.echo(f"  {p}")
    if result.deleted:
        typer.echo("\nDeleted (file was created by installer and is now empty):")
        for p in result.deleted:
            typer.echo(f"  {p}")
    if result.skipped:
        typer.echo("\nSkipped:")
        for p, reason in result.skipped:
            typer.echo(f"  {p} — {reason}")
    if not (result.cleaned or result.deleted or result.skipped):
        typer.echo("Nothing to do. (No manifest entries, and --purge-unknown was not set.)")
    if dry_run:
        typer.echo("\n(dry run — nothing was actually removed)")


@app.command("rules-status")
def rules_status(
    json_output: bool = typer.Option(False, "--json"),
):
    """Show installer health and which rule files are currently managed."""
    from llm_prompt_optimizer.installer import status as do_status

    snapshot = do_status()
    if json_output:
        typer.echo(json.dumps(snapshot, indent=2))
        return

    typer.echo(f"Block version: {snapshot['block_version']}")
    typer.echo(
        f"Health: {'OK' if snapshot['health_ok'] else 'NOT OK'}"
        f" — {snapshot['health_reason']}"
    )
    typer.echo("\nHosts detected on this machine:")
    for host_name, info in snapshot["hosts_detected"].items():
        marker = "✓" if info["present"] else "·"
        mcp = "registered" if info["mcp_registered"] else "not registered"
        typer.echo(f"  {marker} {host_name:<13} mcp: {mcp:<14} cfg: {info['config_path']}")
        # If multiple candidate paths were considered (e.g. Claude Desktop +
        # Claude Code CLI), list them so the user can see what was checked.
        extras = [p for p in info.get("config_paths", []) if p != info.get("config_path")]
        for p in extras:
            typer.echo(f"      also checked: {p}")

    entries = snapshot["manifest_entries"]
    typer.echo(f"\nManaged rule files ({len(entries)}):")
    if not entries:
        typer.echo("  (none — run `llm-prompt-optimizer install-rules`)")
    for e in entries:
        typer.echo(
            f"  [{e['scope']:<12}] {e['host']:<13} v{e['block_version']} "
            f"installed {e['installed_at']}  →  {e['path']}"
        )


def main():
    app()


if __name__ == "__main__":
    main()
