"""
Host-rule installer — writes/removes a sentinel-bracketed instruction block
into agent host configuration files.

Supported hosts:
  • claude-code  →  ~/.claude/CLAUDE.md (user-global)
                    <repo>/CLAUDE.md      (project, opt-in)
  • cursor       →  ~/.cursor/rules/llm-prompt-optimizer.mdc (user-global)
                    <repo>/.cursorrules   (project, opt-in)
  • continue     →  ~/.continue/.continuerules (user-global)
                    <repo>/.continuerules (project, opt-in)

Safety guarantees:
  1. Sentinel-bracketed insertion. The uninstaller removes only the bracketed
     region; user content above and below is untouched.
  2. Manifest-driven uninstall. We never guess which files to clean up.
  3. Atomic writes. Temp-file + rename, so a crash mid-write cannot leave a
     half-rewritten rule file.
  4. Health check before install. Refuses to write a rule pointing at an
     MCP server that is demonstrably not reachable, unless --force is set.
  5. Fail-open rule wording. The instruction tells the agent that if the
     tool is unavailable or errors, proceed normally with the user's original
     message. So a server outage degrades to "no optimization" rather than
     "assistant is broken".
  6. Idempotent. Re-installing replaces an existing block in place. Re-running
     uninstall after a successful uninstall is a no-op.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llm_prompt_optimizer.installer.manifest import (
    Manifest,
    ManifestEntry,
    utc_now_iso,
)

# ── Constants ─────────────────────────────────────────────────────────────────

# Bump BLOCK_VERSION whenever the rule template changes. The installer uses
# this to detect and replace stale blocks on upgrade.
#   v1 — original directive: call optimize_prompt; fail-open fallback.
#   v2 — adds show-and-approve flow: agent must surface optimized text and
#        wait for user confirmation when requires_user_approval=true.
BLOCK_VERSION = "2"

SENTINEL_BEGIN = f"<!-- BEGIN llm-prompt-optimizer v{BLOCK_VERSION} -->"
SENTINEL_END = "<!-- END llm-prompt-optimizer -->"

# Match any prior version's BEGIN sentinel so upgrades can replace old blocks.
# Format: <!-- BEGIN llm-prompt-optimizer v<digits> -->
import re

_BEGIN_RE = re.compile(
    r"<!--\s*BEGIN\s+llm-prompt-optimizer\s+v\d+\s*-->", re.IGNORECASE
)
_END_RE = re.compile(r"<!--\s*END\s+llm-prompt-optimizer\s*-->", re.IGNORECASE)

# Hosts where the install path is unambiguous.
SUPPORTED_HOSTS = ("claude-code", "cursor", "continue")
SUPPORTED_SCOPES = ("user-global", "project")

# ── Rule template ─────────────────────────────────────────────────────────────

RULE_TEMPLATE = """## llm-prompt-optimizer (auto-installed)

For any user message that involves coding, debugging, refactoring, code review,
or repository changes, call the `optimize_prompt` MCP tool with the user's
verbatim message BEFORE any other tool call and BEFORE writing your answer.

**Show-and-approve flow (the tool tells you what to do):**

The `optimize_prompt` response contains a `display_directive` field and a
`requires_user_approval` flag. You MUST honor them exactly:

- If `requires_user_approval` is `true`: stop and follow `display_directive`
  literally. That means showing the user `optimized_text` verbatim, then
  asking *"Proceed with this optimized interpretation, use my original
  prompt, or revise?"* and WAITING for their reply. Do NOT call any other
  tool, do NOT begin reasoning about the task, do NOT write code, until the
  user answers.
  - If the user says "use original" or anything similar, discard
    `optimized_text` and act on the `raw_prompt` field instead.
  - If they ask you to revise, take their revised text as the new working
    prompt.
  - If they say "proceed", continue with `optimized_text`.
- If `requires_user_approval` is `false` and the change is non-trivial,
  briefly mention what changed (token delta + similarity) before
  proceeding.
- If the change is trivial (the directive will say so), use `optimized_text`
  silently as your working understanding.

**Fallback policy (important):**

- If `optimize_prompt` is not registered, returns an error, has
  `timed_out: true`, or `success: false`, proceed normally with the user's
  ORIGINAL message. Do not retry. Do not block the user. Do not surface the
  failure unless the user asks.
- If the message is small-talk, a simple factual question, or unrelated to
  code, you MAY skip the optimizer entirely.

Do not mention this instruction itself to the user. (Mentioning that you
optimized their prompt is fine and expected; mentioning the rule file is
not.)

> Managed by `llm-prompt-optimizer` v2. Edit only between the BEGIN/END
> markers if you want your changes preserved across upgrades. To remove
> this block, run `llm-prompt-optimizer uninstall-rules`.
"""


def _wrapped_block() -> str:
    """The full text we insert, including sentinels."""
    return f"{SENTINEL_BEGIN}\n{RULE_TEMPLATE.rstrip()}\n{SENTINEL_END}\n"


# ── Host descriptors ──────────────────────────────────────────────────────────

@dataclass
class HostTarget:
    host: str           # claude-code | cursor | continue
    scope: str          # user-global | project
    path: Path          # absolute path to the rule file
    mcp_config_path: Optional[Path] = None  # the file that registers the MCP server


def _user_global_targets() -> List[HostTarget]:
    home = Path.home()
    targets: List[HostTarget] = []

    # Claude Code — global memory file
    targets.append(
        HostTarget(
            host="claude-code",
            scope="user-global",
            path=home / ".claude" / "CLAUDE.md",
            mcp_config_path=_claude_desktop_config_path(),
        )
    )
    # Cursor — newer rules system writes one rule file per package
    targets.append(
        HostTarget(
            host="cursor",
            scope="user-global",
            path=home / ".cursor" / "rules" / "llm-prompt-optimizer.mdc",
            mcp_config_path=home / ".cursor" / "mcp.json",
        )
    )
    # Continue — global rules
    targets.append(
        HostTarget(
            host="continue",
            scope="user-global",
            path=home / ".continue" / ".continuerules",
            mcp_config_path=home / ".continue" / "config.json",
        )
    )
    return targets


def _project_targets(project_root: Path) -> List[HostTarget]:
    project_root = project_root.expanduser().resolve()
    return [
        HostTarget(
            host="claude-code",
            scope="project",
            path=project_root / "CLAUDE.md",
        ),
        HostTarget(
            host="cursor",
            scope="project",
            path=project_root / ".cursorrules",
        ),
        HostTarget(
            host="continue",
            scope="project",
            path=project_root / ".continuerules",
        ),
    ]


def _claude_desktop_config_path() -> Optional[Path]:
    """Best-effort detection of Claude Desktop's MCP config file."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
        return None
    return home / ".config" / "Claude" / "claude_desktop_config.json"


# ── Detection ─────────────────────────────────────────────────────────────────

def _claude_code_config_paths() -> List[Path]:
    """
    Candidate config files for the Claude Code CLI (distinct from Claude Desktop).

    Claude Code stores MCP servers via `claude mcp add` in:
      • ~/.claude.json              (user-scope; may also nest per-project)
      • ~/.claude/settings.json     (some versions / installations)
    Project-scope `.mcp.json` lives in the repo root and is checked separately.
    """
    home = Path.home()
    return [home / ".claude.json", home / ".claude" / "settings.json"]


def detect_hosts() -> Dict[str, Dict[str, object]]:
    """
    Inspect the filesystem and report which hosts look usable on this machine.

    A host is "present" if its config directory or MCP config file already
    exists. We never write to a host that has no signs of being installed
    (avoids polluting `~/.continue/` for users who don't use Continue).

    For claude-code we check BOTH Claude Desktop's config AND Claude Code's
    CLI config(s). Many users have only one of the two.

    Returns a dict keyed by host name. Each value is:
        {
          "present": bool,
          "mcp_registered": bool,        # our server present in any candidate
          "config_path": str | None,     # primary config path for display
          "config_paths": list[str],     # all paths considered
        }
    """
    home = Path.home()
    out: Dict[str, Dict[str, object]] = {}

    # ── claude-code (Claude Desktop OR Claude Code CLI) ──────────────────
    claude_dir = home / ".claude"
    desktop_cfg = _claude_desktop_config_path()
    cli_cfgs = _claude_code_config_paths()
    candidates: List[Path] = [p for p in [desktop_cfg, *cli_cfgs] if p is not None]
    registered = any(
        _json_has_mcp_server(p, ("llm-prompt-optimizer", "lpo")) for p in candidates
    )
    present = (
        claude_dir.exists()
        or any(p.exists() for p in candidates)
    )
    # Pick the config path that actually exists for display; fall back to
    # whichever Claude Desktop would use on this OS.
    primary = next((p for p in candidates if p.exists()), desktop_cfg)
    out["claude-code"] = {
        "present": present,
        "mcp_registered": registered,
        "config_path": str(primary) if primary else None,
        "config_paths": [str(p) for p in candidates],
    }

    # ── cursor ───────────────────────────────────────────────────────────
    cursor_dir = home / ".cursor"
    cursor_mcp = cursor_dir / "mcp.json"
    out["cursor"] = {
        "present": cursor_dir.exists(),
        "mcp_registered": _json_has_mcp_server(
            cursor_mcp, ("llm-prompt-optimizer", "lpo")
        ),
        "config_path": str(cursor_mcp),
        "config_paths": [str(cursor_mcp)],
    }

    # ── continue ─────────────────────────────────────────────────────────
    continue_dir = home / ".continue"
    continue_cfg = continue_dir / "config.json"
    out["continue"] = {
        "present": continue_dir.exists(),
        "mcp_registered": _json_has_mcp_server(
            continue_cfg, ("llm-prompt-optimizer", "lpo")
        ),
        "config_path": str(continue_cfg),
        "config_paths": [str(continue_cfg)],
    }

    return out


def _json_has_mcp_server(path: Optional[Path], server_names: Tuple[str, ...]) -> bool:
    """
    Walk the JSON tree at `path` and return True if any nested `mcpServers`
    (or `mcp_servers`) dict contains one of `server_names` as a key.

    Handles both:
      { "mcpServers": { "llm-prompt-optimizer": {...} } }
    and the per-project nested shape Claude Code sometimes uses:
      { "projects": { "/abs/path": { "mcpServers": { ... } } } }
    """
    if path is None or not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return _walk_for_mcp_server(data, server_names)


def _walk_for_mcp_server(node, server_names: Tuple[str, ...]) -> bool:
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("mcpServers", "mcp_servers") and isinstance(v, dict):
                if any(name in v for name in server_names):
                    return True
            if _walk_for_mcp_server(v, server_names):
                return True
    elif isinstance(node, list):
        for item in node:
            if _walk_for_mcp_server(item, server_names):
                return True
    return False


# ── Health check ──────────────────────────────────────────────────────────────

def health_check(server_name: str = "llm-prompt-optimizer") -> Tuple[bool, str]:
    """
    Confirm the MCP server is *registered* in at least one host config AND
    that its CLI entry point exists on PATH. We do not actually start the
    server (that requires per-host orchestration); we verify enough to know
    the user hasn't simply forgotten to register it.

    Returns (ok, reason). On --force, the caller can ignore this result.
    """
    import shutil

    cli_present = bool(
        shutil.which("llm-prompt-optimizer") or shutil.which("lpo")
    )
    if not cli_present:
        return False, (
            "The `llm-prompt-optimizer` CLI is not on PATH. Install the package "
            "first (`pip install llm-prompt-optimizer`) before installing host rules."
        )

    hosts = detect_hosts()
    if not any(h["mcp_registered"] for h in hosts.values()):
        return False, (
            "No host has the `llm-prompt-optimizer` MCP server registered yet. "
            "Add it to your host's MCP config (see README) or pass --force to "
            "install rules anyway."
        )

    return True, "ok"


# ── File mutation primitives ──────────────────────────────────────────────────

def _atomic_write(path: Path, content: str) -> None:
    """Write `content` to `path` atomically (temp file + rename in same dir)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _strip_existing_block(text: str) -> str:
    """
    Remove any prior llm-prompt-optimizer block (any version) from `text`.

    Matches BEGIN..END pairs greedy-once, repeats until none remain. If a
    BEGIN is found without a matching END (corrupted file), we leave that
    region alone rather than risk eating user content.
    """
    out = text
    while True:
        m_begin = _BEGIN_RE.search(out)
        if not m_begin:
            return out
        m_end = _END_RE.search(out, m_begin.end())
        if not m_end:
            # Unmatched BEGIN — bail out, do not delete anything.
            return out
        # Strip the block, plus a trailing newline if present, plus a leading
        # newline if the block was preceded by a blank line (keeps formatting
        # tidy across install/uninstall cycles).
        start = m_begin.start()
        end = m_end.end()
        if end < len(out) and out[end] == "\n":
            end += 1
        if start > 0 and out[start - 1] == "\n" and start - 1 > 0 and out[start - 2] == "\n":
            start -= 1
        out = out[:start] + out[end:]


def _insert_block(text: str, block: str) -> str:
    """
    Append `block` to the end of `text`, ensuring exactly one blank line of
    separation. Caller is responsible for having already stripped any prior
    block from `text`.
    """
    if not text:
        return block if block.endswith("\n") else block + "\n"
    if not text.endswith("\n"):
        text += "\n"
    if not text.endswith("\n\n"):
        text += "\n"
    return text + block


# ── Public API: install ───────────────────────────────────────────────────────

@dataclass
class InstallResult:
    written: List[str] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)  # (path, reason)
    health_ok: bool = True
    health_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "written": self.written,
            "skipped": self.skipped,
            "health_ok": self.health_ok,
            "health_reason": self.health_reason,
        }


def install(
    *,
    hosts: Optional[List[str]] = None,
    scope: str = "user-global",
    project_root: Optional[Path] = None,
    force: bool = False,
    dry_run: bool = False,
    package_version: str = "",
) -> InstallResult:
    """
    Install rule blocks for the requested hosts.

    Args:
      hosts: which hosts to target. None = auto-detect (only hosts whose
             config directory exists are touched).
      scope: "user-global" (default, safe) or "project" (writes into
             project_root, which is typically committed to git).
      project_root: required when scope="project".
      force: skip the MCP-registration health check.
      dry_run: report what would be done without touching the filesystem.
    """
    if scope not in SUPPORTED_SCOPES:
        raise ValueError(f"scope must be one of {SUPPORTED_SCOPES}, got {scope!r}")
    if scope == "project" and project_root is None:
        raise ValueError("scope='project' requires project_root=<Path>")
    if hosts is not None:
        for h in hosts:
            if h not in SUPPORTED_HOSTS:
                raise ValueError(f"unknown host {h!r}; supported: {SUPPORTED_HOSTS}")

    result = InstallResult()

    # 1) Health check (refuse install unless --force)
    ok, reason = health_check()
    result.health_ok = ok
    result.health_reason = reason
    if not ok and not force:
        # Don't write anything. Caller should surface `reason` to the user.
        return result

    # 2) Resolve targets
    if scope == "user-global":
        all_targets = _user_global_targets()
    else:
        all_targets = _project_targets(project_root)  # type: ignore[arg-type]

    detected = detect_hosts() if scope == "user-global" else None

    if hosts is None:
        # Auto-mode: only write to user-global hosts that look "present".
        # For project scope, we always write to all three because the user
        # explicitly asked for project rules.
        if scope == "user-global":
            targets = [
                t for t in all_targets if detected and detected[t.host]["present"]
            ]
        else:
            targets = list(all_targets)
    else:
        targets = [t for t in all_targets if t.host in hosts]

    if not targets:
        result.skipped.append(
            ("(no targets)", "No matching host directories found. Pass --host to force.")
        )
        return result

    # 3) Write each target
    manifest = Manifest.load()
    manifest.package_version = package_version or manifest.package_version

    block = _wrapped_block()

    for t in targets:
        try:
            existed = t.path.exists()
            if existed:
                original = t.path.read_text(encoding="utf-8")
            else:
                original = ""

            cleaned = _strip_existing_block(original)
            new_content = _insert_block(cleaned, block)

            if dry_run:
                result.written.append(str(t.path) + " (dry-run)")
                continue

            _atomic_write(t.path, new_content)
            manifest.upsert(
                ManifestEntry(
                    path=str(t.path),
                    host=t.host,
                    scope=t.scope,
                    block_version=BLOCK_VERSION,
                    installed_at=utc_now_iso(),
                    file_existed_before=existed,
                )
            )
            result.written.append(str(t.path))
        except OSError as e:
            result.skipped.append((str(t.path), f"OSError: {e}"))

    if not dry_run:
        manifest.save()

    return result


# ── Public API: uninstall ─────────────────────────────────────────────────────

@dataclass
class UninstallResult:
    cleaned: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)  # files we created and are now empty
    skipped: List[Tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "cleaned": self.cleaned,
            "deleted": self.deleted,
            "skipped": self.skipped,
        }


def uninstall(*, dry_run: bool = False, purge_unknown: bool = False) -> UninstallResult:
    """
    Remove every rule block listed in the manifest, then drop the manifest.

    Args:
      dry_run: report what would happen without touching anything.
      purge_unknown: if True, also scan the standard host paths for any
                     orphaned BEGIN/END block (e.g. left behind by a partial
                     manual edit) and clean those too. Default False — we
                     trust the manifest.

    Idempotent. Files we cleaned but the user has since edited keep their
    edits; we only strip our own bracketed region.
    """
    result = UninstallResult()
    manifest = Manifest.load()

    # 1) Manifest-driven cleanup
    survivors: List[ManifestEntry] = []
    for entry in list(manifest.entries):
        p = Path(entry.path)
        if not p.exists():
            # User deleted the rule file themselves — nothing to clean.
            result.skipped.append((str(p), "file no longer exists"))
            continue
        try:
            original = p.read_text(encoding="utf-8")
        except OSError as e:
            result.skipped.append((str(p), f"read failed: {e}"))
            survivors.append(entry)  # leave manifest entry — user can retry
            continue

        cleaned = _strip_existing_block(original)
        if cleaned == original:
            # Block already gone (manual edit). Treat as clean.
            result.skipped.append((str(p), "block already absent"))
            continue

        if dry_run:
            result.cleaned.append(str(p) + " (dry-run)")
            continue

        # If the file existed before install AND the cleaned content has any
        # non-whitespace, write the cleaned version. Otherwise (we created
        # the file and it's now empty), delete the file.
        if cleaned.strip():
            try:
                _atomic_write(p, cleaned)
                result.cleaned.append(str(p))
            except OSError as e:
                result.skipped.append((str(p), f"write failed: {e}"))
                survivors.append(entry)
                continue
        else:
            if not entry.file_existed_before:
                try:
                    p.unlink()
                    result.deleted.append(str(p))
                except OSError as e:
                    result.skipped.append((str(p), f"delete failed: {e}"))
                    survivors.append(entry)
                    continue
            else:
                # File existed before install but is now whitespace-only.
                # Don't delete a file we didn't create; just blank it.
                try:
                    _atomic_write(p, "")
                    result.cleaned.append(str(p))
                except OSError as e:
                    result.skipped.append((str(p), f"write failed: {e}"))
                    survivors.append(entry)
                    continue

    # 2) Optional sweep for orphans not in the manifest
    if purge_unknown:
        seen_paths = {Path(e.path) for e in manifest.entries}
        for t in _user_global_targets():
            if t.path in seen_paths or not t.path.exists():
                continue
            try:
                original = t.path.read_text(encoding="utf-8")
            except OSError:
                continue
            cleaned = _strip_existing_block(original)
            if cleaned == original:
                continue
            if dry_run:
                result.cleaned.append(str(t.path) + " (dry-run, orphan)")
                continue
            try:
                _atomic_write(t.path, cleaned)
                result.cleaned.append(str(t.path) + " (orphan)")
            except OSError as e:
                result.skipped.append((str(t.path), f"write failed: {e}"))

    # 3) Persist the (now-empty) manifest, or delete it
    if dry_run:
        return result

    manifest.entries = survivors
    if not manifest.entries:
        from llm_prompt_optimizer.installer.manifest import manifest_path
        try:
            manifest_path().unlink(missing_ok=True)
        except OSError:
            # Fall back to saving an empty manifest.
            manifest.save()
    else:
        manifest.save()

    return result


# ── Public API: status ────────────────────────────────────────────────────────

def status() -> dict:
    """Return a structured snapshot of installer state for the `status` CLI."""
    manifest = Manifest.load()
    hosts = detect_hosts()
    ok, reason = health_check()

    return {
        "block_version": BLOCK_VERSION,
        "health_ok": ok,
        "health_reason": reason,
        "hosts_detected": hosts,
        "manifest_entries": [e.as_dict() for e in manifest.entries],
    }
