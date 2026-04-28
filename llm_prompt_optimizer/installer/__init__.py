"""
Host-rule installer for llm-prompt-optimizer.

Writes/removes a small, sentinel-bracketed instruction block into agent host
configuration files (Claude Code's CLAUDE.md, Cursor's .cursorrules, Continue's
config) so that the agent calls the `optimize_prompt` MCP tool by default.

Public API:
    install(...)      — write rule blocks, record manifest
    uninstall(...)    — remove rule blocks recorded in the manifest
    status(...)       — show what is installed and where
    detect_hosts(...) — return which hosts are currently usable on this machine

Design invariants:
  • All inserted text is wrapped in BEGIN/END sentinels that include a
    block-version. The uninstaller removes ONLY the bracketed region; user
    content above and below is preserved byte-for-byte.
  • Every install records a manifest entry. Uninstall is manifest-driven and
    idempotent (running it twice is a no-op).
  • The rule template includes explicit fail-open language so that if the
    MCP server is down, missing, or slow, the agent proceeds normally with
    the user's original message instead of looping or refusing.
  • install() refuses to write if the MCP server is not reachable, unless
    --force is passed. This prevents shipping a rule that points at a tool
    that demonstrably does not exist.
"""

from llm_prompt_optimizer.installer.host_rules import (
    BLOCK_VERSION,
    InstallResult,
    UninstallResult,
    detect_hosts,
    install,
    status,
    uninstall,
)

__all__ = [
    "BLOCK_VERSION",
    "InstallResult",
    "UninstallResult",
    "detect_hosts",
    "install",
    "status",
    "uninstall",
]
