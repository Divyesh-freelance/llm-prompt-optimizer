"""
Install manifest — records every file the installer touched, so that
uninstall is surgical, idempotent, and survives package upgrades.

File location: $LPO_INSTALL_HOME/install-manifest.json
  Default: ~/.config/llm-prompt-optimizer/install-manifest.json (Linux/macOS)
           %APPDATA%\\llm-prompt-optimizer\\install-manifest.json (Windows)

Schema:
{
  "version": "1",                    # manifest schema version
  "package_version": "0.1.0",        # llm-prompt-optimizer version at write time
  "entries": [
    {
      "path": "/abs/path/to/CLAUDE.md",
      "host": "claude-code",         # claude-code | cursor | continue
      "scope": "user-global",        # user-global | project
      "block_version": "1",          # version of the inserted block
      "installed_at": "2026-04-25T12:00:00Z",
      "file_existed_before": true    # if false, uninstall may delete the file
    }
  ]
}

The manifest is the single source of truth for uninstall. If a user manually
deletes a rule file we wrote to, the manifest entry for it is simply skipped
on uninstall (idempotent — never errors).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

MANIFEST_SCHEMA_VERSION = "1"


def _install_home() -> Path:
    """Resolve the directory where the manifest lives. Override with $LPO_INSTALL_HOME."""
    override = os.environ.get("LPO_INSTALL_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "llm-prompt-optimizer"
    # Linux / macOS — XDG convention
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else (Path.home() / ".config")
    return base / "llm-prompt-optimizer"


def manifest_path() -> Path:
    return _install_home() / "install-manifest.json"


@dataclass
class ManifestEntry:
    path: str
    host: str
    scope: str
    block_version: str
    installed_at: str
    file_existed_before: bool

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Manifest:
    version: str = MANIFEST_SCHEMA_VERSION
    package_version: str = ""
    entries: List[ManifestEntry] = field(default_factory=list)

    # ── load / save ────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "Manifest":
        p = manifest_path()
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Corrupt manifest — treat as empty rather than crash. We never
            # want a broken manifest to block uninstall.
            return cls()
        entries = [ManifestEntry(**e) for e in data.get("entries", [])]
        return cls(
            version=data.get("version", MANIFEST_SCHEMA_VERSION),
            package_version=data.get("package_version", ""),
            entries=entries,
        )

    def save(self) -> None:
        p = manifest_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "package_version": self.package_version,
            "entries": [e.as_dict() for e in self.entries],
        }
        # Atomic write: temp file + rename. Prevents a partially-written
        # manifest if the process is killed mid-write.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(p.parent),
            prefix=".manifest-",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(payload, tmp, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, p)

    # ── mutators ───────────────────────────────────────────────────────────

    def upsert(self, entry: ManifestEntry) -> None:
        """
        Add or replace the entry for entry.path. Path is the unique key.

        On replace, we preserve `file_existed_before` from the prior entry —
        this matters for re-install/upgrade: the file exists when we
        re-install only because we created it the first time. Forgetting
        that would cause uninstall to leave behind an installer-created file.
        """
        canonical = str(Path(entry.path).expanduser().resolve(strict=False))
        entry.path = canonical
        prior = next((e for e in self.entries if e.path == canonical), None)
        if prior is not None:
            entry.file_existed_before = prior.file_existed_before
        self.entries = [e for e in self.entries if e.path != canonical]
        self.entries.append(entry)

    def remove(self, path: str) -> Optional[ManifestEntry]:
        canonical = str(Path(path).expanduser().resolve(strict=False))
        for i, e in enumerate(self.entries):
            if e.path == canonical:
                return self.entries.pop(i)
        return None

    def find(self, path: str) -> Optional[ManifestEntry]:
        canonical = str(Path(path).expanduser().resolve(strict=False))
        for e in self.entries:
            if e.path == canonical:
                return e
        return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
