"""
FallbackGraphEngine — Standalone dependency discovery without external graph tools.

Implements a full dependency graph using:
  - Python AST import chain analysis
  - Symbol reference tracking
  - Call graph construction
  - Execution path heuristics
  - Git-aware context (when .git available)
  - Folder heuristics

Priority order (as per spec):
  1. Optional Graph MCP (handled by adapters, not here)
  2. Local AST Graph ← this module
  3. Import Resolver ← this module
  4. Symbol Resolver ← this module
  5. Execution Path Discovery ← this module
  6. Git Context ← this module
  7. Folder Heuristic ← this module
  8. Strict User Scope ← this module
"""

from __future__ import annotations

import ast
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from llm_prompt_optimizer.models.context import DependencyNode, ContextSpan
from llm_prompt_optimizer.models.intent import IntentLock
from llm_prompt_optimizer.utils.helpers import (
    get_logger,
    safe_read_file,
    parse_ast_safe,
    estimate_tokens,
)

logger = get_logger(__name__)


# ── File index cache ─────────────────────────────────────────────────────────
#
# Previously every call into FallbackGraphEngine re-read each file from disk
# and re-parsed it with `ast.parse` once per (file, symbol) lookup. For a real
# repo this dominated `optimize_prompt` latency. We now cache an index per
# absolute path, keyed on the file's mtime — so when nothing has changed on
# disk between MCP calls, lookups are O(1) hash hits.
#
# Cache lives on the engine *instance*. The MCP server creates one Optimizer
# (and therefore one FallbackGraphEngine) at startup and reuses it across
# every tool call, so warm-cache benefits accrue across the whole session.

@dataclass
class _FileIndex:
    """Cached per-file data, valid until the file's mtime changes."""
    path: str
    mtime: float
    source: str
    tree: Optional[ast.AST]
    imports: List[str] = field(default_factory=list)
    # Lazy per-symbol caches — populated on demand
    _defs: Dict[str, Optional[Tuple[int, int]]] = field(default_factory=dict)
    _call_sites: Dict[str, List[int]] = field(default_factory=dict)

    def find_definition(self, symbol_name: str) -> Optional[Tuple[int, int]]:
        if symbol_name in self._defs:
            return self._defs[symbol_name]
        result: Optional[Tuple[int, int]] = None
        if self.tree is not None:
            for node in ast.walk(self.tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name == symbol_name:
                        end = getattr(node, "end_lineno", node.lineno + 10)
                        result = (node.lineno, end)
                        break
        self._defs[symbol_name] = result
        return result

    def find_call_sites(self, symbol_name: str) -> List[int]:
        if symbol_name in self._call_sites:
            return self._call_sites[symbol_name]
        lines: List[int] = []
        if self.tree is not None:
            for node in ast.walk(self.tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = None
                    if isinstance(func, ast.Name):
                        name = func.id
                    elif isinstance(func, ast.Attribute):
                        name = func.attr
                    if name == symbol_name:
                        lines.append(node.lineno)
        self._call_sites[symbol_name] = lines
        return lines


class ImportResolver:
    """Resolves Python import statements to file paths."""

    def __init__(self, repo_root: str) -> None:
        self.repo_root = Path(repo_root)

    def resolve(self, import_name: str) -> Optional[str]:
        """Convert a dotted import name to a file path within the repo."""
        parts = import_name.split(".")
        candidates = [
            self.repo_root / Path(*parts).with_suffix(".py"),
            self.repo_root / Path(*parts) / "__init__.py",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def extract_imports(self, source: str) -> List[str]:
        """Extract all import names from Python source."""
        tree = parse_ast_safe(source)
        if tree is None:
            return []
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports


class SymbolResolver:
    """Resolves symbol references to their definition locations."""

    def find_definition(self, source: str, symbol_name: str) -> Optional[Tuple[int, int]]:
        """
        Find the line span of a symbol definition in source.
        Returns (start_line, end_line) or None.
        """
        tree = parse_ast_safe(source)
        if tree is None:
            return None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == symbol_name:
                    end = getattr(node, "end_lineno", node.lineno + 10)
                    return (node.lineno, end)
        return None

    def find_call_sites(self, source: str, symbol_name: str) -> List[int]:
        """Return line numbers where symbol_name is called."""
        tree = parse_ast_safe(source)
        if tree is None:
            return []
        lines = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name == symbol_name:
                    lines.append(node.lineno)
        return lines


class GitContextProvider:
    """Extracts relevant files from git history."""

    def __init__(self, repo_root: str) -> None:
        self.repo_root = repo_root

    def get_recently_changed_files(self, target_file: str, n: int = 5) -> List[str]:
        """Get files most frequently changed alongside target_file."""
        try:
            result = subprocess.run(
                ["git", "log", "--name-only", "--pretty=format:", "-n", "50"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return []
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            # Count co-occurrence with target_file
            commits: List[List[str]] = []
            current: List[str] = []
            for line in lines:
                if not line:
                    if current:
                        commits.append(current)
                    current = []
                else:
                    current.append(line)
            if current:
                commits.append(current)

            cooccurrence: Dict[str, int] = {}
            target_base = os.path.basename(target_file)
            for commit in commits:
                commit_bases = [os.path.basename(f) for f in commit]
                if target_base in commit_bases:
                    for f in commit:
                        if os.path.basename(f) != target_base:
                            cooccurrence[f] = cooccurrence.get(f, 0) + 1

            sorted_files = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)
            return [f for f, _ in sorted_files[:n]]
        except Exception:
            return []


class FallbackGraphEngine:
    """
    Standalone dependency graph engine — no external tools required.

    Combines AST analysis, import resolution, symbol tracking, call graphs,
    execution path discovery, git context, and folder heuristics.
    """

    def __init__(self, repo_root: Optional[str] = None) -> None:
        self.repo_root = repo_root or os.getcwd()
        self._import_resolver = ImportResolver(self.repo_root)
        self._symbol_resolver = SymbolResolver()
        self._git_ctx = GitContextProvider(self.repo_root)
        self._visited: Set[str] = set()
        # path → _FileIndex. Survives across MCP calls; invalidated by mtime.
        self._index_cache: Dict[str, _FileIndex] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    # ── File-index cache ─────────────────────────────────────────────────

    def _get_file_index(self, file_path: str) -> Optional[_FileIndex]:
        """
        Return a cached _FileIndex for `file_path`, refreshing if mtime changed.
        Returns None if the file is unreadable or unparseable on first load.
        """
        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            return None

        cached = self._index_cache.get(file_path)
        if cached is not None and cached.mtime == mtime:
            self._cache_hits += 1
            return cached

        self._cache_misses += 1
        source = safe_read_file(file_path)
        if source is None:
            return None
        tree = parse_ast_safe(source)
        imports = self._import_resolver.extract_imports(source) if tree is not None else []
        index = _FileIndex(
            path=file_path,
            mtime=mtime,
            source=source,
            tree=tree,
            imports=imports,
        )
        self._index_cache[file_path] = index
        return index

    def cache_stats(self) -> Dict[str, int]:
        """Return (hits, misses, size) for diagnostics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "entries": len(self._index_cache),
        }

    def clear_cache(self) -> None:
        """Drop all cached file indices. Called on explicit invalidation."""
        self._index_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def discover_dependencies(
        self,
        intent_lock: IntentLock,
        max_depth: int = 5,
    ) -> List[DependencyNode]:
        """
        Discover minimal required dependencies for the given intent.
        Respects expansion mode from IntentLock.
        """
        self._visited.clear()
        nodes: List[DependencyNode] = []

        for file_path in intent_lock.target_files:
            file_nodes = self._discover_from_file(
                file_path=file_path,
                symbols=intent_lock.target_symbols,
                expansion_mode=intent_lock.allowed_expansion.value,
                depth=0,
                max_depth=max_depth,
            )
            nodes.extend(file_nodes)

        return nodes

    def _discover_from_file(
        self,
        file_path: str,
        symbols: List[str],
        expansion_mode: str,
        depth: int,
        max_depth: int,
    ) -> List[DependencyNode]:
        if file_path in self._visited or depth > max_depth:
            return []
        self._visited.add(file_path)

        if expansion_mode == "strict" and depth > 0:
            return []

        nodes: List[DependencyNode] = []
        index = self._get_file_index(file_path)
        if index is None:
            return []

        # 1. Import resolution (uses cached imports)
        imports = index.imports
        for imp in imports:
            resolved = self._import_resolver.resolve(imp)
            if resolved and resolved not in self._visited:
                node = DependencyNode(
                    file_path=resolved,
                    dependency_type="import",
                    source_file=file_path,
                    depth=depth + 1,
                    confidence=0.85,
                    discovery_method="ast",
                )
                # Recursively discover if controlled/open mode
                if expansion_mode in ("controlled", "open"):
                    child_nodes = self._discover_from_file(
                        resolved, symbols, expansion_mode, depth + 1, max_depth
                    )
                    node.spans.extend(
                        self._extract_symbol_spans(resolved, symbols)
                    )
                    nodes.extend(child_nodes)
                nodes.append(node)

        # 2. Symbol call sites (cached per-symbol lookup on the index)
        for symbol in symbols:
            call_lines = index.find_call_sites(symbol)
            if call_lines:
                node = DependencyNode(
                    file_path=file_path,
                    dependency_type="call",
                    source_file=file_path,
                    target_symbol=symbol,
                    depth=depth,
                    confidence=0.90,
                    discovery_method="symbol",
                )
                nodes.append(node)

        # 3. Git context (only at depth 0, controlled/open mode)
        if depth == 0 and expansion_mode in ("controlled", "open"):
            git_files = self._git_ctx.get_recently_changed_files(file_path)
            for gf in git_files[:3]:
                full = os.path.join(self.repo_root, gf)
                if full not in self._visited and os.path.exists(full):
                    node = DependencyNode(
                        file_path=full,
                        dependency_type="git",
                        source_file=file_path,
                        depth=depth + 1,
                        confidence=0.55,
                        discovery_method="git",
                    )
                    nodes.append(node)

        return nodes

    def _extract_symbol_spans(self, file_path: str, symbols: List[str]) -> List[ContextSpan]:
        """Extract line spans for requested symbols from a file (uses cache)."""
        index = self._get_file_index(file_path)
        if index is None:
            return []
        spans = []
        source_lines = index.source.splitlines()
        for symbol in symbols:
            loc = index.find_definition(symbol)
            if loc:
                start, end = loc
                content = "\n".join(source_lines[start - 1 : end])
                spans.append(
                    ContextSpan(
                        file_path=file_path,
                        start_line=start,
                        end_line=end,
                        symbol=symbol,
                        confidence=0.88,
                        relevance_score=0.9,
                        token_cost=estimate_tokens(content),
                        content=content,
                        reason=f"Symbol definition: {symbol}",
                    )
                )
        return spans

    def build_call_graph(self, file_path: str) -> Dict[str, List[str]]:
        """Build a simple call graph for a file: {caller -> [callees]} (uses cache)."""
        index = self._get_file_index(file_path)
        if index is None or index.tree is None:
            return {}
        tree = index.tree
        graph: Dict[str, List[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                callees = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        if isinstance(func, ast.Name):
                            callees.append(func.id)
                        elif isinstance(func, ast.Attribute):
                            callees.append(func.attr)
                graph[node.name] = callees
        return graph
