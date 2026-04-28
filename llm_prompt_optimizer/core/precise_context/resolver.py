"""
PreciseContextResolver — Line-level context extraction.

PRINCIPLE: Relevant Lines > Relevant Files
NEVER sends entire files.

Produces ContextSpan objects with:
  - Exact line numbers
  - Symbol-aware extraction
  - AST span computation
  - Call-site tracing
  - Execution window discovery
  - Minimal span generation
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional, Tuple

from llm_prompt_optimizer.models.context import ContextSpan
from llm_prompt_optimizer.models.intent import IntentLock
from llm_prompt_optimizer.models.classification import PromptClassification
from llm_prompt_optimizer.utils.helpers import (
    get_logger,
    safe_read_file,
    parse_ast_safe,
    estimate_tokens,
    get_file_lines,
)

logger = get_logger(__name__)

# Lines of context to include above/below a symbol definition
_CONTEXT_WINDOW_LINES = 5


class PreciseContextResolver:
    """
    Extracts minimal, line-level context from source files.

    Priority:
      1. Exact symbol definition spans
      2. Call-site windows
      3. Execution path windows
      4. Heuristic line windows (stacktrace line numbers)
    """

    def resolve(
        self,
        intent_lock: IntentLock,
        classification: PromptClassification,
        repo_root: Optional[str] = None,
    ) -> List[ContextSpan]:
        """
        Main entry point. Returns minimal line-level spans for all target files.
        """
        spans: List[ContextSpan] = []

        for file_path in intent_lock.target_files:
            file_spans = self._resolve_file(
                file_path=file_path,
                symbols=intent_lock.target_symbols,
                classification=classification,
                repo_root=repo_root,
            )
            spans.extend(file_spans)

        return spans

    def _resolve_file(
        self,
        file_path: str,
        symbols: List[str],
        classification: PromptClassification,
        repo_root: Optional[str],
    ) -> List[ContextSpan]:
        source = safe_read_file(file_path)
        if source is None:
            logger.warning(f"Cannot read file: {file_path}")
            return []

        lines = source.splitlines()
        total_lines = len(lines)
        spans: List[ContextSpan] = []

        if symbols:
            # Strategy 1: Symbol-based extraction
            for symbol in symbols:
                span = self._extract_symbol_definition(file_path, source, symbol, total_lines)
                if span:
                    spans.append(span)
                call_spans = self._extract_call_sites(file_path, source, symbol, total_lines)
                spans.extend(call_spans)

        if not spans:
            # Strategy 2: Stacktrace line extraction
            if classification.has_stacktrace:
                stack_spans = self._extract_from_stacktrace_hints(
                    file_path, source, total_lines
                )
                spans.extend(stack_spans)

        if not spans:
            # Strategy 3: Fallback — top N lines of the file as minimal context
            span = self._extract_head_context(file_path, source, total_lines)
            if span:
                spans.append(span)

        return spans

    def _extract_symbol_definition(
        self,
        file_path: str,
        source: str,
        symbol: str,
        total_lines: int,
    ) -> Optional[ContextSpan]:
        """Extract the exact definition span of a function or class."""
        tree = parse_ast_safe(source)
        if tree is None:
            return None

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == symbol:
                    start = max(1, node.lineno - _CONTEXT_WINDOW_LINES)
                    end_node = getattr(node, "end_lineno", node.lineno + 20)
                    end = min(total_lines, end_node + 2)
                    content = get_file_lines(file_path, start, end) or ""
                    return ContextSpan(
                        file_path=file_path,
                        start_line=start,
                        end_line=end,
                        symbol=symbol,
                        confidence=0.93,
                        relevance_score=0.95,
                        token_cost=estimate_tokens(content),
                        content=content,
                        reason=f"Symbol definition: {symbol}",
                        execution_proximity=1.0,
                    )
        return None

    def _extract_call_sites(
        self,
        file_path: str,
        source: str,
        symbol: str,
        total_lines: int,
    ) -> List[ContextSpan]:
        """Extract windows around every call site of a symbol."""
        tree = parse_ast_safe(source)
        if tree is None:
            return []

        call_lines: List[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name == symbol:
                    call_lines.append(node.lineno)

        spans = []
        for lineno in call_lines[:5]:  # cap at 5 call sites per symbol
            start = max(1, lineno - 3)
            end = min(total_lines, lineno + 5)
            content = get_file_lines(file_path, start, end) or ""
            spans.append(
                ContextSpan(
                    file_path=file_path,
                    start_line=start,
                    end_line=end,
                    symbol=symbol,
                    confidence=0.78,
                    relevance_score=0.80,
                    token_cost=estimate_tokens(content),
                    content=content,
                    reason=f"Call site: {symbol} at line {lineno}",
                    execution_proximity=0.85,
                )
            )
        return spans

    def _extract_from_stacktrace_hints(
        self,
        file_path: str,
        source: str,
        total_lines: int,
    ) -> List[ContextSpan]:
        """
        If source contains 'line N' references (from embedded stacktrace),
        extract windows around those lines.
        """
        import re
        pattern = re.compile(r"line\s+(\d+)", re.IGNORECASE)
        matches = pattern.findall(source)
        spans = []
        for match in matches[:3]:
            lineno = int(match)
            if 1 <= lineno <= total_lines:
                start = max(1, lineno - 5)
                end = min(total_lines, lineno + 10)
                content = get_file_lines(file_path, start, end) or ""
                spans.append(
                    ContextSpan(
                        file_path=file_path,
                        start_line=start,
                        end_line=end,
                        confidence=0.65,
                        relevance_score=0.70,
                        token_cost=estimate_tokens(content),
                        content=content,
                        reason=f"Stacktrace reference: line {lineno}",
                        execution_proximity=0.75,
                    )
                )
        return spans

    def _extract_head_context(
        self,
        file_path: str,
        source: str,
        total_lines: int,
    ) -> Optional[ContextSpan]:
        """Fallback: first 60 lines as minimal file context."""
        end = min(60, total_lines)
        content = get_file_lines(file_path, 1, end) or ""
        if not content:
            return None
        return ContextSpan(
            file_path=file_path,
            start_line=1,
            end_line=end,
            confidence=0.50,
            relevance_score=0.55,
            token_cost=estimate_tokens(content),
            content=content,
            reason="Fallback: file header context",
            execution_proximity=0.50,
        )

    def resolve_span_for_dependency(
        self,
        file_path: str,
        symbols: List[str],
    ) -> List[ContextSpan]:
        """Resolve spans for a dependency file (lower base confidence)."""
        source = safe_read_file(file_path)
        if not source:
            return []
        total_lines = len(source.splitlines())
        spans = []
        for symbol in symbols:
            span = self._extract_symbol_definition(file_path, source, symbol, total_lines)
            if span:
                # Lower confidence for dependency spans
                span.confidence *= 0.85
                span.relevance_score *= 0.80
                span.reason = f"Dependency symbol: {symbol}"
                spans.append(span)
        return spans
