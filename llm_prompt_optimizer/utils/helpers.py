"""Utility functions for LLM Prompt Optimizer."""

from __future__ import annotations

import ast
import logging
import json
import re
import hashlib
from pathlib import Path
from typing import Optional, List, Tuple, Any


def get_logger(name: str) -> logging.Logger:
    """Return a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation: ~4 chars per token (cl100k approximation).
    For production, swap with tiktoken or model-specific tokenizer.
    """
    return max(1, len(text) // 4)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def extract_file_paths(text: str) -> List[str]:
    """Extract file-like paths from text (e.g. signals/IndexSignals.py)."""
    pattern = r"[\w\-./]+\.\w{1,10}"
    matches = re.findall(pattern, text)
    return [m for m in matches if "/" in m or m.endswith((".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c"))]


def extract_symbols(text: str) -> List[str]:
    """Extract likely function/class names from text."""
    pattern = r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|[A-Z][a-z]+(?:[A-Z][a-z]+)+|[a-z_]+_[a-z_]+)\b"
    return list(set(re.findall(pattern, text)))


def safe_read_file(file_path: str) -> Optional[str]:
    """Read a file safely, returning None on error."""
    try:
        path = Path(file_path)
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass
    return None


def get_file_lines(file_path: str, start: int, end: int) -> Optional[str]:
    """Extract specific lines from a file (1-indexed, inclusive)."""
    content = safe_read_file(file_path)
    if content is None:
        return None
    lines = content.splitlines()
    s = max(0, start - 1)
    e = min(len(lines), end)
    return "\n".join(lines[s:e])


def parse_ast_safe(source: str) -> Optional[ast.AST]:
    """Parse Python source to AST, returning None on failure."""
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def detect_stacktrace(text: str) -> bool:
    """Heuristically detect if text contains a stack trace."""
    indicators = ["Traceback (most recent call last)", "  File ", "Error:", "Exception:"]
    return sum(1 for i in indicators if i in text) >= 2


def detect_log_content(text: str) -> bool:
    """Heuristically detect if text contains log output."""
    patterns = [r"\d{4}-\d{2}-\d{2}", r"INFO|WARN|ERROR|DEBUG|CRITICAL", r"\[.*?\].*?:"]
    return sum(1 for p in patterns if re.search(p, text)) >= 2


def detect_code_snippet(text: str) -> bool:
    """Heuristically detect if text contains a code snippet."""
    markers = ["```", "def ", "class ", "import ", "function ", "const ", "var ", "let "]
    return sum(1 for m in markers if m in text) >= 2


def cosine_similarity_simple(a: str, b: str) -> float:
    """
    Very lightweight token-overlap based similarity (no ML dependency required).
    For production, replace with sentence-transformers or similar.
    """
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)  # Jaccard as fallback
