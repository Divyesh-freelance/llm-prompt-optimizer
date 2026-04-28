"""Context-related data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ContextSpan:
    """A precise, line-level slice of a source file."""

    file_path: str
    start_line: int
    end_line: int
    symbol: Optional[str] = None
    confidence: float = 1.0
    relevance_score: float = 1.0
    token_cost: int = 0
    content: Optional[str] = None
    reason: Optional[str] = None  # why this span was included
    execution_proximity: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def line_count(self) -> int:
        return max(0, self.end_line - self.start_line + 1)

    def context_value_score(self) -> float:
        """
        Value-based score: relevance × confidence × execution_proximity / token_cost.
        Higher is better.
        """
        denominator = max(1, self.token_cost)
        return (self.relevance_score * self.confidence * self.execution_proximity) / denominator


@dataclass
class DependencyNode:
    """Represents a discovered dependency in the codebase."""

    file_path: str
    dependency_type: str  # import | call | symbol | runtime | git
    source_file: Optional[str] = None
    source_symbol: Optional[str] = None
    target_symbol: Optional[str] = None
    depth: int = 0
    confidence: float = 1.0
    discovery_method: str = "ast"  # ast | import | symbol | git | heuristic
    spans: List[ContextSpan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextBundle:
    """Complete context package assembled for prompt injection."""

    spans: List[ContextSpan] = field(default_factory=list)
    dependencies: List[DependencyNode] = field(default_factory=list)
    total_token_cost: int = 0
    total_lines: int = 0
    files_included: List[str] = field(default_factory=list)
    expansion_stopped_reason: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_span(self, span: ContextSpan) -> None:
        self.spans.append(span)
        self.total_token_cost += span.token_cost
        self.total_lines += span.line_count()
        if span.file_path not in self.files_included:
            self.files_included.append(span.file_path)
