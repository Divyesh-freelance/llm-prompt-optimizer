"""Intent-related data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ExpansionMode(str, Enum):
    STRICT = "strict"
    CONTROLLED = "controlled"
    OPEN = "open"


@dataclass
class IntentConstraint:
    """A single constraint extracted from a user prompt."""

    constraint_type: str  # e.g. "no_code_changes", "no_repo_scan", "file_scope"
    value: Any
    source: str = "user_explicit"  # user_explicit | inferred | policy
    confidence: float = 1.0


@dataclass
class IntentLock:
    """
    Locked representation of user intent.
    Once set, all downstream pipeline stages read from this and MUST NOT override it.
    """

    intent_summary: str
    task_category: str
    target_files: List[str] = field(default_factory=list)
    target_symbols: List[str] = field(default_factory=list)
    constraints: List[IntentConstraint] = field(default_factory=list)
    allowed_expansion: ExpansionMode = ExpansionMode.CONTROLLED
    forbidden_files: List[str] = field(default_factory=list)
    repo_boundaries: Optional[str] = None
    output_expectations: Optional[str] = None
    confidence: float = 1.0
    locked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def lock(self) -> None:
        """Freeze the intent lock — downstream stages cannot modify it after this."""
        self.locked = True

    def assert_unlocked(self) -> None:
        if self.locked:
            raise RuntimeError(
                "IntentLock is sealed. No downstream stage may modify user intent."
            )
