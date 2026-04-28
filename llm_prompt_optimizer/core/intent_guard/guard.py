"""
IntentGuard — Highest-priority pipeline component.

Extracts and locks user intent before any other processing occurs.
Prevents hallucination, scope drift, and unauthorized task expansion.

INVARIANT: Once an IntentLock is sealed, NO downstream stage may alter it.
"""

from __future__ import annotations

import re
from typing import List, Optional

from llm_prompt_optimizer.models.intent import IntentLock, IntentConstraint, ExpansionMode
from llm_prompt_optimizer.models.prompt import RawPrompt
from llm_prompt_optimizer.utils.helpers import (
    get_logger,
    extract_file_paths,
    extract_symbols,
    detect_stacktrace,
    detect_log_content,
)

logger = get_logger(__name__)

# Explicit constraint patterns — look for strong user signals
_NO_CHANGE_PATTERNS = [
    r"\bdon[''']?t\s+(?:change|modify|alter|edit|touch)\b",
    r"\bno\s+(?:code\s+)?changes?\b",
    r"\bread[\s-]only\b",
    r"\bjust\s+(?:explain|show|debug|find|identify|check)\b",
]

_NO_SCAN_PATTERNS = [
    r"\bno\s+(?:repo|repository)\s+scan\b",
    r"\bdon[''']?t\s+scan\b",
    r"\bstay\s+(?:in|within)\s+(?:this\s+)?file\b",
    r"\bonly\s+(?:this|the\s+specified?|the\s+given)\s+file\b",
]

_EXPANSION_CONTROL = {
    "strict": [r"\bstrict(?:ly)?\b", r"\bno\s+expansion\b", r"\bexact\s+file\b"],
    "open": [r"\bexplore\b", r"\bscan\s+(?:the\s+)?(?:repo|codebase)\b", r"\bbroad(?:ly)?\b"],
}

# Debugging verbs — signal intent without creating new tasks
_DEBUG_VERBS = [
    "debug", "fix", "investigate", "diagnose", "trace", "identify",
    "analyze", "understand", "explain", "find", "check", "inspect",
    "why", "what causes", "how does",
]


class IntentGuard:
    """
    Extracts user intent from a raw prompt and locks it immutably.

    This component runs FIRST in the pipeline and produces an IntentLock
    that all subsequent stages must respect without modification.
    """

    def __init__(self) -> None:
        self._compiled_no_change = [re.compile(p, re.IGNORECASE) for p in _NO_CHANGE_PATTERNS]
        self._compiled_no_scan = [re.compile(p, re.IGNORECASE) for p in _NO_SCAN_PATTERNS]
        self._compiled_expansion = {
            k: [re.compile(p, re.IGNORECASE) for p in pats]
            for k, pats in _EXPANSION_CONTROL.items()
        }

    def extract_and_lock(self, prompt: RawPrompt) -> IntentLock:
        """
        Parse the prompt text, extract intent signals, and return a sealed IntentLock.
        """
        text = prompt.text

        intent_summary = self._extract_intent_summary(text)
        task_category = self._infer_task_category(text)
        target_files = extract_file_paths(text)
        if prompt.working_directory:
            target_files = [
                f if f.startswith("/") else f
                for f in target_files
            ]
        target_symbols = extract_symbols(text)
        constraints = self._extract_constraints(text)
        expansion_mode = self._infer_expansion_mode(text, constraints)
        repo_boundaries = prompt.repo_root
        output_expectations = self._infer_output_expectations(text)
        confidence = self._compute_confidence(text, target_files, target_symbols)

        lock = IntentLock(
            intent_summary=intent_summary,
            task_category=task_category,
            target_files=target_files,
            target_symbols=target_symbols,
            constraints=constraints,
            allowed_expansion=expansion_mode,
            repo_boundaries=repo_boundaries,
            output_expectations=output_expectations,
            confidence=confidence,
        )
        lock.lock()

        logger.info(
            "IntentLock sealed",
            extra={
                "prompt_id": prompt.prompt_id,
                "intent": intent_summary,
                "category": task_category,
                "files": target_files,
                "expansion": expansion_mode.value,
                "confidence": confidence,
            },
        )
        return lock

    # ── Private extraction helpers ────────────────────────────────────────────

    def _extract_intent_summary(self, text: str) -> str:
        """
        Distill the core user goal into a terse, literal summary.
        Never embellishes or generalizes — preserves exact meaning.
        """
        text_lower = text.lower().strip()

        # Prefer the first sentence as the most direct statement of intent
        sentences = re.split(r"[.!?\n]", text)
        primary = sentences[0].strip() if sentences else text

        # If it's short, it IS the intent
        if len(primary) <= 200:
            return primary

        # Truncate at word boundary
        words = primary.split()
        return " ".join(words[:40]) + "..."

    def _infer_task_category(self, text: str) -> str:
        """Rule-based task category inference (no external ML required)."""
        text_lower = text.lower()

        category_signals = {
            "debugging": _DEBUG_VERBS + ["bug", "error", "mismatch", "wrong", "fail", "broken"],
            "implementation": ["implement", "build", "create", "add feature", "write"],
            "refactoring": ["refactor", "clean up", "restructure", "reorganize"],
            "optimization": ["optimize", "speed up", "performance", "slow", "latency"],
            "testing": ["test", "spec", "pytest", "unittest", "coverage"],
            "architecture": ["architect", "design", "system design", "diagram"],
            "migration": ["migrate", "upgrade", "move from", "convert"],
            "security": ["security", "vulnerability", "auth", "permission", "exploit"],
            "log_analysis": ["log", "trace", "stacktrace", "traceback", "exception"],
        }

        scores: dict[str, int] = {}
        for category, signals in category_signals.items():
            scores[category] = sum(1 for s in signals if s in text_lower)

        best = max(scores, key=lambda k: scores[k]) if scores else "unknown"
        if scores.get(best, 0) == 0:
            return "unknown"
        return best

    def _extract_constraints(self, text: str) -> List[IntentConstraint]:
        """Extract explicit user constraints from the prompt text."""
        constraints: List[IntentConstraint] = []

        for pattern in self._compiled_no_change:
            if pattern.search(text):
                constraints.append(
                    IntentConstraint(
                        constraint_type="no_code_changes",
                        value=True,
                        source="user_explicit",
                        confidence=0.95,
                    )
                )
                break

        for pattern in self._compiled_no_scan:
            if pattern.search(text):
                constraints.append(
                    IntentConstraint(
                        constraint_type="no_repo_scan",
                        value=True,
                        source="user_explicit",
                        confidence=0.95,
                    )
                )
                break

        if detect_stacktrace(text):
            constraints.append(
                IntentConstraint(
                    constraint_type="has_stacktrace",
                    value=True,
                    source="inferred",
                    confidence=0.90,
                )
            )

        if detect_log_content(text):
            constraints.append(
                IntentConstraint(
                    constraint_type="has_logs",
                    value=True,
                    source="inferred",
                    confidence=0.85,
                )
            )

        return constraints

    def _infer_expansion_mode(
        self, text: str, constraints: List[IntentConstraint]
    ) -> ExpansionMode:
        """Determine how aggressively context may be expanded."""
        constraint_types = {c.constraint_type for c in constraints}
        if "no_repo_scan" in constraint_types:
            return ExpansionMode.STRICT

        for pattern in self._compiled_expansion["open"]:
            if pattern.search(text):
                return ExpansionMode.OPEN

        for pattern in self._compiled_expansion["strict"]:
            if pattern.search(text):
                return ExpansionMode.STRICT

        return ExpansionMode.CONTROLLED

    def _infer_output_expectations(self, text: str) -> Optional[str]:
        """Extract what the user expects as output."""
        text_lower = text.lower()
        if any(v in text_lower for v in ["explain", "why", "what"]):
            return "explanation"
        if any(v in text_lower for v in ["fix", "implement", "write", "build"]):
            return "code_change"
        if "list" in text_lower or "show" in text_lower:
            return "listing"
        return None

    def _compute_confidence(
        self, text: str, files: List[str], symbols: List[str]
    ) -> float:
        """
        Confidence in intent extraction.
        Higher when files and symbols are explicit, lower for vague prompts.
        """
        score = 0.6  # base
        if files:
            score += min(0.2, len(files) * 0.07)
        if symbols:
            score += min(0.1, len(symbols) * 0.03)
        if len(text.split()) > 15:
            score += 0.1
        return min(1.0, round(score, 3))
