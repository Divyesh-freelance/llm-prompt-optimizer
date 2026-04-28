"""
PromptClassifier — Multi-layer, multi-label prompt classification.

Does NOT use keyword-only matching.
Layers:
  1. Syntax detection (AST, code patterns)
  2. Log / stacktrace detection
  3. File path / symbol extraction
  4. NLP pattern matching (rule-based NLU)
  5. Complexity scoring
  6. Multi-label scoring
  7. Optional: semantic embedding similarity (when available)
"""

from __future__ import annotations

import re
from typing import List, Dict, Tuple

from llm_prompt_optimizer.models.classification import (
    PromptClassification,
    TaskCategory,
    ClassificationLabel,
)
from llm_prompt_optimizer.models.intent import IntentLock
from llm_prompt_optimizer.models.prompt import RawPrompt
from llm_prompt_optimizer.utils.helpers import (
    get_logger,
    detect_stacktrace,
    detect_log_content,
    detect_code_snippet,
    extract_file_paths,
    extract_symbols,
)

logger = get_logger(__name__)

# ── Category signal definitions ──────────────────────────────────────────────
# Each category has a list of (pattern, weight) tuples.
# Multiple evidence types contribute to a score — not just simple keyword hits.

_CATEGORY_SIGNALS: Dict[TaskCategory, List[Tuple[str, float]]] = {
    TaskCategory.DEBUGGING: [
        (r"\b(debug|fix|bug|broken|error|mismatch|wrong|fail|crash|exception|traceback)\b", 2.0),
        (r"\b(why|what\s+causes?|how\s+does|investigate|diagnose|trace|identify)\b", 1.5),
        (r"\b(unexpected|incorrect|off.by|discrepancy|drift)\b", 1.2),
    ],
    TaskCategory.IMPLEMENTATION: [
        (r"\b(implement|build|create|add|write|develop|make)\b", 2.0),
        (r"\b(feature|endpoint|function|class|module|service)\b", 1.0),
        (r"\b(from\s+scratch|new\s+file|new\s+module)\b", 1.5),
    ],
    TaskCategory.REFACTORING: [
        (r"\b(refactor|clean\s+up|restructure|reorganize|simplify|extract)\b", 2.5),
        (r"\b(duplicate|coupling|tight|smell|technical\s+debt)\b", 1.2),
    ],
    TaskCategory.OPTIMIZATION: [
        (r"\b(optim|speed\s+up|performance|slow|latency|throughput|bottleneck)\b", 2.5),
        (r"\b(profile|benchmark|cache|memory|cpu|io)\b", 1.5),
    ],
    TaskCategory.TESTING: [
        (r"\b(test|spec|coverage|pytest|unittest|mock|stub|assert)\b", 2.5),
        (r"\b(e2e|integration\s+test|unit\s+test|regression)\b", 2.0),
    ],
    TaskCategory.ARCHITECTURE: [
        (r"\b(architect|design|system\s+design|diagram|component|service\s+mesh)\b", 2.5),
        (r"\b(scalab|pattern|microservice|monolith|event.driven)\b", 1.5),
    ],
    TaskCategory.SECURITY: [
        (r"\b(security|vulnerab|auth|permission|exploit|injection|xss|csrf|privilege)\b", 2.5),
        (r"\b(sanitize|validate|encrypt|token|secret)\b", 1.5),
    ],
    TaskCategory.LOG_ANALYSIS: [
        (r"\b(log|trace|traceback|stacktrace|exception|stderr|stdout)\b", 2.0),
        (r"\b(tail|grep|parse\s+log|log\s+line|log\s+entry)\b", 1.8),
    ],
    TaskCategory.ML_AI: [
        (r"\b(model|ema|signal|embedding|loss|train|inference|neural|llm|gpt|bert)\b", 2.0),
        (r"\b(weights|gradient|epoch|batch|predict|classify)\b", 1.5),
    ],
    TaskCategory.MIGRATION: [
        (r"\b(migrat|upgrade|move\s+from|convert|port|transition)\b", 2.5),
    ],
    TaskCategory.DEVOPS: [
        (r"\b(deploy|ci|cd|pipeline|docker|kubernetes|helm|ansible|terraform)\b", 2.5),
        (r"\b(devops|build|release|artifact|registry)\b", 1.5),
    ],
    TaskCategory.DATABASE: [
        (r"\b(sql|query|schema|migration|orm|index|transaction|database|db)\b", 2.5),
        (r"\b(postgres|mysql|sqlite|mongodb|redis|cassandra)\b", 2.0),
    ],
    TaskCategory.CONCURRENCY: [
        (r"\b(thread|async|await|concurrent|parallel|race\s+condition|deadlock|lock)\b", 2.5),
        (r"\b(asyncio|goroutine|coroutine|future|promise)\b", 2.0),
    ],
}


class PromptClassifier:
    """
    Multi-layer, multi-label prompt classifier.

    Combines syntax detection, log/trace awareness, NLP pattern matching,
    and complexity scoring to produce a PromptClassification.
    """

    def __init__(self) -> None:
        self._compiled: Dict[TaskCategory, List[Tuple[re.Pattern, float]]] = {
            cat: [(re.compile(pat, re.IGNORECASE), w) for pat, w in sigs]
            for cat, sigs in _CATEGORY_SIGNALS.items()
        }

    def classify(self, prompt: RawPrompt, intent_lock: IntentLock) -> PromptClassification:
        text = prompt.text

        # Layer 1 — Structural detectors
        has_stacktrace = detect_stacktrace(text)
        has_logs = detect_log_content(text)
        has_code = detect_code_snippet(text)

        # Layer 2 — File / symbol extraction
        file_paths = extract_file_paths(text)
        symbols = extract_symbols(text)

        # If intent lock already captured files, merge them
        all_files = list(set(file_paths + intent_lock.target_files))

        # Layer 3 — Multi-label scoring
        scores: Dict[TaskCategory, float] = {}
        for cat, compiled_sigs in self._compiled.items():
            score = 0.0
            for pattern, weight in compiled_sigs:
                matches = pattern.findall(text)
                score += weight * min(len(matches), 3)  # cap multiplier at 3 per pattern
            scores[cat] = score

        # Layer 4 — Boost based on structural evidence
        if has_stacktrace:
            scores[TaskCategory.DEBUGGING] = scores.get(TaskCategory.DEBUGGING, 0) + 3.0
            scores[TaskCategory.LOG_ANALYSIS] = scores.get(TaskCategory.LOG_ANALYSIS, 0) + 1.5

        if has_logs:
            scores[TaskCategory.LOG_ANALYSIS] = scores.get(TaskCategory.LOG_ANALYSIS, 0) + 2.0
            scores[TaskCategory.DEBUGGING] = scores.get(TaskCategory.DEBUGGING, 0) + 1.0

        # Layer 5 — Category from intent lock (trusted signal)
        lock_cat = self._map_intent_category(intent_lock.task_category)
        if lock_cat:
            scores[lock_cat] = scores.get(lock_cat, 0) + 2.0

        # Normalize
        total = sum(scores.values()) or 1.0
        normalized = {cat: s / total for cat, s in scores.items()}

        # Build label list
        labels = [
            ClassificationLabel(
                category=cat,
                confidence=round(conf, 4),
                method="layered_nlp",
            )
            for cat, conf in sorted(normalized.items(), key=lambda x: x[1], reverse=True)
            if conf > 0.01
        ]

        primary = labels[0] if labels else ClassificationLabel(
            category=TaskCategory.UNKNOWN, confidence=0.0, method="fallback"
        )

        # Complexity scoring
        complexity = self._score_complexity(text, file_paths, symbols, has_stacktrace, has_code)

        result = PromptClassification(
            primary_category=primary.category,
            primary_confidence=primary.confidence,
            labels=labels,
            complexity_score=complexity,
            has_stacktrace=has_stacktrace,
            has_logs=has_logs,
            has_code_snippet=has_code,
            extracted_file_paths=all_files,
            extracted_symbols=symbols,
            requires_execution_path=self._needs_exec_path(primary.category, has_stacktrace),
            multi_file=len(all_files) > 1,
        )

        logger.info(
            "Prompt classified",
            extra={
                "prompt_id": prompt.prompt_id,
                "primary": primary.category.value,
                "confidence": primary.confidence,
                "complexity": complexity,
            },
        )
        return result

    def _map_intent_category(self, category_str: str) -> TaskCategory | None:
        try:
            return TaskCategory(category_str)
        except ValueError:
            return None

    def _score_complexity(
        self,
        text: str,
        files: List[str],
        symbols: List[str],
        has_trace: bool,
        has_code: bool,
    ) -> float:
        """
        Score task complexity from 0.0 (trivial) to 1.0 (high complexity).
        """
        score = 0.0
        word_count = len(text.split())
        score += min(0.3, word_count / 300)      # more words → more complex
        score += min(0.2, len(files) * 0.05)     # more files → more scope
        score += min(0.15, len(symbols) * 0.03)  # more symbols → more complexity
        if has_trace:
            score += 0.15
        if has_code:
            score += 0.1
        return min(1.0, round(score, 3))

    def _needs_exec_path(self, category: TaskCategory, has_trace: bool) -> bool:
        return category in {
            TaskCategory.DEBUGGING,
            TaskCategory.LOG_ANALYSIS,
            TaskCategory.CONCURRENCY,
            TaskCategory.INCIDENT_ANALYSIS if hasattr(TaskCategory, "INCIDENT_ANALYSIS") else category,
        } or has_trace
