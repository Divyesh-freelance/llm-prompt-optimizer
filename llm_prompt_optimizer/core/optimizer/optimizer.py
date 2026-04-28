"""
PromptOptimizer — Core optimization stage.

ALLOWED:
  - compression (remove redundancy)
  - formatting (structure for clarity)
  - constraint injection (embed intent_lock constraints)
  - structure normalization

FORBIDDEN:
  - creative rewriting
  - inferred goals (never add what wasn't stated)
  - architecture expansion
  - business assumptions

HIGHEST PRIORITY: Intent preservation > compression
"""

from __future__ import annotations

import re
from typing import List, Optional

from llm_prompt_optimizer.models.prompt import RawPrompt, OptimizedPrompt
from llm_prompt_optimizer.models.intent import IntentLock
from llm_prompt_optimizer.models.context import ContextBundle
from llm_prompt_optimizer.models.classification import PromptClassification
from llm_prompt_optimizer.utils.helpers import get_logger, estimate_tokens

logger = get_logger(__name__)

# Redundancy patterns — safe to normalize (not to remove)
_REDUNDANT_PHRASES = [
    (r"\bplease\s+", ""),
    (r"\bcould\s+you\s+(?:please\s+)?", ""),
    (r"\bi\s+would\s+(?:like|love)\s+(?:you\s+to\s+)?", ""),
    (r"\bcan\s+you\s+(?:please\s+)?", ""),
    (r"\bwould\s+you\s+(?:mind\s+)?", ""),
    (r"\s{2,}", " "),  # collapse multiple spaces
]


class PromptOptimizer:
    """
    Transforms a raw prompt into an optimized, context-injected prompt.

    Does NOT rewrite, expand scope, or invent intent.
    """

    def __init__(self) -> None:
        self._redundancy_patterns = [
            (re.compile(p, re.IGNORECASE), r) for p, r in _REDUNDANT_PHRASES
        ]

    def optimize(
        self,
        raw_prompt: RawPrompt,
        intent_lock: IntentLock,
        classification: PromptClassification,
        context_bundle: Optional[ContextBundle] = None,
        strict_mode: bool = False,
    ) -> OptimizedPrompt:
        """
        Main optimization entry point.

        Steps:
          1. Normalize whitespace / light compression
          2. Inject constraint directives
          3. Inject context spans (minimal, line-level)
          4. Inject structured task header
          5. Final formatting
        """
        text = raw_prompt.text

        # Step 1: Light compression — remove only redundant politeness filler
        compressed = self._compress(text)

        # Step 2: Inject constraint block
        constraint_block = self._build_constraint_block(intent_lock)

        # Step 3: Build context injection block
        context_block = ""
        if context_bundle and context_bundle.spans:
            context_block = self._build_context_block(context_bundle, intent_lock)

        # Step 4: Assemble optimized prompt
        parts: List[str] = []

        # Task header — literal, not invented
        task_header = self._build_task_header(intent_lock, classification)
        parts.append(task_header)

        # Original compressed intent
        parts.append("## Task\n" + compressed.strip())

        # Constraints
        if constraint_block:
            parts.append(constraint_block)

        # Context
        if context_block:
            parts.append(context_block)

        optimized_text = "\n\n".join(parts)

        # Compute metrics
        original_tokens = estimate_tokens(raw_prompt.text)
        optimized_tokens = estimate_tokens(optimized_text)
        compression_ratio = (
            original_tokens / optimized_tokens if optimized_tokens > 0 else 1.0
        )

        result = OptimizedPrompt(
            text=optimized_text,
            original_prompt_id=raw_prompt.prompt_id,
            token_estimate=optimized_tokens,
            context_spans=[
                {
                    "file": s.file_path,
                    "start_line": s.start_line,
                    "end_line": s.end_line,
                    "symbol": s.symbol,
                    "confidence": s.confidence,
                }
                for s in (context_bundle.spans if context_bundle else [])
            ],
            injected_constraints=[c.constraint_type for c in intent_lock.constraints],
            compression_ratio=round(compression_ratio, 3),
            confidence=intent_lock.confidence,
            metadata={
                "category": classification.primary_category.value,
                "complexity": classification.complexity_score,
                "strict_mode": strict_mode,
            },
        )

        logger.info(
            f"Prompt optimized: {original_tokens}→{optimized_tokens} tokens "
            f"(ratio={compression_ratio:.2f})"
        )
        return result

    # ── Private helpers ──────────────────────────────────────────────────────

    def _compress(self, text: str) -> str:
        """Apply safe, non-semantic compression — only removes filler."""
        result = text
        for pattern, replacement in self._redundancy_patterns:
            result = pattern.sub(replacement, result)
        return result.strip()

    def _build_task_header(
        self, intent_lock: IntentLock, classification: PromptClassification
    ) -> str:
        lines = [
            f"# Optimization Context",
            f"**Category**: {classification.primary_category.value}",
            f"**Intent**: {intent_lock.intent_summary}",
            f"**Expansion**: {intent_lock.allowed_expansion.value}",
        ]
        if intent_lock.target_files:
            files = ", ".join(f"`{f}`" for f in intent_lock.target_files[:5])
            lines.append(f"**Scope**: {files}")
        return "\n".join(lines)

    def _build_constraint_block(self, intent_lock: IntentLock) -> str:
        if not intent_lock.constraints:
            return ""
        lines = ["## Constraints"]
        for c in intent_lock.constraints:
            if c.constraint_type == "no_code_changes":
                lines.append("- Do NOT make code changes. Explain only.")
            elif c.constraint_type == "no_repo_scan":
                lines.append("- Do NOT scan the full repository. Stay within specified files.")
            elif c.constraint_type == "has_stacktrace":
                lines.append("- A stacktrace is included. Focus on the error origin.")
            elif c.constraint_type == "has_logs":
                lines.append("- Log output is included. Trace the log sequence.")
            else:
                lines.append(f"- {c.constraint_type}: {c.value}")
        return "\n".join(lines)

    def _build_context_block(
        self, bundle: ContextBundle, intent_lock: IntentLock
    ) -> str:
        lines = ["## Relevant Context"]
        for span in bundle.spans:
            header = f"### `{span.file_path}` (lines {span.start_line}–{span.end_line})"
            if span.symbol:
                header += f" · `{span.symbol}`"
            if span.reason:
                header += f"\n> {span.reason}"
            code = span.content or "(content not loaded)"
            lines.append(header)
            lines.append(f"```\n{code}\n```")
        return "\n\n".join(lines)
