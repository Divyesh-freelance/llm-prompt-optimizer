"""
PromptCompiler — Final stage. Assembles the optimized prompt for LLM consumption.

ALLOWED:
  - compression
  - formatting
  - constraint injection
  - structure normalization

FORBIDDEN:
  - creative rewriting
  - inferred goals
  - architecture expansion
  - business assumptions
"""

from __future__ import annotations

from llm_prompt_optimizer.models.prompt import OptimizedPrompt
from llm_prompt_optimizer.models.intent import IntentLock
from llm_prompt_optimizer.models.validation import ValidationResult, DriftReport
from llm_prompt_optimizer.utils.helpers import get_logger

logger = get_logger(__name__)


class PromptCompiler:
    """
    Compiles the final prompt string ready for LLM consumption.

    Validates that validation and drift are clear before finalizing.
    If drift is blocked or validation failed, raises or returns fallback.
    """

    def compile(
        self,
        optimized: OptimizedPrompt,
        intent_lock: IntentLock,
        validation: ValidationResult,
        drift_report: DriftReport,
        fallback_to_raw: bool = True,
        raw_text: str = "",
    ) -> str:
        """
        Finalize the prompt. Returns the compiled prompt string.

        If validation failed or drift blocked, falls back to annotated raw prompt.
        """
        if drift_report.blocked:
            logger.warning("Drift blocked — falling back to constrained raw prompt")
            return self._compile_safe_fallback(raw_text, intent_lock)

        if validation.rejected:
            logger.warning(
                f"Semantic validation failed ({validation.failure_reason}) "
                f"— falling back to raw prompt"
            )
            if fallback_to_raw and raw_text:
                return self._compile_safe_fallback(raw_text, intent_lock)
            raise ValueError(
                f"Prompt optimization rejected: {validation.failure_reason}"
            )

        # Compile final output
        compiled = optimized.text

        # Append compiler footer (metadata, not instructions)
        footer = self._build_footer(optimized, validation)
        compiled = compiled + "\n\n" + footer

        logger.info(
            f"Prompt compiled: {optimized.token_estimate} tokens, "
            f"similarity={validation.semantic_similarity:.3f}"
        )
        return compiled

    def _compile_safe_fallback(self, raw_text: str, intent_lock: IntentLock) -> str:
        """Return a minimally annotated raw prompt as safe fallback."""
        constraints = "\n".join(
            f"- {c.constraint_type}" for c in intent_lock.constraints
        )
        parts = [f"# Task\n{raw_text}"]
        if constraints:
            parts.append(f"# Constraints (preserved)\n{constraints}")
        return "\n\n".join(parts)

    def _build_footer(self, optimized: OptimizedPrompt, validation: ValidationResult) -> str:
        return (
            f"---\n"
            f"*Optimization ID: {optimized.optimization_id} | "
            f"Tokens: {optimized.token_estimate} | "
            f"Similarity: {validation.semantic_similarity:.3f} | "
            f"Compression: {optimized.compression_ratio:.2f}x*"
        )
