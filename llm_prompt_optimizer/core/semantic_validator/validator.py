"""
SemanticValidator — Compares raw vs optimized prompt for semantic preservation.

RULE: Reject if semantic_similarity < 0.90

Core insight: the optimized prompt always ADDS structure (headers, constraints,
context blocks, metadata). A Jaccard or F1 bigram score unfairly penalises this
because the output denominator grows. We use RECALL metrics throughout:

  - Verbatim check: raw text appears verbatim in output → automatic 1.0
  - Token recall  : fraction of raw key-tokens present in output
  - Bigram recall : fraction of raw bigrams present in output (not Jaccard)

Optional: sentence-transformers for true embedding similarity when available.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from llm_prompt_optimizer.config.settings import PolicyConfig
from llm_prompt_optimizer.models.intent import IntentLock
from llm_prompt_optimizer.models.validation import ValidationResult
from llm_prompt_optimizer.utils.helpers import get_logger, cosine_similarity_simple

logger = get_logger(__name__)

_STOP_WORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "it", "this", "that", "be", "was", "are", "i", "we", "you",
    "my", "me", "its", "do", "not", "as", "by", "with", "from", "up",
}


def _ngram_recall(text_a: str, text_b: str, n: int = 2) -> float:
    """
    N-gram RECALL: fraction of text_a's n-grams that appear in text_b.

    Uses recall not Jaccard — optimized prompts always add structure tokens,
    so Jaccard would unfairly penalise a correct output that is longer than input.
    """
    def ngrams(tokens: List[str], n: int) -> set:
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()

    if len(tokens_a) < n:
        return cosine_similarity_simple(text_a, text_b)
    if len(tokens_b) < n:
        return 0.0

    grams_a = ngrams(tokens_a, n)
    grams_b = ngrams(tokens_b, n)

    if not grams_a:
        return 1.0

    return len(grams_a & grams_b) / len(grams_a)


# ── SentenceTransformer model singleton ─────────────────────────────────────
#
# The previous implementation reloaded the ~90 MB MiniLM model on every
# `validate()` call, which made `optimize_prompt` 5–30 s slower per invocation
# and was the dominant latency under MCP-default-on usage. We now lazy-load
# the model exactly once per process and reuse it across all validators.
#
# Failure modes handled:
#   • ImportError (sentence-transformers not installed)  → fall back to n-gram
#   • Any runtime error during model load (OOM, network) → fall back to n-gram
# After a hard failure we cache the failure so subsequent calls don't retry.

_EMBEDDING_MODEL = None        # SentenceTransformer instance, once loaded
_EMBEDDING_UTIL = None         # sentence_transformers.util module
_EMBEDDING_DISABLED = False    # set True after a permanent failure


def _get_embedding_model():
    """Lazy-load the embedding model exactly once per process; reuse forever."""
    global _EMBEDDING_MODEL, _EMBEDDING_UTIL, _EMBEDDING_DISABLED
    if _EMBEDDING_DISABLED:
        return None, None
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL, _EMBEDDING_UTIL
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _EMBEDDING_UTIL = util
        logger.info("SentenceTransformer model loaded (cached for process lifetime)")
        return _EMBEDDING_MODEL, _EMBEDDING_UTIL
    except ImportError:
        # Soft-disable: package not installed → use fast n-gram fallback.
        _EMBEDDING_DISABLED = True
        return None, None
    except Exception as e:
        # Hard-disable: model download failed, OOM, etc. Don't retry.
        logger.warning(f"SentenceTransformer load failed ({e!r}); using n-gram fallback")
        _EMBEDDING_DISABLED = True
        return None, None


def _try_embedding_similarity(text_a: str, text_b: str) -> Optional[float]:
    """
    Attempt sentence-transformer similarity using a process-wide cached model.
    Returns None if the model is unavailable, in which case callers should
    fall back to the n-gram recall path.
    """
    model, util = _get_embedding_model()
    if model is None:
        return None
    try:
        emb_a = model.encode(text_a, convert_to_tensor=True)
        emb_b = model.encode(text_b, convert_to_tensor=True)
        return float(util.cos_sim(emb_a, emb_b)[0][0])
    except Exception as e:
        # Per-call failure (e.g. malformed input) — don't poison the cache.
        logger.warning(f"Embedding similarity failed ({e!r}); using n-gram fallback")
        return None


def reset_embedding_cache() -> None:
    """Test hook: drop the cached model so the next call reloads it."""
    global _EMBEDDING_MODEL, _EMBEDDING_UTIL, _EMBEDDING_DISABLED
    _EMBEDDING_MODEL = None
    _EMBEDDING_UTIL = None
    _EMBEDDING_DISABLED = False


def _strip_markdown_structure(text: str) -> str:
    """Remove injected markdown structure to expose semantic content."""
    text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"^[-─]{3,}.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*Optimization ID:.*\*", "", text)
    return re.sub(r"\s+", " ", text).strip()


class SemanticValidator:
    """
    Validates that an optimized prompt semantically preserves the raw prompt.

    Scoring (no sentence-transformers):
      1. Verbatim fast-path  — raw text found verbatim in output  → 1.0
      2. Token recall        — raw key-tokens present in output   (weight 0.65)
      3. Bigram recall       — raw bigrams present in output      (weight 0.35)

    Penalties applied after scoring:
      - Intent keywords lost  → −0.08
      - Constraints removed   → −0.05
    """

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        self.config = config or PolicyConfig()
        self.threshold = self.config.semantic_similarity_threshold

    def validate(
        self,
        raw_text: str,
        optimized_text: str,
        intent_lock: IntentLock,
        fast_mode: bool = False,
    ) -> ValidationResult:
        similarity, method = self._compute_similarity(
            raw_text, optimized_text, fast_mode=fast_mode
        )

        intent_preserved = self._check_intent_preservation(
            raw_text, optimized_text, intent_lock
        )
        constraints_preserved = self._check_constraints_preserved(
            raw_text, optimized_text, intent_lock
        )

        adjusted = similarity
        if not intent_preserved:
            adjusted -= 0.08
            logger.warning("Intent keywords lost — penalising similarity")
        if not constraints_preserved:
            adjusted -= 0.05
            logger.warning("Constraints lost — penalising similarity")

        adjusted = round(max(0.0, min(1.0, adjusted)), 4)
        passed = adjusted >= self.threshold
        failure_reason = (
            None
            if passed
            else f"Semantic similarity {adjusted:.3f} below threshold {self.threshold:.2f}"
        )

        result = ValidationResult(
            passed=passed,
            semantic_similarity=adjusted,
            threshold=self.threshold,
            method=method,
            failure_reason=failure_reason,
            details={
                "raw_similarity": similarity,
                "intent_preserved": intent_preserved,
                "constraints_preserved": constraints_preserved,
                "adjusted_similarity": adjusted,
            },
        )
        logger.info(
            f"Semantic validation: passed={passed} sim={adjusted:.3f} method={method}"
        )
        return result

    def _compute_similarity(
        self, text_a: str, text_b: str, fast_mode: bool = False
    ) -> Tuple[float, str]:
        # Best path: sentence-transformer embeddings — but skip in fast_mode,
        # which is used for short prompts where the n-gram path is plenty.
        if not fast_mode:
            emb = _try_embedding_similarity(text_a, text_b)
            if emb is not None:
                return emb, "sentence_transformer"

        clean_b = _strip_markdown_structure(text_b)

        # Fast path: raw text is verbatim inside the output
        if text_a.lower().strip() in clean_b.lower():
            return 1.0, "verbatim_match"

        # Token recall: fraction of non-trivial raw tokens that survived
        raw_tokens = set(text_a.lower().split())
        key_tokens = raw_tokens - _STOP_WORDS
        if not key_tokens:
            return 1.0, "empty_key_tokens"

        out_tokens_clean = set(clean_b.lower().split())
        out_tokens_raw = set(text_b.lower().split())
        token_recall = max(
            len(key_tokens & out_tokens_clean) / len(key_tokens),
            len(key_tokens & out_tokens_raw) / len(key_tokens),
        )

        # Bigram recall: fraction of raw bigrams found in the output
        bigram_recall = _ngram_recall(text_a, clean_b, n=2)

        combined = token_recall * 0.65 + bigram_recall * 0.35
        return round(min(1.0, combined), 4), "recall_composite"

    def _check_intent_preservation(
        self, raw: str, optimized: str, intent_lock: IntentLock
    ) -> bool:
        summary_words = set(intent_lock.intent_summary.lower().split())
        key_words = summary_words - _STOP_WORDS
        if not key_words:
            return True
        opt_lower = optimized.lower()
        preserved = sum(1 for w in key_words if w in opt_lower)
        return (preserved / len(key_words)) >= 0.70

    def _check_constraints_preserved(
        self, raw: str, optimized: str, intent_lock: IntentLock
    ) -> bool:
        opt_lower = optimized.lower()
        for c in intent_lock.constraints:
            if c.constraint_type == "no_code_changes":
                if any(w in opt_lower for w in ["implement", "add the following", "write this"]):
                    return False
        return True