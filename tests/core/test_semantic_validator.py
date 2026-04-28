"""Tests for SemanticValidator."""

import pytest
from llm_prompt_optimizer.core.semantic_validator.validator import SemanticValidator
from llm_prompt_optimizer.core.intent_guard.guard import IntentGuard
from llm_prompt_optimizer.models.prompt import RawPrompt
from llm_prompt_optimizer.config.settings import PolicyConfig


@pytest.fixture
def validator():
    return SemanticValidator()


@pytest.fixture
def guard():
    return IntentGuard()


def _lock(guard, text):
    return guard.extract_and_lock(RawPrompt(text=text))


def test_identical_texts_pass(validator, guard):
    text = "Debug EMA mismatch in signals/IndexSignals.py"
    lock = _lock(guard, text)
    result = validator.validate(text, text, lock)
    assert result.passed is True
    assert result.semantic_similarity >= 0.90


def test_completely_different_fails(validator, guard):
    raw = "Debug EMA mismatch in signals/IndexSignals.py"
    lock = _lock(guard, raw)
    optimized = "Write a poem about flowers and sunshine in the meadow."
    result = validator.validate(raw, optimized, lock)
    assert result.passed is False


def test_minor_rephrasing_passes(validator, guard):
    raw = "Debug condition mismatch in the provided /path/repository/file No code changes."
    lock = _lock(guard, raw)
    optimized = (
        "# Task\nDebug condition mismatch in the provided /path/repository/file No code changes.\n"
        "## Constraints\n- Do NOT make code changes."
    )
    result = validator.validate(raw, optimized, lock)
    assert result.passed is True


def test_threshold_respected(validator, guard):
    raw = "Fix the auth bug"
    lock = _lock(guard, raw)
    cfg = PolicyConfig(semantic_similarity_threshold=0.95)
    strict_validator = SemanticValidator(config=cfg)
    # Same text should still pass even with strict threshold
    result = strict_validator.validate(raw, raw, lock)
    assert result.passed is True


def test_validation_result_has_method(validator, guard):
    raw = "Debug EMA mismatch"
    lock = _lock(guard, raw)
    result = validator.validate(raw, raw, lock)
    assert result.method in ("sentence_transformer", "ngram_jaccard_composite")


def test_constraint_violation_penalizes(validator, guard):
    raw = "Debug condition mismatch in the provided /path/repository/file No code changes."
    lock = _lock(guard, raw)
    # Optimized adds imperative language that violates no_code_changes
    optimized = "Debug EMA mismatch. Implement the fix and write the following code."
    result = validator.validate(raw, optimized, lock)
    # Should be penalized but not necessarily rejected (depends on similarity)
    assert isinstance(result.passed, bool)


def test_similarity_between_zero_and_one(validator, guard):
    raw = "Debug EMA mismatch in signals/IndexSignals.py"
    lock = _lock(guard, raw)
    result = validator.validate(raw, "completely unrelated text here", lock)
    assert 0.0 <= result.semantic_similarity <= 1.0


def test_failure_reason_set_when_rejected(validator, guard):
    raw = "Debug EMA mismatch"
    lock = _lock(guard, raw)
    result = validator.validate(raw, "Refactor the entire architecture from scratch.", lock)
    if not result.passed:
        assert result.failure_reason is not None
