"""Integration tests for the full optimization pipeline."""

import pytest
from llm_prompt_optimizer import Optimizer, OptimizerConfig
from llm_prompt_optimizer.models.prompt import RawPrompt
from llm_prompt_optimizer.models.classification import TaskCategory
from llm_prompt_optimizer.models.intent import ExpansionMode


@pytest.fixture(scope="module")
def optimizer():
    cfg = OptimizerConfig()
    cfg.policy.enable_audit_log = False
    return Optimizer(config=cfg)


class TestFullPipeline:

    def test_simple_debug_prompt_succeeds(self, optimizer):
        result = optimizer.optimize("Debug EMA mismatch in signals/IndexSignals.py")
        assert result.optimized_prompt is not None
        assert result.optimized_prompt.text != ""

    def test_result_has_intent_lock(self, optimizer):
        result = optimizer.optimize("Debug EMA mismatch in signals/IndexSignals.py")
        assert result.intent_lock is not None
        assert result.intent_lock.locked is True

    def test_result_has_classification(self, optimizer):
        result = optimizer.optimize("Debug EMA mismatch in signals/IndexSignals.py")
        assert result.classification is not None
        assert result.classification.primary_category == TaskCategory.DEBUGGING

    def test_result_has_validation(self, optimizer):
        result = optimizer.optimize("Debug EMA mismatch in signals/IndexSignals.py")
        assert result.validation_result is not None
        assert isinstance(result.validation_result.passed, bool)

    def test_result_has_drift_report(self, optimizer):
        result = optimizer.optimize("Debug EMA mismatch in signals/IndexSignals.py")
        assert result.drift_report is not None

    def test_result_has_telemetry(self, optimizer):
        result = optimizer.optimize("Debug EMA mismatch in signals/IndexSignals.py")
        assert result.telemetry is not None
        assert result.telemetry.original_token_estimate > 0

    def test_optimized_text_not_empty(self, optimizer):
        result = optimizer.optimize("Fix the authentication bug in auth/handler.py")
        assert len(result.optimized_prompt.text) > 0

    def test_strict_mode_sets_strict_expansion(self, optimizer):
        result = optimizer.optimize(
            "Debug EMA mismatch in signals/IndexSignals.py",
            strict_mode=True,
        )
        assert result.intent_lock.allowed_expansion == ExpansionMode.STRICT

    def test_no_code_changes_constraint_preserved(self, optimizer):
        result = optimizer.optimize(
            "Debug condition mismatch in the provided /path/repository/file No code changes."
        )
        lock = result.intent_lock
        types = [c.constraint_type for c in lock.constraints]
        assert "no_code_changes" in types

    def test_pipeline_duration_recorded(self, optimizer):
        result = optimizer.optimize("Fix the bug in utils/helper.py")
        assert result.pipeline_duration_ms > 0

    def test_token_estimate_positive(self, optimizer):
        result = optimizer.optimize("Debug EMA mismatch in signals/IndexSignals.py")
        assert result.optimized_prompt.token_estimate > 0

    def test_classify_standalone(self, optimizer):
        cls = optimizer.classify("Implement a new /health endpoint in api/routes.py")
        assert cls.primary_category == TaskCategory.IMPLEMENTATION

    def test_validate_standalone(self, optimizer):
        raw = "Debug EMA mismatch in signals/IndexSignals.py"
        result = optimizer.validate(raw, raw)
        assert result.passed is True
        assert result.semantic_similarity >= 0.9

    def test_detect_drift_standalone(self, optimizer):
        raw = "Debug EMA mismatch"
        report = optimizer.detect_drift(raw, raw)
        assert report.is_clean is True

    def test_estimate_cost(self, optimizer):
        cost = optimizer.estimate_cost("Debug EMA mismatch in signals/IndexSignals.py")
        assert cost["estimated_tokens"] > 0
        assert cost["approx_chars"] > 0

    def test_stacktrace_prompt_classified_as_debugging(self, optimizer):
        text = (
            "Traceback (most recent call last):\n"
            "  File 'signals/IndexSignals.py', line 87, in calculate\n"
            "ValueError: EMA length mismatch\nWhy?"
        )
        result = optimizer.optimize(text)
        assert result.classification.primary_category == TaskCategory.DEBUGGING
        assert result.classification.has_stacktrace is True

    def test_vague_prompt_does_not_crash(self, optimizer):
        result = optimizer.optimize("Fix the bug")
        assert result.optimized_prompt is not None
        # May succeed or fail gracefully — must not raise
        assert isinstance(result.success, bool)

    def test_implementation_prompt_classified_correctly(self, optimizer):
        result = optimizer.optimize(
            "Implement a POST /users endpoint in api/users.py that creates a new user"
        )
        assert result.classification.primary_category == TaskCategory.IMPLEMENTATION

    def test_optimized_includes_category_header(self, optimizer):
        result = optimizer.optimize("Debug EMA mismatch in signals/IndexSignals.py")
        text = result.optimized_prompt.text.lower()
        assert "debug" in text or "optimization" in text

    def test_semantic_similarity_within_range(self, optimizer):
        result = optimizer.optimize("Debug EMA mismatch in signals/IndexSignals.py")
        sim = result.optimized_prompt.semantic_similarity
        assert 0.0 <= sim <= 1.0

    def test_policy_violations_is_list(self, optimizer):
        result = optimizer.optimize("Debug EMA mismatch")
        assert isinstance(result.policy_violations, list)

    def test_list_plugins_returns_list(self, optimizer):
        plugins = optimizer.list_plugins()
        assert isinstance(plugins, list)
