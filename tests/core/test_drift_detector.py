"""Tests for DriftDetector."""

import pytest
from llm_prompt_optimizer.core.drift_detection.detector import DriftDetector
from llm_prompt_optimizer.core.intent_guard.guard import IntentGuard
from llm_prompt_optimizer.models.prompt import RawPrompt


@pytest.fixture
def detector():
    return DriftDetector()


@pytest.fixture
def guard():
    return IntentGuard()


def _lock(guard, text):
    return guard.extract_and_lock(RawPrompt(text=text))


def test_clean_prompt_no_drift(detector, guard):
    raw = "Debug EMA mismatch in signals/IndexSignals.py"
    lock = _lock(guard, raw)
    report = detector.detect(raw, raw, lock)
    assert report.is_clean is True
    assert report.overall_severity == "none"


def test_scope_widening_detected(detector, guard):
    raw = "Debug EMA mismatch in signals/IndexSignals.py"
    lock = _lock(guard, raw)
    optimized = (
        "Debug EMA mismatch in signals/IndexSignals.py. "
        "Also update utils/ema.py and data/loader.py while we're at it."
    )
    report = detector.detect(raw, optimized, lock)
    assert not report.is_clean
    types = [d.drift_type for d in report.drifts_detected]
    assert "scope_widening" in types


def test_added_task_detected(detector, guard):
    raw = "Debug EMA mismatch in signals/IndexSignals.py"
    lock = _lock(guard, raw)
    optimized = (
        "Debug EMA mismatch. Additionally implement a new caching layer."
    )
    report = detector.detect(raw, optimized, lock)
    assert not report.is_clean


def test_missing_constraint_detected(detector, guard):
    raw = "Debug EMA mismatch in signals/IndexSignals.py. No code changes needed."
    lock = _lock(guard, raw)
    # Optimized removes constraint, adds imperative language
    optimized = (
        "Debug EMA mismatch in signals/IndexSignals.py. "
        "Implement the fix using the following code."
    )
    report = detector.detect(raw, optimized, lock)
    # constraint violation should be detected
    types = [d.drift_type for d in report.drifts_detected]
    assert "missing_constraint" in types


def test_critical_drift_blocks(detector, guard):
    raw = "Debug the issue"
    lock = _lock(guard, raw)
    # Simulate a forbidden file scenario
    lock2 = _lock(guard, raw)
    object.__setattr__(lock2, "locked", False)
    lock2.forbidden_files = ["utils/secret.py"]
    lock2.lock()
    from llm_prompt_optimizer.models.context import ContextBundle, ContextSpan
    bundle = ContextBundle()
    bundle.add_span(ContextSpan(
        file_path="utils/secret.py",
        start_line=1,
        end_line=10,
        token_cost=50,
    ))
    report = detector.detect(raw, raw, lock2, context_bundle=bundle)
    assert report.blocked is True


def test_severity_escalation(detector, guard):
    raw = "Debug condition mismatch in the provided /path/repository/file No code changes."
    lock = _lock(guard, raw)
    optimized = (
        "Debug EMA mismatch. Also scan the repo. "
        "Additionally implement the fix."
    )
    report = detector.detect(raw, optimized, lock)
    severity_rank = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    assert severity_rank[report.overall_severity] >= severity_rank["medium"]


def test_report_has_suggested_fix(detector, guard):
    raw = "Debug EMA mismatch in signals/IndexSignals.py"
    lock = _lock(guard, raw)
    optimized = (
        "Debug EMA mismatch in signals/IndexSignals.py and other_module/x.py. "
        "Also update utils/ema.py while we're at it."
    )
    report = detector.detect(raw, optimized, lock)
    for drift in report.drifts_detected:
        if drift.drift_type == "scope_widening":
            assert drift.suggested_fix is not None
