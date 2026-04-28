"""Tests for PromptClassifier."""

import pytest
from llm_prompt_optimizer.core.classifier.classifier import PromptClassifier
from llm_prompt_optimizer.core.intent_guard.guard import IntentGuard
from llm_prompt_optimizer.models.classification import TaskCategory
from llm_prompt_optimizer.models.prompt import RawPrompt


@pytest.fixture
def classifier():
    return PromptClassifier()


@pytest.fixture
def guard():
    return IntentGuard()


def _classify(classifier, guard, text):
    prompt = RawPrompt(text=text)
    lock = guard.extract_and_lock(prompt)
    return classifier.classify(prompt, lock)


def test_debug_classification(classifier, guard):
    cls = _classify(classifier, guard, "Debug EMA mismatch in signals/IndexSignals.py")
    assert cls.primary_category == TaskCategory.DEBUGGING


def test_implementation_classification(classifier, guard):
    cls = _classify(classifier, guard, "Implement a /health endpoint in api/routes.py")
    assert cls.primary_category == TaskCategory.IMPLEMENTATION


def test_refactoring_classification(classifier, guard):
    cls = _classify(classifier, guard, "Refactor utils/helpers.py to remove duplicates")
    assert cls.primary_category == TaskCategory.REFACTORING


def test_testing_classification(classifier, guard):
    cls = _classify(classifier, guard, "Write pytest tests for the auth module")
    assert cls.primary_category == TaskCategory.TESTING


def test_stacktrace_boosts_debugging(classifier, guard):
    text = (
        "Traceback (most recent call last):\n"
        "  File 'app.py', line 10, in run\n"
        "ValueError: bad input\n"
        "Why is this happening?"
    )
    cls = _classify(classifier, guard, text)
    assert cls.primary_category == TaskCategory.DEBUGGING
    assert cls.has_stacktrace is True


def test_log_detection(classifier, guard):
    text = "2024-01-15 ERROR failed to connect\nINFO starting server\nDEBUG retry 3"
    cls = _classify(classifier, guard, text)
    assert cls.has_logs is True


def test_code_snippet_detection(classifier, guard):
    text = "```python\ndef foo():\n    return 1\n```\nWhat does this function do?"
    cls = _classify(classifier, guard, text)
    assert cls.has_code_snippet is True


def test_file_path_extraction(classifier, guard):
    text = "The bug is in signals/IndexSignals.py in the calculate_ema function"
    cls = _classify(classifier, guard, text)
    assert any("IndexSignals" in f for f in cls.extracted_file_paths)


def test_complexity_increases_with_length(classifier, guard):
    short_cls = _classify(classifier, guard, "Fix bug")
    long_cls = _classify(
        classifier, guard,
        "Debug the EMA mismatch in signals/IndexSignals.py. "
        "The issue occurs when the window size doesn't match. "
        "Check utils/ema.py calculate_ema function for the root cause. "
        "Also look at the data/loader.py for the input pipeline. "
        "No code changes needed, just explain the issue."
    )
    assert long_cls.complexity_score >= short_cls.complexity_score


def test_multi_label_output(classifier, guard):
    cls = _classify(classifier, guard, "Debug and optimize the slow query in db/queries.py")
    assert len(cls.labels) > 1


def test_primary_confidence_range(classifier, guard):
    cls = _classify(classifier, guard, "Debug the authentication bug")
    assert 0.0 <= cls.primary_confidence <= 1.0


def test_ml_ai_classification(classifier, guard):
    cls = _classify(
        classifier, guard,
        "The model loss is not converging after epoch 10. Check training/trainer.py"
    )
    assert cls.primary_category in (TaskCategory.ML_AI, TaskCategory.DEBUGGING)
