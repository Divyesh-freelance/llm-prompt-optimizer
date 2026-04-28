"""Tests for IntentGuard."""

import pytest
from llm_prompt_optimizer.core.intent_guard.guard import IntentGuard
from llm_prompt_optimizer.models.prompt import RawPrompt
from llm_prompt_optimizer.models.intent import ExpansionMode


@pytest.fixture
def guard():
    return IntentGuard()


def test_extracts_file_paths(guard):
    prompt = RawPrompt(text="Debug EMA mismatch in signals/IndexSignals.py")
    lock = guard.extract_and_lock(prompt)
    assert "signals/IndexSignals.py" in lock.target_files


def test_detects_no_code_changes(guard):
    prompt = RawPrompt(text="Debug the issue. No code changes needed.")
    lock = guard.extract_and_lock(prompt)
    types = [c.constraint_type for c in lock.constraints]
    assert "no_code_changes" in types


def test_detects_no_repo_scan(guard):
    prompt = RawPrompt(text="Check only this file, no repo scan.")
    lock = guard.extract_and_lock(prompt)
    types = [c.constraint_type for c in lock.constraints]
    assert "no_repo_scan" in types
    assert lock.allowed_expansion == ExpansionMode.STRICT


def test_lock_is_sealed(guard):
    prompt = RawPrompt(text="Fix the bug in utils/helper.py")
    lock = guard.extract_and_lock(prompt)
    assert lock.locked is True
    with pytest.raises(RuntimeError):
        lock.assert_unlocked()


def test_intent_summary_preserved(guard):
    text = "Debug EMA mismatch in signals/IndexSignals.py"
    prompt = RawPrompt(text=text)
    lock = guard.extract_and_lock(prompt)
    assert "ema" in lock.intent_summary.lower() or "debug" in lock.intent_summary.lower()


def test_expansion_mode_default_controlled(guard):
    prompt = RawPrompt(text="Help me fix the issue in api/routes.py")
    lock = guard.extract_and_lock(prompt)
    assert lock.allowed_expansion in (ExpansionMode.CONTROLLED, ExpansionMode.STRICT)


def test_stacktrace_detected(guard):
    text = (
        "Getting:\nTraceback (most recent call last):\n"
        "  File 'app.py', line 10, in run\nValueError: bad input"
    )
    prompt = RawPrompt(text=text)
    lock = guard.extract_and_lock(prompt)
    types = [c.constraint_type for c in lock.constraints]
    assert "has_stacktrace" in types


def test_confidence_with_file_and_symbol(guard):
    prompt = RawPrompt(text="Debug calculate_ema in utils/ema.py no code changes")
    lock = guard.extract_and_lock(prompt)
    assert lock.confidence > 0.7


def test_empty_prompt_raises(guard):
    with pytest.raises(ValueError):
        RawPrompt(text="")


def test_expansion_open_mode(guard):
    prompt = RawPrompt(text="Explore the entire codebase and find all usages.")
    lock = guard.extract_and_lock(prompt)
    assert lock.allowed_expansion == ExpansionMode.OPEN
