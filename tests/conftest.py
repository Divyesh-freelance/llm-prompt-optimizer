"""Shared pytest fixtures for LLM Prompt Optimizer tests."""

import pytest
from llm_prompt_optimizer import Optimizer, OptimizerConfig
from llm_prompt_optimizer.models.prompt import RawPrompt
from llm_prompt_optimizer.config.settings import PolicyConfig, TokenBudgetConfig


@pytest.fixture(scope="session")
def config():
    cfg = OptimizerConfig()
    cfg.policy.enable_audit_log = False  # disable disk writes during tests
    cfg.policy.enable_telemetry = True
    cfg.token_budget.default_budget_tokens = 4000
    return cfg


@pytest.fixture(scope="session")
def optimizer(config):
    return Optimizer(config=config)


@pytest.fixture
def simple_debug_prompt():
    return RawPrompt(
        text="Debug EMA mismatch in signals/IndexSignals.py. No code changes needed.",
    )


@pytest.fixture
def stacktrace_prompt():
    return RawPrompt(
        text=(
            "Getting this error:\n"
            "Traceback (most recent call last):\n"
            "  File 'signals/IndexSignals.py', line 87, in calculate\n"
            "    ValueError: EMA length mismatch\n"
            "Why is this happening?"
        )
    )


@pytest.fixture
def implementation_prompt():
    return RawPrompt(
        text="Implement a /health endpoint in api/routes.py that returns 200 OK."
    )


@pytest.fixture
def vague_prompt():
    return RawPrompt(text="Fix the bug")
