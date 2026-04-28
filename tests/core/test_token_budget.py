"""Tests for TokenBudgetEngine."""

import pytest
from llm_prompt_optimizer.core.token_budget.engine import TokenBudgetEngine, BudgetState
from llm_prompt_optimizer.models.context import ContextSpan
from llm_prompt_optimizer.config.settings import TokenBudgetConfig


@pytest.fixture
def engine():
    return TokenBudgetEngine()


def make_span(file_path="test.py", lines=20, confidence=0.9, relevance=0.8, proximity=1.0):
    content = "\n".join([f"line {i}" for i in range(lines)])
    from llm_prompt_optimizer.utils.helpers import estimate_tokens
    return ContextSpan(
        file_path=file_path,
        start_line=1,
        end_line=lines,
        confidence=confidence,
        relevance_score=relevance,
        execution_proximity=proximity,
        token_cost=estimate_tokens(content),
        content=content,
    )


def test_budget_created(engine):
    budget = engine.create_budget(prompt_token_cost=200)
    assert budget.total_budget > 0
    assert budget.remaining > 0


def test_budget_scales_with_prompt_size(engine):
    small = engine.create_budget(100)
    large = engine.create_budget(1000)
    assert large.total_budget >= small.total_budget


def test_consume_reduces_remaining(engine):
    budget = engine.create_budget(200)
    initial = budget.remaining
    budget.consume(100)
    assert budget.remaining == initial - 100


def test_consume_returns_false_when_exhausted(engine):
    budget = engine.create_budget(200)
    cfg = TokenBudgetConfig(min_budget_tokens=100, reserve_tokens=0)
    engine2 = TokenBudgetEngine(config=cfg)
    budget2 = engine2.create_budget(50)
    budget2.used = budget2.total_budget  # exhaust it
    result = budget2.consume(1)
    assert result is False


def test_value_score_computed(engine):
    span = make_span(confidence=0.9, relevance=0.8, proximity=1.0, lines=20)
    score = engine.score_span_value(span)
    assert score > 0


def test_higher_confidence_higher_score(engine):
    low = make_span(confidence=0.3, relevance=0.5, proximity=0.5, lines=20)
    high = make_span(confidence=0.95, relevance=0.95, proximity=1.0, lines=20)
    assert engine.score_span_value(high) > engine.score_span_value(low)


def test_rank_spans_descending(engine):
    spans = [
        make_span(confidence=0.5, relevance=0.5, proximity=0.5),
        make_span(confidence=0.95, relevance=0.95, proximity=1.0),
        make_span(confidence=0.7, relevance=0.7, proximity=0.8),
    ]
    ranked = engine.rank_spans(spans)
    scores = [engine.score_span_value(s) for s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_fit_spans_respects_budget(engine):
    cfg = TokenBudgetConfig(default_budget_tokens=200, reserve_tokens=0)
    engine2 = TokenBudgetEngine(config=cfg)
    budget = engine2.create_budget(50)
    spans = [make_span(lines=100) for _ in range(10)]  # each ~300 tokens
    fitted = engine2.fit_spans_to_budget(spans, budget)
    # Should fit very few due to small budget
    assert len(fitted) < 10


def test_stop_expansion_on_exhaustion(engine):
    budget = BudgetState(total_budget=100, reserved=0, used=100)
    stop, reason = engine.should_stop_expansion(budget, 0.8, 0.7)
    assert stop is True
    assert reason == "budget_exhausted"


def test_stop_expansion_on_diminishing_returns(engine):
    budget = BudgetState(total_budget=8000, reserved=200, used=100)
    stop, reason = engine.should_stop_expansion(budget, 0.501, 0.500)
    assert stop is True
    assert reason == "diminishing_returns"
