"""
TokenBudgetEngine — Adaptive, token-safe context expansion budgeting.

Manages the total token envelope available for context injection.
Provides:
  - Token estimation
  - Dependency cost prediction
  - Dynamic expansion budgeting
  - Context entropy scoring (diminishing return detection)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from llm_prompt_optimizer.config.settings import TokenBudgetConfig
from llm_prompt_optimizer.models.context import ContextSpan
from llm_prompt_optimizer.utils.helpers import estimate_tokens, get_logger

logger = get_logger(__name__)


@dataclass
class BudgetState:
    """Live budget snapshot during expansion."""
    total_budget: int
    reserved: int
    used: int = 0
    iterations: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.total_budget - self.reserved - self.used)

    @property
    def utilization_pct(self) -> float:
        return (self.used / max(1, self.total_budget - self.reserved)) * 100

    def consume(self, tokens: int) -> bool:
        """Try to consume tokens. Returns False if budget exhausted."""
        if tokens > self.remaining:
            return False
        self.used += tokens
        return True


class TokenBudgetEngine:
    """
    Adaptive token budget controller for context expansion.

    Does NOT use fixed file counts or depth limits.
    Uses value-based budgeting with entropy-based stop conditions.
    """

    def __init__(self, config: Optional[TokenBudgetConfig] = None) -> None:
        self.config = config or TokenBudgetConfig()

    def create_budget(self, prompt_token_cost: int) -> BudgetState:
        """Create a fresh budget state for an optimization run."""
        budget = self.config.default_budget_tokens
        if self.config.adaptive_budgeting:
            # Scale budget relative to prompt size — larger prompts get more room
            scale = min(2.0, max(1.0, prompt_token_cost / 500))
            budget = min(self.config.max_budget_tokens, int(budget * scale))
        budget = max(self.config.min_budget_tokens, budget)
        state = BudgetState(
            total_budget=budget,
            reserved=self.config.reserve_tokens,
        )
        logger.debug(f"Budget created: {budget} tokens (prompt cost: {prompt_token_cost})")
        return state

    def estimate_span_cost(self, span: ContextSpan) -> int:
        """Estimate token cost for a context span."""
        if span.content:
            return estimate_tokens(span.content)
        # Rough estimate: ~3 tokens per line of code
        return span.line_count() * 3

    def should_stop_expansion(
        self,
        state: BudgetState,
        last_value_score: float,
        current_value_score: float,
    ) -> tuple[bool, str]:
        """
        Determine whether adaptive expansion should halt.
        Returns (should_stop, reason).
        """
        if state.remaining <= 0:
            return True, "budget_exhausted"

        gain = last_value_score - current_value_score
        if gain < self.config.entropy_threshold:
            return True, "diminishing_returns"

        if state.utilization_pct > 90:
            return True, "high_utilization"

        return False, ""

    def score_span_value(self, span: ContextSpan) -> float:
        """
        Value score for a span:
          (relevance × confidence × execution_proximity) / token_cost
        """
        cost = max(1, self.estimate_span_cost(span))
        return (span.relevance_score * span.confidence * span.execution_proximity) / cost

    def rank_spans(self, spans: List[ContextSpan]) -> List[ContextSpan]:
        """Rank spans by value score descending."""
        return sorted(spans, key=self.score_span_value, reverse=True)

    def fit_spans_to_budget(
        self, spans: List[ContextSpan], state: BudgetState
    ) -> List[ContextSpan]:
        """
        Greedily pack highest-value spans into remaining budget.
        Never truncates a span — either it fits or it doesn't.
        """
        ranked = self.rank_spans(spans)
        selected: List[ContextSpan] = []
        for span in ranked:
            cost = self.estimate_span_cost(span)
            span.token_cost = cost
            if state.consume(cost):
                selected.append(span)
            else:
                break
        return selected
