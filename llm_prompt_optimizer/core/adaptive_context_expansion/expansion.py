"""
AdaptiveContextExpansion — Value-based, token-aware dependency traversal.

PRINCIPLE: Context Value > File Count

Uses the formula:
  context_value_score = (relevance × confidence × execution_proximity) / token_cost

Stop conditions (NOT fixed limits):
  - confidence gain too low
  - token cost exceeds value
  - execution relevance weakens
  - semantic confidence decreases
  - marginal debugging value drops
  - budget exhausted
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from llm_prompt_optimizer.config.settings import AdaptiveExpansionConfig
from llm_prompt_optimizer.core.fallback_graph.engine import FallbackGraphEngine
from llm_prompt_optimizer.core.precise_context.resolver import PreciseContextResolver
from llm_prompt_optimizer.core.token_budget.engine import TokenBudgetEngine, BudgetState
from llm_prompt_optimizer.models.context import ContextBundle, ContextSpan, DependencyNode
from llm_prompt_optimizer.models.intent import IntentLock, ExpansionMode
from llm_prompt_optimizer.models.classification import PromptClassification
from llm_prompt_optimizer.utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class ExpansionIteration:
    """Record of a single expansion iteration."""
    iteration: int
    node: DependencyNode
    spans_added: List[ContextSpan]
    value_score: float
    tokens_consumed: int
    stop_reason: Optional[str] = None


class AdaptiveContextExpansion:
    """
    Adaptive context expansion engine.

    Traverses dependency graph iteratively, scoring each candidate node
    against a value formula. Stops when marginal value drops below threshold
    or budget is exhausted — NOT at a fixed file/depth limit.
    """

    def __init__(
        self,
        config: Optional[AdaptiveExpansionConfig] = None,
        token_engine: Optional[TokenBudgetEngine] = None,
        graph_engine: Optional[FallbackGraphEngine] = None,
        context_resolver: Optional[PreciseContextResolver] = None,
    ) -> None:
        self.config = config or AdaptiveExpansionConfig()
        self.token_engine = token_engine or TokenBudgetEngine()
        # graph_engine is now an optional last-resort fallback. The expected
        # path is for the caller (Optimizer) to pass `candidates=...` it
        # already discovered via DependencyResolver — which honors the
        # external-graph-provider priority chain. We only fall back to a
        # local AST walk here if NO candidates were given AND a graph_engine
        # was provided. This avoids the previous bug where stage 7
        # re-walked the AST even when Graphify / Code Review Graph had
        # already produced the answer in stage 5.
        self.graph_engine = graph_engine
        self.context_resolver = context_resolver or PreciseContextResolver()

    def expand(
        self,
        intent_lock: IntentLock,
        classification: PromptClassification,
        initial_spans: List[ContextSpan],
        budget_state: BudgetState,
        repo_root: Optional[str] = None,
        candidates: Optional[List[DependencyNode]] = None,
    ) -> ContextBundle:
        """
        Expand context adaptively from the initial seed spans.

        Args:
            candidates: Pre-discovered dependency nodes from DependencyResolver.
                When provided, we skip our own discovery entirely — important
                for performance because (a) it avoids walking the AST a
                second time after stage 5 already did the work, and (b) it
                lets us honor the external graph provider priority chain.
                When None and a graph_engine was injected, we fall back to a
                local AST walk for backwards compatibility.

        Returns a ContextBundle with all selected spans and dependency metadata.
        """
        bundle = ContextBundle()

        # Seed: always include initial direct spans
        for span in initial_spans:
            cost = self.token_engine.estimate_span_cost(span)
            span.token_cost = cost
            if budget_state.consume(cost):
                bundle.add_span(span)
            else:
                bundle.expansion_stopped_reason = "budget_exhausted_on_seed"
                return bundle

        # If strict mode, stop here — no expansion
        if intent_lock.allowed_expansion == ExpansionMode.STRICT:
            bundle.expansion_stopped_reason = "strict_mode"
            logger.info("Strict expansion mode — no dependency traversal")
            return bundle

        # Source candidates from caller-provided list first; fall back to a
        # local AST discovery only if the caller didn't supply any AND a
        # graph engine was injected.
        if candidates is None:
            if not self.graph_engine:
                bundle.expansion_stopped_reason = "no_candidates_no_graph_engine"
                return bundle
            logger.info("No candidates passed — falling back to local AST walk")
            candidates = self.graph_engine.discover_dependencies(intent_lock)
        else:
            logger.debug(
                f"Using {len(candidates)} pre-discovered candidates "
                "(no redundant graph walk)"
            )

        # Score and rank candidates
        scored = self._score_candidates(candidates, intent_lock, classification)

        # Iterative expansion
        last_value = float("inf")
        iteration = 0

        for node, node_score in scored:
            iteration += 1
            budget_state.iterations = iteration

            # Stop condition check
            stop, reason = self.token_engine.should_stop_expansion(
                budget_state, last_value, node_score
            )
            if stop:
                bundle.expansion_stopped_reason = reason
                logger.info(f"Expansion stopped at iteration {iteration}: {reason}")
                break

            # Confidence gain check
            gain = last_value - node_score
            if gain < self.config.confidence_gain_threshold and iteration > 1:
                bundle.expansion_stopped_reason = "confidence_gain_below_threshold"
                break

            # Execution relevance check
            if node_score < self.config.execution_relevance_min:
                bundle.expansion_stopped_reason = "execution_relevance_too_low"
                break

            # Resolve precise spans from this node
            node_spans = self.context_resolver.resolve_span_for_dependency(
                node.file_path,
                intent_lock.target_symbols,
            )

            if not node_spans and node.spans:
                node_spans = node.spans

            # Fit spans to remaining budget
            fitted = self.token_engine.fit_spans_to_budget(node_spans, budget_state)
            for span in fitted:
                bundle.add_span(span)
                bundle.dependencies.append(node)

            last_value = node_score

            logger.debug(
                f"Iteration {iteration}: node={node.file_path} "
                f"score={node_score:.3f} spans={len(fitted)} "
                f"budget_remaining={budget_state.remaining}"
            )

            # Safety cap — not a semantic limit, just prevents infinite loops
            if iteration >= self.config.max_iterations:
                bundle.expansion_stopped_reason = "max_iterations_safety_cap"
                break

        bundle.confidence = self._compute_bundle_confidence(bundle)
        logger.info(
            f"Expansion complete: {len(bundle.spans)} spans, "
            f"{len(bundle.files_included)} files, "
            f"stop_reason={bundle.expansion_stopped_reason}"
        )
        return bundle

    def _score_candidates(
        self,
        nodes: List[DependencyNode],
        intent_lock: IntentLock,
        classification: PromptClassification,
    ) -> List[Tuple[DependencyNode, float]]:
        """
        Score dependency nodes using value formula:
          score = (relevance × confidence × execution_proximity) / estimated_token_cost
        """
        weights = self.config.value_formula_weights
        scored: List[Tuple[DependencyNode, float]] = []

        for node in nodes:
            # Compute execution proximity — nodes closer to root file are more proximate
            exec_proximity = max(0.1, 1.0 - (node.depth * 0.25))

            # Relevance: higher for symbol/call nodes, lower for git/heuristic
            relevance_map = {
                "import": 0.80,
                "call": 0.90,
                "symbol": 0.85,
                "git": 0.45,
                "heuristic": 0.35,
                "runtime": 0.70,
            }
            relevance = relevance_map.get(node.dependency_type, 0.60)

            # Boost if node file matches any classification signal
            for sig_file in classification.extracted_file_paths:
                if sig_file in node.file_path or node.file_path in sig_file:
                    relevance = min(1.0, relevance + 0.15)

            # Estimated token cost of the node (rough)
            est_lines = 50  # default
            if node.spans:
                est_lines = sum(s.line_count() for s in node.spans)
            est_cost = max(1, est_lines * 3)

            score = (
                weights.get("relevance", 1.0) * relevance
                * weights.get("confidence", 1.0) * node.confidence
                * weights.get("execution_proximity", 1.2) * exec_proximity
            ) / est_cost

            scored.append((node, score))

        # Sort by score descending
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def _compute_bundle_confidence(self, bundle: ContextBundle) -> float:
        if not bundle.spans:
            return 0.0
        avg = sum(s.confidence for s in bundle.spans) / len(bundle.spans)
        return round(avg, 4)
