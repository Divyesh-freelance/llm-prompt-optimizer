"""
DependencyResolver — Orchestrates dependency discovery with graceful enhancement.

Priority order:
  1. Optional Graph MCP (graphify / code_review_graph adapter)
  2. Local AST Graph (FallbackGraphEngine)
  3. Import Resolver
  4. Symbol Resolver
  5. Execution Path Discovery
  6. Git Context
  7. Folder Heuristic
  8. Strict User Scope

Degrades gracefully when external tools are unavailable.
"""

from __future__ import annotations

from typing import List, Optional, Protocol

from llm_prompt_optimizer.core.fallback_graph.engine import FallbackGraphEngine
from llm_prompt_optimizer.models.context import DependencyNode
from llm_prompt_optimizer.models.intent import IntentLock, ExpansionMode
from llm_prompt_optimizer.models.classification import PromptClassification
from llm_prompt_optimizer.utils.helpers import get_logger

logger = get_logger(__name__)


class GraphProvider(Protocol):
    """Interface any external graph provider must implement."""

    def available(self) -> bool: ...

    def discover(
        self, intent_lock: IntentLock, max_depth: int
    ) -> List[DependencyNode]: ...


class DependencyResolver:
    """
    Resolves dependencies using the best available method.

    Always works standalone via FallbackGraphEngine.
    Optionally enhanced by injected graph providers.
    """

    def __init__(
        self,
        repo_root: Optional[str] = None,
        graph_provider: Optional[GraphProvider] = None,
    ) -> None:
        self.repo_root = repo_root
        self._graph_provider = graph_provider
        self._fallback = FallbackGraphEngine(repo_root=repo_root)

    def resolve(
        self,
        intent_lock: IntentLock,
        classification: PromptClassification,
        max_depth: int = 5,
    ) -> List[DependencyNode]:
        """
        Discover dependencies using the best available engine.
        Always returns results — never raises due to unavailable tools.
        """
        # Strict mode: only use explicitly stated files, no discovery
        if intent_lock.allowed_expansion == ExpansionMode.STRICT:
            logger.info("Strict scope — skipping dependency discovery")
            return self._strict_scope_nodes(intent_lock)

        # Priority 1: External graph provider
        if self._graph_provider:
            try:
                if self._graph_provider.available():
                    logger.info("Using external graph provider")
                    nodes = self._graph_provider.discover(intent_lock, max_depth)
                    if nodes:
                        return nodes
                    logger.info("Graph provider returned empty — falling back to AST")
            except Exception as e:
                logger.warning(f"Graph provider failed: {e} — falling back")

        # Priority 2–7: FallbackGraphEngine (AST, imports, symbols, git, heuristics)
        logger.info("Using FallbackGraphEngine (AST + import + symbol + git)")
        try:
            nodes = self._fallback.discover_dependencies(intent_lock, max_depth=max_depth)
            self._annotate_with_classification(nodes, classification)
            return nodes
        except Exception as e:
            logger.error(f"FallbackGraphEngine failed: {e}")

        # Priority 8: Strict user scope only
        return self._strict_scope_nodes(intent_lock)

    def _strict_scope_nodes(self, intent_lock: IntentLock) -> List[DependencyNode]:
        """Return minimal nodes representing only the user-specified files."""
        return [
            DependencyNode(
                file_path=f,
                dependency_type="user_explicit",
                depth=0,
                confidence=1.0,
                discovery_method="user_scope",
            )
            for f in intent_lock.target_files
        ]

    def _annotate_with_classification(
        self,
        nodes: List[DependencyNode],
        classification: PromptClassification,
    ) -> None:
        """Boost confidence for nodes matching classified file signals."""
        classified_files = set(classification.extracted_file_paths)
        for node in nodes:
            for cf in classified_files:
                if cf in node.file_path or node.file_path.endswith(cf):
                    node.confidence = min(1.0, node.confidence + 0.10)
                    break

    def register_graph_provider(self, provider: GraphProvider) -> None:
        """Register an external graph provider at runtime."""
        self._graph_provider = provider
        logger.info(f"Graph provider registered: {type(provider).__name__}")
