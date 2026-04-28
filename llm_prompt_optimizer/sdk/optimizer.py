"""
Optimizer SDK — Primary user-facing entry point.

Usage:
    from llm_prompt_optimizer import Optimizer

    optimizer = Optimizer()
    result = optimizer.optimize(
        prompt="debug EMA mismatch",
        strict_mode=True,
    )
    print(result.optimized_prompt.text)
    print(result.telemetry)
"""

from __future__ import annotations

import time
from typing import Optional, Union

from llm_prompt_optimizer.config.settings import OptimizerConfig
from llm_prompt_optimizer.core.intent_guard.guard import IntentGuard
from llm_prompt_optimizer.core.classifier.classifier import PromptClassifier
from llm_prompt_optimizer.core.dependency_resolution.resolver import DependencyResolver
from llm_prompt_optimizer.core.adaptive_context_expansion.expansion import AdaptiveContextExpansion
from llm_prompt_optimizer.core.precise_context.resolver import PreciseContextResolver
from llm_prompt_optimizer.core.optimizer.optimizer import PromptOptimizer
from llm_prompt_optimizer.core.optimizer.compiler import PromptCompiler
from llm_prompt_optimizer.core.semantic_validator.validator import SemanticValidator
from llm_prompt_optimizer.core.drift_detection.detector import DriftDetector
from llm_prompt_optimizer.core.policy.engine import PolicyEngine
from llm_prompt_optimizer.core.token_budget.engine import TokenBudgetEngine
from llm_prompt_optimizer.core.telemetry.engine import TelemetryEngine, PromptAuditLogger
from llm_prompt_optimizer.core.fallback_graph.engine import FallbackGraphEngine
from llm_prompt_optimizer.models.prompt import RawPrompt, PromptOptimizationResult
from llm_prompt_optimizer.plugins.system import PluginSystem
from llm_prompt_optimizer.utils.helpers import get_logger, estimate_tokens

logger = get_logger(__name__)


class Optimizer:
    """
    Main SDK entry point for the LLM Prompt Optimizer.

    Wires together the full pipeline:
      IntentGuard → Classifier → DependencyResolver →
      AdaptiveContextExpansion → PreciseContext → PromptOptimizer →
      SemanticValidator → DriftDetector → PolicyEngine → PromptCompiler

    All components are injected — can be replaced/extended for testing or enterprise use.
    """

    def __init__(
        self,
        config: Optional[OptimizerConfig] = None,
        plugin_system: Optional[PluginSystem] = None,
        repo_root: Optional[str] = None,
    ) -> None:
        self.config = config or OptimizerConfig.from_env()
        self.repo_root = repo_root
        self.plugins = plugin_system or PluginSystem()

        # Load plugin directories
        for plugin_dir in self.config.plugin_dirs:
            self.plugins.load_directory(plugin_dir)

        # Instantiate pipeline components
        self._intent_guard = IntentGuard()
        self._classifier = PromptClassifier()
        self._token_engine = TokenBudgetEngine(self.config.token_budget)
        self._fallback_graph = FallbackGraphEngine(repo_root=repo_root)
        self._dep_resolver = DependencyResolver(
            repo_root=repo_root,
            graph_provider=self.plugins.get_best_graph_provider(),
        )
        self._context_resolver = PreciseContextResolver()
        self._adaptive_expansion = AdaptiveContextExpansion(
            config=self.config.adaptive_expansion,
            token_engine=self._token_engine,
            graph_engine=self._fallback_graph,
            context_resolver=self._context_resolver,
        )
        self._optimizer = PromptOptimizer()
        self._compiler = PromptCompiler()
        self._validator = SemanticValidator(self.config.policy)
        self._drift_detector = DriftDetector()
        self._policy = PolicyEngine(self.config.policy)
        self._telemetry = TelemetryEngine(backend=self.config.telemetry_backend)
        self._audit = PromptAuditLogger()

        # Wire telemetry plugins
        for tp in self.plugins.registry.get_telemetry():
            self._telemetry.register_handler(tp.handle_event)

        logger.info("Optimizer initialized")

    # ── Public API ────────────────────────────────────────────────────────────

    def optimize(
        self,
        prompt: Union[str, RawPrompt],
        strict_mode: bool = False,
        repo_root: Optional[str] = None,
        fast_mode: bool = False,
    ) -> PromptOptimizationResult:
        """
        Run the full optimization pipeline.

        Args:
            prompt: Raw prompt string or RawPrompt object.
            strict_mode: If True, override expansion to STRICT regardless of prompt signals.
            repo_root: Optional override for repository root path.
            fast_mode: If True, skip adaptive context expansion and embedding-
                based validation. Use for short prompts or when latency
                matters more than maximal compression. Implies strict_mode.

        Returns:
            PromptOptimizationResult with all pipeline outputs.
        """
        t_start = time.time()

        if isinstance(prompt, str):
            raw = RawPrompt(
                text=prompt,
                repo_root=repo_root or self.repo_root,
            )
        else:
            raw = prompt
            if repo_root:
                raw.repo_root = repo_root

        result = PromptOptimizationResult(
            raw_prompt=raw,
            optimized_prompt=None,  # type: ignore[arg-type]
        )

        try:
            # ── Stage 1: IntentGuard ──────────────────────────────────────────
            intent_lock = self._intent_guard.extract_and_lock(raw)
            if strict_mode:
                from llm_prompt_optimizer.models.intent import ExpansionMode
                # Override via a new lock copy (since lock is sealed, we re-seal)
                object.__setattr__(intent_lock, "locked", False)
                intent_lock.allowed_expansion = ExpansionMode.STRICT
                intent_lock.lock()
            result.intent_lock = intent_lock

            # ── Stage 2: Classification ───────────────────────────────────────
            classification = self._classifier.classify(raw, intent_lock)
            result.classification = classification

            # ── Stage 3: Policy pre-check on intent ──────────────────────────
            intent_policy = self._policy.check_intent(intent_lock)
            if intent_policy.violations:
                result.warnings.extend(
                    f"[Policy] {v.rule}: {v.description}"
                    for v in intent_policy.violations
                    if v.severity == "warn"
                )

            # ── Stage 4: Token budget ─────────────────────────────────────────
            budget = self._token_engine.create_budget(estimate_tokens(raw.text))

            if fast_mode:
                # Fast path: skip dependency discovery and adaptive expansion
                # entirely. These are the dominant cost on a real repo. We
                # still extract direct spans (the user explicitly named files
                # in their prompt) but no graph walking, no git history.
                logger.debug("fast_mode: skipping dependency + adaptive expansion")
                dep_nodes = []
                direct_spans = self._context_resolver.resolve(
                    intent_lock, classification, repo_root=raw.repo_root
                )
                from llm_prompt_optimizer.models.context import ContextBundle
                context_bundle = ContextBundle(
                    spans=direct_spans,
                    total_token_cost=sum(s.token_cost for s in direct_spans),
                    total_lines=sum(s.line_count() for s in direct_spans),
                    files_included=list({s.file_path for s in direct_spans}),
                    expansion_stopped_reason="fast_mode",
                )
                result.context_bundle = context_bundle
            else:
                # ── Stage 5: Dependency resolution ───────────────────────────────
                # This honors the priority chain: external graph provider
                # (Graphify, Code Review Graph) first, FallbackGraphEngine
                # only if none registered.
                dep_nodes = self._dep_resolver.resolve(intent_lock, classification)

                # ── Stage 6: Precise context extraction ──────────────────────────
                direct_spans = self._context_resolver.resolve(
                    intent_lock, classification, repo_root=raw.repo_root
                )

                # ── Stage 7: Adaptive context expansion ──────────────────────────
                # Reuse the candidates discovered in stage 5 instead of
                # re-walking the graph. When Graphify is registered this
                # means the AST engine never runs in this turn.
                context_bundle = self._adaptive_expansion.expand(
                    intent_lock=intent_lock,
                    classification=classification,
                    initial_spans=direct_spans,
                    budget_state=budget,
                    repo_root=raw.repo_root,
                    candidates=dep_nodes,
                )
                result.context_bundle = context_bundle

            # ── Stage 8: Plugin pre-optimize middleware ───────────────────────
            pre_text = self.plugins.apply_optimizer_plugins_pre(
                raw.text, {"intent_lock": intent_lock, "classification": classification}
            )
            if pre_text != raw.text:
                logger.debug("Plugin modified prompt pre-optimization")

            # ── Stage 9: Prompt optimization ─────────────────────────────────
            optimized = self._optimizer.optimize(
                raw_prompt=RawPrompt(
                    text=pre_text,
                    prompt_id=raw.prompt_id,
                    repo_root=raw.repo_root,
                ),
                intent_lock=intent_lock,
                classification=classification,
                context_bundle=context_bundle,
                strict_mode=strict_mode,
            )

            # ── Stage 10: Semantic validation ─────────────────────────────────
            validation = self._validator.validate(
                raw.text, optimized.text, intent_lock, fast_mode=fast_mode
            )
            result.validation_result = validation
            optimized.semantic_similarity = validation.semantic_similarity

            # ── Stage 11: Drift detection ─────────────────────────────────────
            drift = self._drift_detector.detect(
                raw_text=raw.text,
                optimized_text=optimized.text,
                intent_lock=intent_lock,
                context_bundle=context_bundle,
                repo_root=raw.repo_root,
            )
            result.drift_report = drift

            # ── Stage 12: Full policy check ───────────────────────────────────
            policy_result = self._policy.check_all(intent_lock, validation, drift)
            result.policy_violations = [
                f"{v.rule}: {v.description}" for v in policy_result.violations
            ]
            if policy_result.blocked:
                result.errors.append("Pipeline blocked by policy engine")
                result.success = False

            # ── Stage 13: Plugin policy extensions ────────────────────────────
            plugin_violations = self.plugins.evaluate_policy_plugins({
                "intent_lock": intent_lock,
                "classification": classification,
                "validation": validation,
                "drift": drift,
            })
            result.policy_violations.extend(plugin_violations)

            # ── Stage 14: Prompt compilation ──────────────────────────────────
            compiled_text = self._compiler.compile(
                optimized=optimized,
                intent_lock=intent_lock,
                validation=validation,
                drift_report=drift,
                fallback_to_raw=True,
                raw_text=raw.text,
            )

            # ── Stage 15: Plugin post-optimize middleware ─────────────────────
            compiled_text = self.plugins.apply_optimizer_plugins_post(
                compiled_text, {"intent_lock": intent_lock}
            )

            optimized.text = compiled_text
            result.optimized_prompt = optimized
            result.success = result.success and True

        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            result.errors.append(str(e))
            result.success = False
            # Ensure optimized_prompt is set even on failure
            if result.optimized_prompt is None:
                from llm_prompt_optimizer.models.prompt import OptimizedPrompt
                result.optimized_prompt = OptimizedPrompt(
                    text=raw.text,
                    original_prompt_id=raw.prompt_id,
                )

        # ── Telemetry & Audit ─────────────────────────────────────────────────
        result.pipeline_duration_ms = (time.time() - t_start) * 1000
        metrics = self._telemetry.build_metrics(result)
        result.telemetry = metrics
        if self.config.policy.enable_audit_log:
            self._audit.log_result(result, metrics)

        logger.info(
            f"Optimization complete: success={result.success} "
            f"duration={result.pipeline_duration_ms:.1f}ms "
            f"tokens={metrics.optimized_token_estimate}"
        )
        return result

    def classify(self, prompt: Union[str, RawPrompt]):
        """Classify a prompt without full optimization."""
        if isinstance(prompt, str):
            raw = RawPrompt(text=prompt)
        else:
            raw = prompt
        intent_lock = self._intent_guard.extract_and_lock(raw)
        return self._classifier.classify(raw, intent_lock)

    def validate(self, raw_text: str, optimized_text: str):
        """Run semantic validation only."""
        raw = RawPrompt(text=raw_text)
        intent_lock = self._intent_guard.extract_and_lock(raw)
        return self._validator.validate(raw_text, optimized_text, intent_lock)

    def detect_drift(self, raw_text: str, optimized_text: str):
        """Run drift detection only."""
        raw = RawPrompt(text=raw_text)
        intent_lock = self._intent_guard.extract_and_lock(raw)
        return self._drift_detector.detect(raw_text, optimized_text, intent_lock)

    def estimate_cost(self, prompt: Union[str, RawPrompt]) -> dict:
        """Estimate token cost for a prompt."""
        text = prompt if isinstance(prompt, str) else prompt.text
        tokens = estimate_tokens(text)
        return {"estimated_tokens": tokens, "approx_chars": len(text)}

    def benchmark(self, cases=None):
        """Run the built-in benchmark suite."""
        from llm_prompt_optimizer.core.benchmarking.engine import BenchmarkEngine
        engine = BenchmarkEngine(optimizer=self)
        return engine.run(cases=cases, use_builtin=(cases is None))

    def list_plugins(self):
        return self.plugins.list_plugins()
