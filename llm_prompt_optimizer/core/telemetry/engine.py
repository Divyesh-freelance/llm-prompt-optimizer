"""
TelemetryEngine — Tracks optimization metrics and emits events.
PromptAuditLogger — Append-only audit trail of every optimization run.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable

from llm_prompt_optimizer.models.telemetry import TelemetryEvent, OptimizationMetrics
from llm_prompt_optimizer.models.prompt import PromptOptimizationResult
from llm_prompt_optimizer.utils.helpers import get_logger

logger = get_logger(__name__)


class TelemetryEngine:
    """
    Collects and emits telemetry events from the optimization pipeline.

    Tracks:
      - token savings
      - dependency accuracy
      - semantic preservation
      - context efficiency
      - optimization confidence
      - drift prevention
    """

    def __init__(self, backend: Optional[str] = None) -> None:
        self.backend = backend  # None | "opentelemetry" | "prometheus"
        self._handlers: List[Callable[[TelemetryEvent], None]] = []
        self._events: List[TelemetryEvent] = []

    def register_handler(self, handler: Callable[[TelemetryEvent], None]) -> None:
        self._handlers.append(handler)

    def emit(self, event: TelemetryEvent) -> None:
        self._events.append(event)
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                logger.debug(f"Telemetry handler error: {e}")
        logger.debug(f"Telemetry: {event.event_type} [{event.component}]")

    def build_metrics(self, result: PromptOptimizationResult) -> OptimizationMetrics:
        """Build aggregated metrics from an optimization result."""
        opt = result.optimized_prompt
        raw = result.raw_prompt
        from llm_prompt_optimizer.utils.helpers import estimate_tokens

        raw_tokens = estimate_tokens(raw.text)
        opt_tokens = opt.token_estimate

        metrics = OptimizationMetrics(prompt_id=raw.prompt_id)
        metrics.original_token_estimate = raw_tokens
        metrics.optimized_token_estimate = opt_tokens
        metrics.compute_token_reduction()
        metrics.semantic_similarity = opt.semantic_similarity
        metrics.context_spans_count = len(opt.context_spans)
        metrics.pipeline_duration_ms = result.pipeline_duration_ms
        metrics.policy_violations = len(result.policy_violations)
        metrics.validation_passed = result.success

        if result.context_bundle:
            metrics.dependency_files_count = len(result.context_bundle.files_included)
            metrics.context_lines_total = result.context_bundle.total_lines

        if result.drift_report:
            metrics.drift_events_count = len(result.drift_report.drifts_detected)

        if result.classification:
            metrics.classification_confidence = result.classification.primary_confidence

        if result.intent_lock:
            metrics.intent_confidence = result.intent_lock.confidence

        return metrics

    def get_events(self) -> List[TelemetryEvent]:
        return list(self._events)


class PromptAuditLogger:
    """
    Append-only structured audit trail for every optimization run.
    Writes JSONL to a configurable audit log file.
    """

    def __init__(self, log_path: Optional[str] = None) -> None:
        self.log_path = log_path or str(
            Path.home() / ".llm_prompt_optimizer" / "audit.jsonl"
        )
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

    def log_result(self, result: PromptOptimizationResult, metrics: OptimizationMetrics) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt_id": result.raw_prompt.prompt_id,
            "optimization_id": result.optimized_prompt.optimization_id,
            "success": result.success,
            "token_savings": metrics.token_savings,
            "token_reduction_pct": round(metrics.token_reduction_pct, 2),
            "semantic_similarity": metrics.semantic_similarity,
            "category": result.classification.primary_category.value if result.classification else "unknown",
            "drift_events": metrics.drift_events_count,
            "policy_violations": metrics.policy_violations,
            "validation_passed": metrics.validation_passed,
            "pipeline_duration_ms": round(result.pipeline_duration_ms, 2),
            "errors": result.errors,
            "warnings": result.warnings,
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Audit log write failed: {e}")

    def read_recent(self, n: int = 50) -> List[dict]:
        """Read the N most recent audit entries."""
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return [json.loads(l) for l in lines[-n:]]
        except Exception:
            return []
