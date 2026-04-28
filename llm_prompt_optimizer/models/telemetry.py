"""Telemetry and metrics models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
import uuid


@dataclass
class TelemetryEvent:
    """A single telemetry event emitted from the pipeline."""

    event_type: str
    component: str
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    prompt_id: Optional[str] = None
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class OptimizationMetrics:
    """Aggregated metrics for a single optimization run."""

    prompt_id: str
    original_token_estimate: int = 0
    optimized_token_estimate: int = 0
    token_savings: int = 0
    token_reduction_pct: float = 0.0
    semantic_similarity: float = 1.0
    context_spans_count: int = 0
    dependency_files_count: int = 0
    context_lines_total: int = 0
    drift_events_count: int = 0
    classification_confidence: float = 0.0
    intent_confidence: float = 0.0
    adaptive_expansion_iterations: int = 0
    pipeline_duration_ms: float = 0.0
    policy_violations: int = 0
    validation_passed: bool = True
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_token_reduction(self) -> None:
        if self.original_token_estimate > 0:
            savings = self.original_token_estimate - self.optimized_token_estimate
            self.token_savings = max(0, savings)
            self.token_reduction_pct = (savings / self.original_token_estimate) * 100
