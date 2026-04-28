"""Validation and drift detection models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ValidationResult:
    """Result of semantic validation between raw and optimized prompt."""

    passed: bool
    semantic_similarity: float
    threshold: float = 0.90
    method: str = "cosine"  # cosine | bleu | rouge | llm_judge
    failure_reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        return not self.passed


@dataclass
class DriftEvent:
    """A single detected drift incident."""

    drift_type: str  # scope_widening | hallucinated_file | missing_constraint | altered_intent | added_task
    description: str
    severity: str  # low | medium | high | critical
    source: Optional[str] = None
    suggested_fix: Optional[str] = None


@dataclass
class DriftReport:
    """Comprehensive drift detection report."""

    drifts_detected: List[DriftEvent] = field(default_factory=list)
    is_clean: bool = True
    overall_severity: str = "none"  # none | low | medium | high | critical
    blocked: bool = False
    remediation_applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_drift(self, event: DriftEvent) -> None:
        self.drifts_detected.append(event)
        self.is_clean = False
        severity_rank = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        current = severity_rank.get(self.overall_severity, 0)
        incoming = severity_rank.get(event.severity, 0)
        if incoming > current:
            self.overall_severity = event.severity
        if event.severity == "critical":
            self.blocked = True
