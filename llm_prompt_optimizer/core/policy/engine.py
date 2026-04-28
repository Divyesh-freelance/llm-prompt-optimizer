"""
PolicyEngine — Enforces governance policies across the pipeline.

Reads from PolicyConfig and blocks/warns based on:
  - strict_intent_mode
  - semantic_similarity_threshold
  - adaptive_context_budget
  - allow_scope_expansion
  - require_dependency_validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from llm_prompt_optimizer.config.settings import PolicyConfig
from llm_prompt_optimizer.models.intent import IntentLock, ExpansionMode
from llm_prompt_optimizer.models.validation import ValidationResult, DriftReport
from llm_prompt_optimizer.utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class PolicyViolation:
    rule: str
    description: str
    severity: str  # warn | block
    value: Optional[str] = None


@dataclass
class PolicyResult:
    passed: bool
    violations: List[PolicyViolation] = field(default_factory=list)
    blocked: bool = False

    def add_violation(self, v: PolicyViolation) -> None:
        self.violations.append(v)
        if v.severity == "block":
            self.blocked = True
            self.passed = False


class PolicyEngine:
    """Enforces policy rules at key pipeline checkpoints."""

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        self.config = config or PolicyConfig()

    def check_intent(self, intent_lock: IntentLock) -> PolicyResult:
        """Validate intent lock against policy rules."""
        result = PolicyResult(passed=True)

        if self.config.strict_intent_mode:
            if intent_lock.confidence < 0.50:
                result.add_violation(PolicyViolation(
                    rule="strict_intent_mode",
                    description=f"Intent confidence {intent_lock.confidence:.2f} too low for strict mode.",
                    severity="warn",
                ))

        if (
            self.config.allow_scope_expansion == "strict"
            and intent_lock.allowed_expansion != ExpansionMode.STRICT
        ):
            result.add_violation(PolicyViolation(
                rule="allow_scope_expansion",
                description="Policy requires strict scope but prompt allows expansion.",
                severity="warn",
                value=intent_lock.allowed_expansion.value,
            ))

        return result

    def check_validation(self, validation: ValidationResult) -> PolicyResult:
        """Enforce semantic similarity threshold."""
        result = PolicyResult(passed=True)

        if validation.semantic_similarity < self.config.semantic_similarity_threshold:
            result.add_violation(PolicyViolation(
                rule="semantic_similarity_threshold",
                description=(
                    f"Similarity {validation.semantic_similarity:.3f} < "
                    f"threshold {self.config.semantic_similarity_threshold:.2f}"
                ),
                severity="block",
                value=str(validation.semantic_similarity),
            ))

        return result

    def check_drift(self, drift_report: DriftReport) -> PolicyResult:
        """Apply policy to drift results."""
        result = PolicyResult(passed=True)

        if drift_report.blocked:
            result.add_violation(PolicyViolation(
                rule="drift_detection",
                description=f"Critical drift detected: {drift_report.overall_severity}",
                severity="block",
            ))
        elif not drift_report.is_clean:
            result.add_violation(PolicyViolation(
                rule="drift_detection",
                description=f"Drift detected (non-blocking): {drift_report.overall_severity}",
                severity="warn",
            ))

        return result

    def check_all(
        self,
        intent_lock: IntentLock,
        validation: ValidationResult,
        drift_report: DriftReport,
    ) -> PolicyResult:
        """Run all policy checks and aggregate results."""
        combined = PolicyResult(passed=True)

        for check in [
            self.check_intent(intent_lock),
            self.check_validation(validation),
            self.check_drift(drift_report),
        ]:
            for v in check.violations:
                combined.add_violation(v)

        total_blocks = sum(1 for v in combined.violations if v.severity == "block")
        if total_blocks > self.config.max_policy_violations_before_block:
            combined.blocked = True
            combined.passed = False

        if combined.violations:
            logger.warning(
                f"Policy: {len(combined.violations)} violations, "
                f"blocked={combined.blocked}"
            )

        return combined
