"""
DriftDetector — Detects and reports prompt optimization drift.

Monitors for:
  - Scope widening (more files than requested)
  - Hallucinated files (files not in repo or not in intent)
  - Missing constraints (constraints removed during optimization)
  - Altered intent (core goal changed)
  - Added tasks (new tasks invented by optimizer)
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from llm_prompt_optimizer.models.intent import IntentLock
from llm_prompt_optimizer.models.validation import DriftReport, DriftEvent
from llm_prompt_optimizer.models.context import ContextBundle
from llm_prompt_optimizer.utils.helpers import get_logger, extract_file_paths

logger = get_logger(__name__)

_SCOPE_EXPANSION_PHRASES = [
    r"\b(also\s+(?:update|refactor|change|modify|review|check))\b",
    r"\b(additionally|furthermore|as\s+well\s+as)\b.*\b(file|module|function|class)\b",
    r"\b(while\s+(?:we|you)'?re?\s+at\s+it)\b",
    r"\b(also\s+consider)\b",
]

_HALLUCINATION_VERBS = [
    "rewrite", "redesign", "architect", "create new", "build from scratch",
    "refactor entire", "migrate to", "convert all",
]

_ADDED_TASK_PATTERNS = [
    r"\b(additionally|also|moreover|furthermore)\s+(?:implement|build|write|create|add)\b",
    r"\b(and\s+(?:make\s+sure|ensure|verify)\s+(?:to\s+)?(?:implement|build|add))\b",
]


class DriftDetector:
    """
    Detects semantic and scope drift between raw and optimized prompts.
    """

    def __init__(self) -> None:
        self._scope_patterns = [
            re.compile(p, re.IGNORECASE) for p in _SCOPE_EXPANSION_PHRASES
        ]
        self._added_task_patterns = [
            re.compile(p, re.IGNORECASE) for p in _ADDED_TASK_PATTERNS
        ]

    def detect(
        self,
        raw_text: str,
        optimized_text: str,
        intent_lock: IntentLock,
        context_bundle: Optional[ContextBundle] = None,
        repo_root: Optional[str] = None,
    ) -> DriftReport:
        """
        Run all drift detectors. Returns a DriftReport.
        """
        report = DriftReport()

        self._check_scope_widening(raw_text, optimized_text, intent_lock, report)
        self._check_hallucinated_files(optimized_text, intent_lock, repo_root, report)
        self._check_missing_constraints(raw_text, optimized_text, intent_lock, report)
        self._check_altered_intent(raw_text, optimized_text, intent_lock, report)
        self._check_added_tasks(raw_text, optimized_text, report)
        if context_bundle:
            self._check_context_bundle(context_bundle, intent_lock, report)

        if report.drifts_detected:
            logger.warning(
                f"Drift detected: {len(report.drifts_detected)} events, "
                f"severity={report.overall_severity}, blocked={report.blocked}"
            )
        else:
            logger.info("No drift detected")

        return report

    def _check_scope_widening(
        self,
        raw: str,
        optimized: str,
        intent_lock: IntentLock,
        report: DriftReport,
    ) -> None:
        """Detect if optimized prompt added files/modules beyond user scope."""
        raw_files = set(extract_file_paths(raw))
        opt_files = set(extract_file_paths(optimized))
        intent_files = set(intent_lock.target_files)
        allowed = raw_files | intent_files

        hallucinated = opt_files - allowed
        if hallucinated:
            report.add_drift(DriftEvent(
                drift_type="scope_widening",
                description=f"Optimized prompt references files not in original scope: {hallucinated}",
                severity="high",
                suggested_fix="Remove references to out-of-scope files.",
            ))

        for pattern in self._scope_patterns:
            if pattern.search(optimized) and not pattern.search(raw):
                report.add_drift(DriftEvent(
                    drift_type="scope_widening",
                    description=f"Scope-expanding language detected: '{pattern.pattern}'",
                    severity="medium",
                    suggested_fix="Remove scope-expanding qualifiers.",
                ))
                break

    def _check_hallucinated_files(
        self,
        optimized: str,
        intent_lock: IntentLock,
        repo_root: Optional[str],
        report: DriftReport,
    ) -> None:
        """Detect file references in optimized prompt that don't exist on disk."""
        opt_files = extract_file_paths(optimized)
        for f in opt_files:
            if f in intent_lock.target_files:
                continue  # user explicitly mentioned this
            if repo_root:
                full_path = os.path.join(repo_root, f)
                if not os.path.exists(full_path):
                    report.add_drift(DriftEvent(
                        drift_type="hallucinated_file",
                        description=f"File '{f}' referenced in optimized prompt does not exist on disk.",
                        severity="critical",
                        suggested_fix=f"Remove or verify '{f}' before injection.",
                    ))

    def _check_missing_constraints(
        self,
        raw: str,
        optimized: str,
        intent_lock: IntentLock,
        report: DriftReport,
    ) -> None:
        """Ensure user constraints from IntentLock survive optimization."""
        for constraint in intent_lock.constraints:
            if constraint.constraint_type == "no_code_changes":
                # Check that optimized doesn't add imperative code-change language
                imperative = ["implement", "write the following", "add this code"]
                if any(w in optimized.lower() for w in imperative):
                    report.add_drift(DriftEvent(
                        drift_type="missing_constraint",
                        description="'no_code_changes' constraint violated — imperative language added.",
                        severity="high",
                        suggested_fix="Remove imperative code-change directives from optimized prompt.",
                    ))

            if constraint.constraint_type == "no_repo_scan":
                repo_scan_phrases = ["scan the repo", "search the codebase", "look through all files"]
                if any(p in optimized.lower() for p in repo_scan_phrases):
                    report.add_drift(DriftEvent(
                        drift_type="missing_constraint",
                        description="'no_repo_scan' constraint violated.",
                        severity="high",
                    ))

    def _check_altered_intent(
        self,
        raw: str,
        optimized: str,
        intent_lock: IntentLock,
        report: DriftReport,
    ) -> None:
        """Detect if the core goal of the prompt was changed."""
        # Core intent signals must survive
        intent_words = set(intent_lock.intent_summary.lower().split())
        stop_words = {"the", "a", "an", "in", "on", "at", "for", "of", "and", "or"}
        key_words = intent_words - stop_words

        opt_lower = optimized.lower()
        missing = [w for w in key_words if w not in opt_lower and len(w) > 3]

        if len(missing) > len(key_words) * 0.4:
            report.add_drift(DriftEvent(
                drift_type="altered_intent",
                description=f"Core intent keywords lost: {missing[:5]}",
                severity="high",
                suggested_fix="Re-inject the original intent summary.",
            ))

    def _check_added_tasks(
        self, raw: str, optimized: str, report: DriftReport
    ) -> None:
        """Detect if new tasks were invented in the optimized prompt."""
        for pattern in self._added_task_patterns:
            if pattern.search(optimized) and not pattern.search(raw):
                report.add_drift(DriftEvent(
                    drift_type="added_task",
                    description=f"New task-like language introduced: '{pattern.pattern}'",
                    severity="medium",
                    suggested_fix="Remove unoriginated task language.",
                ))

    def _check_context_bundle(
        self,
        bundle: ContextBundle,
        intent_lock: IntentLock,
        report: DriftReport,
    ) -> None:
        """Validate that context bundle doesn't include forbidden files."""
        for file_path in bundle.files_included:
            if file_path in intent_lock.forbidden_files:
                report.add_drift(DriftEvent(
                    drift_type="hallucinated_file",
                    description=f"Forbidden file included in context bundle: {file_path}",
                    severity="critical",
                ))
