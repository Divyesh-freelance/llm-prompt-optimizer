"""
BenchmarkEngine — Measures optimization quality across multiple dimensions.

Measures:
  - debugging quality
  - semantic similarity
  - context usefulness
  - dependency precision
  - token reduction
  - adaptive expansion efficiency
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from llm_prompt_optimizer.models.prompt import RawPrompt, PromptOptimizationResult
from llm_prompt_optimizer.utils.helpers import get_logger, cosine_similarity_simple, estimate_tokens

logger = get_logger(__name__)


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    name: str
    prompt: str
    expected_category: str
    expected_files: List[str] = field(default_factory=list)
    expected_symbols: List[str] = field(default_factory=list)
    expected_constraints: List[str] = field(default_factory=list)
    min_similarity: float = 0.90
    max_token_budget: int = 4000
    repo_root: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Result for a single benchmark case."""
    case_name: str
    passed: bool
    semantic_similarity: float = 0.0
    token_reduction_pct: float = 0.0
    category_correct: bool = False
    files_precision: float = 0.0
    symbols_precision: float = 0.0
    constraints_preserved: float = 0.0
    pipeline_duration_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        weights = {
            "semantic_similarity": 0.35,
            "category_correct": 0.15,
            "token_reduction_pct": 0.15,
            "files_precision": 0.15,
            "constraints_preserved": 0.20,
        }
        score = (
            self.semantic_similarity * weights["semantic_similarity"]
            + (1.0 if self.category_correct else 0.0) * weights["category_correct"]
            + min(1.0, self.token_reduction_pct / 50) * weights["token_reduction_pct"]
            + self.files_precision * weights["files_precision"]
            + self.constraints_preserved * weights["constraints_preserved"]
        )
        return round(score, 4)


@dataclass
class BenchmarkReport:
    results: List[BenchmarkResult] = field(default_factory=list)
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    avg_similarity: float = 0.0
    avg_token_reduction: float = 0.0
    avg_overall_score: float = 0.0
    duration_ms: float = 0.0

    def summarize(self) -> Dict[str, float]:
        if not self.results:
            return {}
        self.total_cases = len(self.results)
        self.passed = sum(1 for r in self.results if r.passed)
        self.failed = self.total_cases - self.passed
        self.avg_similarity = sum(r.semantic_similarity for r in self.results) / self.total_cases
        self.avg_token_reduction = sum(r.token_reduction_pct for r in self.results) / self.total_cases
        self.avg_overall_score = sum(r.overall_score for r in self.results) / self.total_cases
        return {
            "total": self.total_cases,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.passed / self.total_cases,
            "avg_similarity": round(self.avg_similarity, 4),
            "avg_token_reduction_pct": round(self.avg_token_reduction, 2),
            "avg_overall_score": round(self.avg_overall_score, 4),
        }


# Built-in benchmark suite
BUILTIN_CASES: List[BenchmarkCase] = [
    BenchmarkCase(
        name="debug_ema_mismatch",
        prompt="Debug EMA mismatch in signals/IndexSignals.py. No code changes needed.",
        expected_category="debugging",
        expected_files=["signals/IndexSignals.py"],
        expected_constraints=["no_code_changes"],
    ),
    BenchmarkCase(
        name="implement_endpoint",
        prompt="Implement a /health endpoint in api/routes.py that returns 200 OK.",
        expected_category="implementation",
        expected_files=["api/routes.py"],
    ),
    BenchmarkCase(
        name="refactor_utils",
        prompt="Refactor utils/helpers.py to remove duplicate logic in format_date and parse_date.",
        expected_category="refactoring",
        expected_files=["utils/helpers.py"],
    ),
    BenchmarkCase(
        name="debug_with_stacktrace",
        prompt=(
            "Getting this error:\n"
            "Traceback (most recent call last):\n"
            "  File 'signals/IndexSignals.py', line 87, in calculate\n"
            "    ValueError: EMA length mismatch\n"
            "Why is this happening?"
        ),
        expected_category="debugging",
        expected_files=["signals/IndexSignals.py"],
    ),
    BenchmarkCase(
        name="vague_prompt",
        prompt="Fix the bug",
        expected_category="debugging",
        min_similarity=0.85,
    ),
]


class BenchmarkEngine:
    """Runs benchmark cases against the optimizer pipeline."""

    def __init__(self, optimizer=None) -> None:
        """optimizer: an Optimizer instance (injected to avoid circular import)."""
        self.optimizer = optimizer

    def run(
        self,
        cases: Optional[List[BenchmarkCase]] = None,
        use_builtin: bool = True,
    ) -> BenchmarkReport:
        if cases is None:
            cases = BUILTIN_CASES if use_builtin else []

        report = BenchmarkReport()
        start = time.time()

        for case in cases:
            result = self._run_case(case)
            report.results.append(result)
            logger.info(
                f"Benchmark [{case.name}]: score={result.overall_score:.3f} "
                f"passed={result.passed}"
            )

        report.duration_ms = (time.time() - start) * 1000
        report.summarize()
        return report

    def _run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        br = BenchmarkResult(case_name=case.name, passed=False)
        t0 = time.time()

        try:
            if self.optimizer is None:
                br.errors.append("No optimizer injected")
                return br

            raw = RawPrompt(text=case.prompt, repo_root=case.repo_root)
            opt_result: PromptOptimizationResult = self.optimizer.optimize(
                prompt=raw, strict_mode=False
            )
            elapsed = (time.time() - t0) * 1000
            br.pipeline_duration_ms = elapsed

            # Semantic similarity
            br.semantic_similarity = (
                opt_result.optimized_prompt.semantic_similarity
                if opt_result.optimized_prompt.semantic_similarity < 1.0
                else cosine_similarity_simple(
                    case.prompt, opt_result.optimized_prompt.text
                )
            )

            # Token reduction
            raw_tokens = estimate_tokens(case.prompt)
            opt_tokens = opt_result.optimized_prompt.token_estimate
            if raw_tokens > 0:
                br.token_reduction_pct = max(
                    0.0, ((raw_tokens - opt_tokens) / raw_tokens) * 100
                )

            # Category precision
            if opt_result.classification:
                br.category_correct = (
                    opt_result.classification.primary_category.value == case.expected_category
                )

            # File precision
            if case.expected_files and opt_result.intent_lock:
                found = set(opt_result.intent_lock.target_files)
                expected = set(case.expected_files)
                if expected:
                    br.files_precision = len(found & expected) / len(expected)

            # Constraints preserved
            if case.expected_constraints and opt_result.intent_lock:
                found_c = {c.constraint_type for c in opt_result.intent_lock.constraints}
                expected_c = set(case.expected_constraints)
                br.constraints_preserved = len(found_c & expected_c) / len(expected_c)

            br.passed = (
                br.semantic_similarity >= case.min_similarity
                and opt_result.success
            )

        except Exception as e:
            br.errors.append(str(e))
            logger.error(f"Benchmark case '{case.name}' failed: {e}")

        return br
