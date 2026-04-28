"""
Standalone benchmark runner.

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_prompt_optimizer import Optimizer, OptimizerConfig
from llm_prompt_optimizer.core.benchmarking.engine import BenchmarkEngine, BUILTIN_CASES


def main():
    parser = argparse.ArgumentParser(description="LLM Prompt Optimizer Benchmark Runner")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--case", type=str, default=None, help="Run a specific case by name")
    args = parser.parse_args()

    cfg = OptimizerConfig()
    cfg.policy.enable_audit_log = False
    optimizer = Optimizer(config=cfg)

    cases = BUILTIN_CASES
    if args.case:
        cases = [c for c in cases if c.name == args.case]
        if not cases:
            print(f"Case '{args.case}' not found.", file=sys.stderr)
            sys.exit(1)

    engine = BenchmarkEngine(optimizer=optimizer)
    report = engine.run(cases=cases, use_builtin=False)
    summary = report.summarize()

    if args.json:
        output = {
            "summary": summary,
            "duration_ms": report.duration_ms,
            "results": [
                {
                    "case": r.case_name,
                    "passed": r.passed,
                    "overall_score": r.overall_score,
                    "semantic_similarity": r.semantic_similarity,
                    "token_reduction_pct": r.token_reduction_pct,
                    "category_correct": r.category_correct,
                    "constraints_preserved": r.constraints_preserved,
                    "pipeline_duration_ms": r.pipeline_duration_ms,
                    "errors": r.errors,
                }
                for r in report.results
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  LLM Prompt Optimizer — Benchmark Report")
        print(f"{'='*60}")
        print(f"  Cases:         {summary['total']}")
        print(f"  Passed:        {summary['passed']}/{summary['total']}")
        print(f"  Pass rate:     {summary['pass_rate']:.1%}")
        print(f"  Avg score:     {summary['avg_overall_score']:.3f}")
        print(f"  Avg similarity:{summary['avg_similarity']:.3f}")
        print(f"  Avg reduction: {summary['avg_token_reduction_pct']:.1f}%")
        print(f"  Duration:      {report.duration_ms:.0f}ms")
        print(f"\n{'─'*60}")
        print(f"  {'CASE':<35} {'SCORE':>6} {'SIM':>6} {'OK':>4}")
        print(f"{'─'*60}")
        for r in report.results:
            status = "✓" if r.passed else "✗"
            print(
                f"  {status} {r.case_name:<33} "
                f"{r.overall_score:>6.3f} "
                f"{r.semantic_similarity:>6.3f} "
                f"{'yes' if r.category_correct else 'no':>4}"
            )
            if r.errors:
                for err in r.errors:
                    print(f"    ERROR: {err}")
        print(f"{'='*60}\n")

    sys.exit(0 if summary.get("pass_rate", 0) >= 0.6 else 1)


if __name__ == "__main__":
    main()
