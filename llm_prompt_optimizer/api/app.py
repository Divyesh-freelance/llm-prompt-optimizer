"""
FastAPI REST API for LLM Prompt Optimizer.

Endpoints:
  POST /optimize
  POST /classify
  POST /resolve-context
  POST /validate
  POST /estimate-cost
  POST /detect-drift
  GET  /health
  GET  /plugins
  POST /benchmark
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from llm_prompt_optimizer import Optimizer, OptimizerConfig
from llm_prompt_optimizer.models.prompt import RawPrompt

app = FastAPI(
    title="LLM Prompt Optimizer API",
    description="Deterministic prompt optimization middleware for AI coding agents.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Singleton optimizer (initialized once on startup)
_optimizer: Optional[Optimizer] = None


def get_optimizer() -> Optimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = Optimizer(config=OptimizerConfig.from_env())
    return _optimizer


# ── Request / Response schemas ─────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    prompt: str = Field(..., description="The raw user prompt to optimize.")
    strict_mode: bool = Field(False, description="Enable strict scope expansion mode.")
    repo_root: Optional[str] = Field(None, description="Absolute path to repository root.")
    source_agent: Optional[str] = Field(None, description="Identifying the calling agent.")

class OptimizeResponse(BaseModel):
    prompt_id: str
    optimization_id: str
    optimized_text: str
    original_text: str
    token_estimate: int
    semantic_similarity: float
    compression_ratio: float
    category: str
    intent: str
    context_spans: List[Dict[str, Any]]
    injected_constraints: List[str]
    success: bool
    warnings: List[str]
    policy_violations: List[str]
    pipeline_duration_ms: float


class ClassifyRequest(BaseModel):
    prompt: str

class ClassifyResponse(BaseModel):
    primary_category: str
    primary_confidence: float
    complexity_score: float
    has_stacktrace: bool
    has_logs: bool
    has_code_snippet: bool
    extracted_files: List[str]
    extracted_symbols: List[str]
    multi_file: bool


class ValidateRequest(BaseModel):
    raw_text: str
    optimized_text: str

class ValidateResponse(BaseModel):
    passed: bool
    semantic_similarity: float
    threshold: float
    method: str
    failure_reason: Optional[str]


class DriftRequest(BaseModel):
    raw_text: str
    optimized_text: str
    repo_root: Optional[str] = None

class DriftResponse(BaseModel):
    is_clean: bool
    overall_severity: str
    blocked: bool
    drift_events: List[Dict[str, Any]]


class EstimateCostRequest(BaseModel):
    prompt: str

class EstimateCostResponse(BaseModel):
    estimated_tokens: int
    approx_chars: int


class ResolveContextRequest(BaseModel):
    prompt: str
    repo_root: Optional[str] = None

class ResolveContextResponse(BaseModel):
    spans: List[Dict[str, Any]]
    files_included: List[str]
    total_lines: int
    total_token_cost: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(
    req: OptimizeRequest,
    optimizer: Optimizer = Depends(get_optimizer),
):
    raw = RawPrompt(
        text=req.prompt,
        source_agent=req.source_agent,
        repo_root=req.repo_root,
    )
    result = optimizer.optimize(prompt=raw, strict_mode=req.strict_mode)
    opt = result.optimized_prompt
    cls = result.classification
    il = result.intent_lock

    return OptimizeResponse(
        prompt_id=raw.prompt_id,
        optimization_id=opt.optimization_id,
        optimized_text=opt.text,
        original_text=req.prompt,
        token_estimate=opt.token_estimate,
        semantic_similarity=opt.semantic_similarity,
        compression_ratio=opt.compression_ratio,
        category=cls.primary_category.value if cls else "unknown",
        intent=il.intent_summary if il else "",
        context_spans=opt.context_spans,
        injected_constraints=opt.injected_constraints,
        success=result.success,
        warnings=result.warnings,
        policy_violations=result.policy_violations,
        pipeline_duration_ms=result.pipeline_duration_ms,
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(
    req: ClassifyRequest,
    optimizer: Optimizer = Depends(get_optimizer),
):
    cls = optimizer.classify(req.prompt)
    return ClassifyResponse(
        primary_category=cls.primary_category.value,
        primary_confidence=cls.primary_confidence,
        complexity_score=cls.complexity_score,
        has_stacktrace=cls.has_stacktrace,
        has_logs=cls.has_logs,
        has_code_snippet=cls.has_code_snippet,
        extracted_files=cls.extracted_file_paths,
        extracted_symbols=cls.extracted_symbols,
        multi_file=cls.multi_file,
    )


@app.post("/validate", response_model=ValidateResponse)
def validate(
    req: ValidateRequest,
    optimizer: Optimizer = Depends(get_optimizer),
):
    result = optimizer.validate(req.raw_text, req.optimized_text)
    return ValidateResponse(
        passed=result.passed,
        semantic_similarity=result.semantic_similarity,
        threshold=result.threshold,
        method=result.method,
        failure_reason=result.failure_reason,
    )


@app.post("/detect-drift", response_model=DriftResponse)
def detect_drift(
    req: DriftRequest,
    optimizer: Optimizer = Depends(get_optimizer),
):
    report = optimizer.detect_drift(req.raw_text, req.optimized_text)
    return DriftResponse(
        is_clean=report.is_clean,
        overall_severity=report.overall_severity,
        blocked=report.blocked,
        drift_events=[
            {
                "type": d.drift_type,
                "description": d.description,
                "severity": d.severity,
                "suggested_fix": d.suggested_fix,
            }
            for d in report.drifts_detected
        ],
    )


@app.post("/estimate-cost", response_model=EstimateCostResponse)
def estimate_cost(
    req: EstimateCostRequest,
    optimizer: Optimizer = Depends(get_optimizer),
):
    result = optimizer.estimate_cost(req.prompt)
    return EstimateCostResponse(**result)


@app.post("/resolve-context", response_model=ResolveContextResponse)
def resolve_context(
    req: ResolveContextRequest,
    optimizer: Optimizer = Depends(get_optimizer),
):
    raw = RawPrompt(text=req.prompt, repo_root=req.repo_root)
    intent_lock = optimizer._intent_guard.extract_and_lock(raw)
    classification = optimizer._classifier.classify(raw, intent_lock)
    spans = optimizer._context_resolver.resolve(intent_lock, classification, repo_root=req.repo_root)
    return ResolveContextResponse(
        spans=[
            {
                "file": s.file_path,
                "start_line": s.start_line,
                "end_line": s.end_line,
                "symbol": s.symbol,
                "confidence": s.confidence,
                "reason": s.reason,
            }
            for s in spans
        ],
        files_included=list({s.file_path for s in spans}),
        total_lines=sum(s.line_count() for s in spans),
        total_token_cost=sum(s.token_cost for s in spans),
    )


@app.get("/plugins")
def list_plugins(optimizer: Optimizer = Depends(get_optimizer)):
    return {"plugins": optimizer.list_plugins()}


@app.post("/benchmark")
def benchmark(optimizer: Optimizer = Depends(get_optimizer)):
    report = optimizer.benchmark()
    return {
        "summary": report.summarize(),
        "results": [
            {
                "case": r.case_name,
                "passed": r.passed,
                "overall_score": r.overall_score,
                "semantic_similarity": r.semantic_similarity,
                "token_reduction_pct": r.token_reduction_pct,
                "category_correct": r.category_correct,
                "errors": r.errors,
            }
            for r in report.results
        ],
    }
