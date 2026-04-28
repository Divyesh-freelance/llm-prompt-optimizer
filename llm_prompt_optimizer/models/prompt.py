"""Prompt-related data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid


@dataclass
class RawPrompt:
    """Represents an unprocessed user prompt."""

    text: str
    prompt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    source_agent: Optional[str] = None
    repo_root: Optional[str] = None
    working_directory: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("Prompt text cannot be empty.")


@dataclass
class OptimizedPrompt:
    """Represents a processed, optimized prompt ready for LLM consumption."""

    text: str
    original_prompt_id: str
    optimization_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token_estimate: int = 0
    semantic_similarity: float = 1.0
    context_spans: List[Dict[str, Any]] = field(default_factory=list)
    injected_constraints: List[str] = field(default_factory=list)
    compression_ratio: float = 1.0
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptOptimizationResult:
    """Full result of a prompt optimization pipeline run."""

    raw_prompt: RawPrompt
    optimized_prompt: OptimizedPrompt
    classification: Any = None
    intent_lock: Any = None
    context_bundle: Any = None
    validation_result: Any = None
    drift_report: Any = None
    telemetry: Any = None
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    pipeline_duration_ms: float = 0.0
    token_savings: int = 0
    policy_violations: List[str] = field(default_factory=list)
