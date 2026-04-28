"""Classification-related data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class TaskCategory(str, Enum):
    DEBUGGING = "debugging"
    IMPLEMENTATION = "implementation"
    REFACTORING = "refactoring"
    OPTIMIZATION = "optimization"
    ARCHITECTURE = "architecture"
    API_INTEGRATION = "api_integration"
    BACKEND = "backend"
    FRONTEND = "frontend"
    DATABASE = "database"
    TESTING = "testing"
    DEVOPS = "devops"
    CICD = "ci_cd"
    MIGRATION = "migration"
    CLOUD = "cloud"
    INFRASTRUCTURE = "infrastructure"
    MONITORING = "monitoring"
    ML_AI = "ml_ai"
    OBSERVABILITY = "observability"
    PROMPT_ENGINEERING = "prompt_engineering"
    CONCURRENCY = "concurrency"
    ASYNC_WORKFLOWS = "async_workflows"
    INCIDENT_ANALYSIS = "incident_analysis"
    LOG_ANALYSIS = "log_analysis"
    SECURITY = "security"
    DEPENDENCY_RESOLUTION = "dependency_resolution"
    REPO_EXPLORATION = "repo_exploration"
    UNKNOWN = "unknown"


@dataclass
class ClassificationLabel:
    category: TaskCategory
    confidence: float
    method: str  # embedding | nlp | syntax | heuristic


@dataclass
class PromptClassification:
    """Multi-label classification result for a prompt."""

    primary_category: TaskCategory
    primary_confidence: float
    labels: List[ClassificationLabel] = field(default_factory=list)
    complexity_score: float = 0.5  # 0.0 (trivial) → 1.0 (highly complex)
    has_stacktrace: bool = False
    has_logs: bool = False
    has_code_snippet: bool = False
    extracted_file_paths: List[str] = field(default_factory=list)
    extracted_symbols: List[str] = field(default_factory=list)
    requires_execution_path: bool = False
    multi_file: bool = False
    classifier_version: str = "1.0.0"
    metadata: Dict = field(default_factory=dict)

    def top_categories(self, n: int = 3) -> List[ClassificationLabel]:
        return sorted(self.labels, key=lambda x: x.confidence, reverse=True)[:n]
