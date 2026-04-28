"""Configuration management for LLM Prompt Optimizer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class PolicyConfig:
    strict_intent_mode: bool = True
    adaptive_context_budget: bool = True
    allow_scope_expansion: str = "controlled"  # strict | controlled | open
    semantic_similarity_threshold: float = 0.90
    require_dependency_validation: bool = True
    max_policy_violations_before_block: int = 1
    enable_drift_detection: bool = True
    enable_telemetry: bool = True
    enable_audit_log: bool = True


@dataclass
class TokenBudgetConfig:
    default_budget_tokens: int = 8000
    max_budget_tokens: int = 32000
    min_budget_tokens: int = 500
    reserve_tokens: int = 200  # reserved for system/compiler overhead
    adaptive_budgeting: bool = True
    entropy_threshold: float = 0.05  # stop expanding if marginal value < this


@dataclass
class AdaptiveExpansionConfig:
    confidence_gain_threshold: float = 0.05  # stop if gain < this per iteration
    execution_relevance_min: float = 0.3
    semantic_confidence_min: float = 0.4
    max_iterations: int = 20  # hard safety cap (not a fixed limit, just a guard)
    value_formula_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "relevance": 1.0,
            "confidence": 1.0,
            "execution_proximity": 1.2,
        }
    )


@dataclass
class MCPServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    transport: str = "stdio"  # stdio | websocket | http | ipc
    enable_daemon: bool = False
    auth_token: Optional[str] = None
    max_connections: int = 50
    timeout_seconds: int = 30


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"  # json | text
    output: str = "stdout"  # stdout | file
    file_path: Optional[str] = None
    enable_structured: bool = True


@dataclass
class OptimizerConfig:
    """Root configuration object for the entire optimizer system."""

    policy: PolicyConfig = field(default_factory=PolicyConfig)
    token_budget: TokenBudgetConfig = field(default_factory=TokenBudgetConfig)
    adaptive_expansion: AdaptiveExpansionConfig = field(default_factory=AdaptiveExpansionConfig)
    mcp_server: MCPServerConfig = field(default_factory=MCPServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    plugin_dirs: list = field(default_factory=list)
    cache_dir: str = str(Path.home() / ".llm_prompt_optimizer" / "cache")
    telemetry_backend: Optional[str] = None  # None | "opentelemetry" | "prometheus"
    graph_provider: Optional[str] = None  # None | "graphify" | "code_review_graph"
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "OptimizerConfig":
        """Load configuration from environment variables."""
        config = cls()
        config.policy.strict_intent_mode = (
            os.getenv("LPO_STRICT_INTENT", "true").lower() == "true"
        )
        config.policy.semantic_similarity_threshold = float(
            os.getenv("LPO_SEMANTIC_THRESHOLD", "0.90")
        )
        config.token_budget.default_budget_tokens = int(
            os.getenv("LPO_TOKEN_BUDGET", "8000")
        )
        config.mcp_server.host = os.getenv("LPO_MCP_HOST", "127.0.0.1")
        config.mcp_server.port = int(os.getenv("LPO_MCP_PORT", "8765"))
        config.mcp_server.transport = os.getenv("LPO_MCP_TRANSPORT", "stdio")
        config.logging.level = os.getenv("LPO_LOG_LEVEL", "INFO")
        config.graph_provider = os.getenv("LPO_GRAPH_PROVIDER", None)
        return config
