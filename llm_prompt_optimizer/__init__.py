"""
LLM Prompt Optimizer
====================
A universal, deterministic middleware layer for AI coding agents.

Quick start:
    from llm_prompt_optimizer import Optimizer

    optimizer = Optimizer()
    result = optimizer.optimize("debug EMA mismatch in signals/IndexSignals.py")
    print(result.optimized_prompt.text)
"""

from llm_prompt_optimizer.sdk.optimizer import Optimizer
from llm_prompt_optimizer.models.prompt import RawPrompt, OptimizedPrompt, PromptOptimizationResult
from llm_prompt_optimizer.config.settings import OptimizerConfig
from llm_prompt_optimizer.plugins.system import PluginSystem

__version__ = "0.1.0"
__author__ = "LLM Prompt Optimizer Contributors"

__all__ = [
    "Optimizer",
    "RawPrompt",
    "OptimizedPrompt",
    "PromptOptimizationResult",
    "OptimizerConfig",
    "PluginSystem",
]
