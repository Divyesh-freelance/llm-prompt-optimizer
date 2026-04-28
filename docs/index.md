# LLM Prompt Optimizer — Documentation

See [README.md](../README.md) for the full getting-started guide.

## Architecture Decisions

### Why FallbackGraphEngine?
The system is designed standalone-first. External graph tools (Graphify, Code Review Graph)
enhance quality but are never required. The FallbackGraphEngine uses Python's stdlib `ast`
module for zero-dependency dependency discovery.

### Why value-based adaptive expansion?
Fixed limits like `max_files: 5` are arbitrary and context-blind. The value formula:
`(relevance × confidence × execution_proximity) / token_cost` ensures every expansion
decision is justified by marginal value, not arbitrary thresholds.

### Why IntentLock is immutable after sealing?
Downstream pipeline stages (classifier, optimizer, compiler) are not allowed to reinterpret
or expand user intent. Once IntentGuard seals the lock, no other component may modify it.
This is the primary hallucination-prevention mechanism.

### Why semantic similarity threshold of 0.90?
Empirically, prompts that preserve <90% semantic similarity with their original have
demonstrably different intent. This threshold is configurable via `PolicyConfig`.

## Component Reference

- [IntentGuard](../llm_prompt_optimizer/core/intent_guard/guard.py)
- [PromptClassifier](../llm_prompt_optimizer/core/classifier/classifier.py)
- [FallbackGraphEngine](../llm_prompt_optimizer/core/fallback_graph/engine.py)
- [AdaptiveContextExpansion](../llm_prompt_optimizer/core/adaptive_context_expansion/expansion.py)
- [PreciseContextResolver](../llm_prompt_optimizer/core/precise_context/resolver.py)
- [PromptOptimizer](../llm_prompt_optimizer/core/optimizer/optimizer.py)
- [SemanticValidator](../llm_prompt_optimizer/core/semantic_validator/validator.py)
- [DriftDetector](../llm_prompt_optimizer/core/drift_detection/detector.py)
- [PolicyEngine](../llm_prompt_optimizer/core/policy/engine.py)
- [TokenBudgetEngine](../llm_prompt_optimizer/core/token_budget/engine.py)
- [MCPServer](../llm_prompt_optimizer/mcp_server/server.py)
- [PluginSystem](../llm_prompt_optimizer/plugins/system.py)
- [Optimizer SDK](../llm_prompt_optimizer/sdk/optimizer.py)
- [REST API](../llm_prompt_optimizer/api/app.py)
- [CLI](../llm_prompt_optimizer/cli.py)
