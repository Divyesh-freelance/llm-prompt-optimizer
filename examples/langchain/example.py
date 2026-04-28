"""
Example: LangChain Integration

Use LLM Prompt Optimizer as a preprocessing step in LangChain chains.
"""

from llm_prompt_optimizer import Optimizer, OptimizerConfig


def create_optimized_chain(raw_prompt: str, repo_root: str = None):
    """
    Optimizes a prompt before passing it to a LangChain LLM chain.
    
    In a real integration, you'd wire this into a LangChain RunnableLambda
    or a custom Tool.
    """
    optimizer = Optimizer(config=OptimizerConfig.from_env())
    result = optimizer.optimize(prompt=raw_prompt, repo_root=repo_root)

    if not result.success:
        raise ValueError(f"Optimization failed: {result.errors}")

    optimized_text = result.optimized_prompt.text
    print(f"[LPO] Optimized prompt ({result.optimized_prompt.token_estimate} tokens, "
          f"similarity={result.optimized_prompt.semantic_similarity:.3f})")
    return optimized_text


# LangChain usage pattern (requires langchain installed):
#
# from langchain_anthropic import ChatAnthropic
# from langchain_core.runnables import RunnableLambda
#
# lpo_step = RunnableLambda(lambda x: create_optimized_chain(x["prompt"]))
# llm = ChatAnthropic(model="claude-opus-4-5")
# chain = lpo_step | llm
# response = chain.invoke({"prompt": "Debug EMA mismatch in signals/IndexSignals.py"})

if __name__ == "__main__":
    optimized = create_optimized_chain(
        "Debug condition mismatch in the provided /path/repository/file No code changes."
    )
    print(optimized)
