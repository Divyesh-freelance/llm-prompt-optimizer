"""
Example: Claude MCP Integration

Demonstrates how to configure the LLM Prompt Optimizer as an MCP server
for Claude Desktop / Cursor / Continue.dev.

claude_desktop_config.json:
{
  "mcpServers": {
    "llm-prompt-optimizer": {
      "command": "llm-prompt-optimizer",
      "args": ["serve", "--transport", "stdio"],
      "env": {
        "LPO_STRICT_INTENT": "true",
        "LPO_SEMANTIC_THRESHOLD": "0.90",
        "LPO_LOG_LEVEL": "WARNING"
      }
    }
  }
}
"""

from llm_prompt_optimizer import Optimizer, OptimizerConfig
from llm_prompt_optimizer.mcp_server.server import MCPServer


def run_mcp_stdio():
    """Start the MCP server in stdio mode for Claude Desktop."""
    server = MCPServer()
    print("MCP server ready (stdio). Waiting for tool calls...", flush=True)
    server.run_stdio_sync()


def demo_sdk_usage():
    """Demonstrate SDK usage before sending to Claude."""
    cfg = OptimizerConfig()
    optimizer = Optimizer(config=cfg)

    # Raw prompt from user
    raw_prompt = "Debug EMA mismatch in signals/IndexSignals.py. No code changes needed."

    # Optimize before sending to Claude
    result = optimizer.optimize(
        prompt=raw_prompt,
        strict_mode=True,
        repo_root="/path/to/your/repo",
    )

    print("=" * 60)
    print("ORIGINAL:")
    print(raw_prompt)
    print("\nOPTIMIZED (send this to Claude):")
    print(result.optimized_prompt.text)
    print("\nMETRICS:")
    print(f"  Category:    {result.classification.primary_category.value}")
    print(f"  Intent:      {result.intent_lock.intent_summary}")
    print(f"  Tokens:      {result.optimized_prompt.token_estimate}")
    print(f"  Similarity:  {result.optimized_prompt.semantic_similarity:.3f}")
    print(f"  Drift clean: {result.drift_report.is_clean}")
    print(f"  Success:     {result.success}")


if __name__ == "__main__":
    demo_sdk_usage()
