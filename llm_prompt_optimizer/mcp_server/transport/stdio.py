"""stdio transport for MCP server."""
import sys, json, asyncio
from llm_prompt_optimizer.mcp_server.server import MCPServer

def run():
    server = MCPServer()
    server.run_stdio_sync()

if __name__ == "__main__":
    run()
