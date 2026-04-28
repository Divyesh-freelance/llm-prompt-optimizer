"""HTTP transport for MCP server."""
import asyncio
from llm_prompt_optimizer.mcp_server.server import MCPServer

async def run(host="127.0.0.1", port=8765):
    server = MCPServer()
    await server.run_http(host=host, port=port)

if __name__ == "__main__":
    asyncio.run(run())
