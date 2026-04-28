"""IPC transport (Unix socket) for MCP server."""
import asyncio, json, os
from llm_prompt_optimizer.mcp_server.server import MCPServer

SOCKET_PATH = "/tmp/llm_prompt_optimizer.sock"

async def run():
    server = MCPServer()
    async def handle(reader, writer):
        data = await reader.read(65536)
        try:
            req = json.loads(data)
            resp = server._handle_request(req)
        except Exception as e:
            resp = {"error": str(e)}
        writer.write(json.dumps(resp).encode())
        await writer.drain()
        writer.close()

    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    srv = await asyncio.start_unix_server(handle, path=SOCKET_PATH)
    async with srv:
        await srv.serve_forever()

if __name__ == "__main__":
    asyncio.run(run())
