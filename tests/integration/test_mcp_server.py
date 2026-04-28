"""Integration tests for the MCP server handler."""

import pytest
from llm_prompt_optimizer.mcp_server.server import MCPServer, MCPHandler
from llm_prompt_optimizer import Optimizer, OptimizerConfig


@pytest.fixture(scope="module")
def server():
    cfg = OptimizerConfig()
    cfg.policy.enable_audit_log = False
    s = MCPServer(config=cfg)
    return s


class TestMCPHandler:

    def test_optimize_prompt_tool(self, server):
        response = server._handle_request({
            "tool": "optimize_prompt",
            "params": {"prompt": "Debug EMA mismatch in signals/IndexSignals.py"},
        })
        assert response["status"] == "success"
        assert "optimized_text" in response["result"]

    def test_classify_prompt_tool(self, server):
        response = server._handle_request({
            "tool": "classify_prompt",
            "params": {"prompt": "Debug EMA mismatch in signals/IndexSignals.py"},
        })
        assert response["status"] == "success"
        assert response["result"]["primary_category"] == "debugging"

    def test_estimate_cost_tool(self, server):
        response = server._handle_request({
            "tool": "estimate_prompt_cost",
            "params": {"prompt": "Debug EMA mismatch"},
        })
        assert response["status"] == "success"
        assert response["result"]["estimated_tokens"] > 0

    def test_validate_intent_tool(self, server):
        raw = "Debug EMA mismatch"
        response = server._handle_request({
            "tool": "validate_intent",
            "params": {"raw_text": raw, "optimized_text": raw},
        })
        assert response["status"] == "success"
        assert response["result"]["passed"] is True

    def test_detect_drift_tool(self, server):
        raw = "Debug EMA mismatch in signals/IndexSignals.py"
        response = server._handle_request({
            "tool": "detect_prompt_drift",
            "params": {"raw_text": raw, "optimized_text": raw},
        })
        assert response["status"] == "success"
        assert response["result"]["is_clean"] is True

    def test_unknown_tool_returns_error(self, server):
        response = server._handle_request({
            "tool": "nonexistent_tool",
            "params": {},
        })
        assert response["status"] == "error"

    def test_tool_schema_has_all_tools(self, server):
        schema = server.get_tool_schema()
        tool_names = [t["name"] for t in schema["tools"]]
        expected = [
            "optimize_prompt", "classify_prompt", "resolve_precise_context",
            "discover_dependencies", "detect_prompt_drift", "estimate_prompt_cost",
            "compress_context", "validate_intent", "benchmark_prompt",
        ]
        for name in expected:
            assert name in tool_names

    def test_request_id_passthrough(self, server):
        response = server._handle_request({
            "tool": "classify_prompt",
            "params": {"prompt": "Debug EMA mismatch"},
            "id": "test-123",
        })
        assert response.get("id") == "test-123"

    def test_compress_context_tool(self, server):
        response = server._handle_request({
            "tool": "compress_context",
            "params": {"prompt": "Debug EMA mismatch in signals/IndexSignals.py"},
        })
        assert response["status"] == "success"
        assert "compressed_text" in response["result"]
