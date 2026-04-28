#!/usr/bin/env python3
"""
MCP Protocol Wrapper for llm-prompt-optimizer

This wrapper properly implements the MCP specification and bridges
it to the underlying tool implementation.
"""

import json
import sys
from typing import Any, Dict, List

from llm_prompt_optimizer import Optimizer, OptimizerConfig
from llm_prompt_optimizer.models.prompt import RawPrompt


class MCPWrapper:
    def __init__(self):
        self.optimizer = Optimizer(config=OptimizerConfig.from_env())
        self.initialized = False

    def handle_initialize(self, params: Dict) -> Dict:
        """Handle MCP initialize handshake."""
        self.initialized = True
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "llm-prompt-optimizer",
                "version": "0.1.0"
            }
        }

    def handle_tools_list(self) -> Dict:
        """List all available tools."""
        return {
            "tools": [
                {
                    "name": "optimize_prompt",
                    "description": "Optimize a prompt for clarity and efficiency",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "The prompt to optimize"},
                            "strict_mode": {"type": "boolean", "description": "Enable strict scope mode"},
                            "repo_root": {"type": "string", "description": "Repository root path"}
                        },
                        "required": ["prompt"]
                    }
                },
                {
                    "name": "classify_prompt",
                    "description": "Classify a prompt into task categories",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "The prompt to classify"}
                        },
                        "required": ["prompt"]
                    }
                },
                {
                    "name": "detect_prompt_drift",
                    "description": "Detect semantic drift between raw and optimized prompts",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "raw_text": {"type": "string"},
                            "optimized_text": {"type": "string"}
                        },
                        "required": ["raw_text", "optimized_text"]
                    }
                },
                {
                    "name": "estimate_prompt_cost",
                    "description": "Estimate token cost of a prompt",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"}
                        },
                        "required": ["prompt"]
                    }
                },
                {
                    "name": "resolve_precise_context",
                    "description": "Resolve precise context needed for a prompt",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "repo_root": {"type": "string"}
                        },
                        "required": ["prompt"]
                    }
                }
            ]
        }

    def handle_tools_call(self, name: str, arguments: Dict) -> Dict:
        """Execute a tool call."""
        try:
            if name == "optimize_prompt":
                result = self.optimizer.optimize(
                    prompt=arguments["prompt"],
                    strict_mode=arguments.get("strict_mode", False),
                    repo_root=arguments.get("repo_root")
                )
                opt = result.optimized_prompt
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "optimized_text": opt.text,
                            "token_estimate": opt.token_estimate,
                            "semantic_similarity": opt.semantic_similarity,
                            "compression_ratio": opt.compression_ratio
                        })
                    }]
                }
            elif name == "classify_prompt":
                cls = self.optimizer.classify(arguments["prompt"])
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "category": cls.primary_category.value,
                            "confidence": cls.primary_confidence,
                            "complexity": cls.complexity_score
                        })
                    }]
                }
            elif name == "detect_prompt_drift":
                report = self.optimizer.detect_drift(arguments["raw_text"], arguments["optimized_text"])
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "is_clean": report.is_clean,
                            "severity": report.overall_severity
                        })
                    }]
                }
            elif name == "estimate_prompt_cost":
                cost = self.optimizer.estimate_cost(arguments["prompt"])
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(cost)
                    }]
                }
            elif name == "resolve_precise_context":
                raw = RawPrompt(text=arguments["prompt"], repo_root=arguments.get("repo_root"))
                il = self.optimizer._intent_guard.extract_and_lock(raw)
                cls = self.optimizer._classifier.classify(raw, il)
                spans = self.optimizer._context_resolver.resolve(il, cls, repo_root=arguments.get("repo_root"))
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "spans_count": len(spans),
                            "spans": [{"file": s.file_path, "lines": f"{s.start_line}-{s.end_line}"} for s in spans[:10]]
                        })
                    }]
                }
            else:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Unknown tool: {name}"
                    }],
                    "isError": True
                }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: {str(e)}"
                }],
                "isError": True
            }

    def process_message(self, request: Dict) -> Dict:
        """Process an MCP request."""
        req_id = request.get("id")
        method = request.get("method")

        response = {
            "jsonrpc": "2.0",
        }

        if req_id is not None:
            response["id"] = req_id

        try:
            if method == "initialize":
                response["result"] = self.handle_initialize(request.get("params", {}))
            elif method == "tools/list":
                response["result"] = self.handle_tools_list()
            elif method == "tools/call":
                params = request.get("params", {})
                response["result"] = self.handle_tools_call(
                    params.get("name", ""),
                    params.get("arguments", {})
                )
            else:
                response["error"] = {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
        except Exception as e:
            response["error"] = {
                "code": -32603,
                "message": str(e)
            }

        return response


def main():
    wrapper = MCPWrapper()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            response = wrapper.process_message(request)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError as e:
            error = {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {e}"
                }
            }
            sys.stdout.write(json.dumps(error) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
