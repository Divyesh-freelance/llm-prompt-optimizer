"""
MCP Server — Exposes optimizer tools via Model Context Protocol.

Tools exposed:
  optimize_prompt
  classify_prompt
  resolve_precise_context
  discover_dependencies
  detect_prompt_drift
  estimate_prompt_cost
  compress_context
  validate_intent
  benchmark_prompt

Transports:
  stdio (default)
  websocket
  HTTP
  IPC
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys
from typing import Any, Dict, Optional

from llm_prompt_optimizer import Optimizer, OptimizerConfig
from llm_prompt_optimizer.models.prompt import RawPrompt
from llm_prompt_optimizer.utils.helpers import estimate_tokens, get_logger

logger = get_logger(__name__)


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back on parse error."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Bad {name}={raw!r}; using default {default}")
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"Bad {name}={raw!r}; using default {default}")
        return default


# ── Configuration knobs (read at process start) ──────────────────────────────
# Override via env vars in your MCP host config.

# Hard ceiling on optimize_prompt latency. On timeout we return the raw
# prompt + a warning so the agent's fallback clause activates instead of
# the user waiting indefinitely.
OPTIMIZE_TIMEOUT_S: float = _env_float("LPO_OPTIMIZE_TIMEOUT_S", 10.0)

# Prompts shorter than this (token estimate) auto-enable fast_mode:
# skip adaptive expansion and embedding-based validation, use n-gram only.
FAST_MODE_THRESHOLD_TOKENS: int = _env_int("LPO_FAST_MODE_THRESHOLD_TOKENS", 50)

# Display gating for the agent's show-and-approve flow:
#   "always"               — show optimized text every time, ask to proceed
#   "on-significant-change" (default) — only ask when change is non-trivial
#   "never"                — never block; agent uses optimized text silently
DISPLAY_MODE: str = os.environ.get("LPO_DISPLAY_MODE", "on-significant-change").lower()

# A single shared executor so we don't churn threads per call. Worker count
# is 1 because the optimizer is heavily CPU-bound and concurrent calls would
# just contend for the GIL and the shared SentenceTransformer model.
_OPTIMIZE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="lpo-optimize"
)


# ── Response shaping helpers (timeout, error, display directive) ─────────────


def _timeout_response(prompt_text: str, timeout_s: float) -> Dict[str, Any]:
    """
    Return when the optimizer pipeline exceeds its timeout. Mirrors the success
    response shape so the agent can use `optimized_text` (= raw prompt) as a
    drop-in. The `timed_out` and `display_directive` fields tell the agent
    that the rule's fallback clause should activate.
    """
    return {
        "optimized_text": prompt_text,
        "raw_prompt": prompt_text,
        "token_estimate": estimate_tokens(prompt_text),
        "semantic_similarity": 1.0,
        "compression_ratio": 1.0,
        "success": False,
        "timed_out": True,
        "warning": (
            f"Optimization exceeded {timeout_s:g}s and was abandoned. "
            "Proceeding with the user's original message."
        ),
        "context_spans_count": 0,
        "policy_violations": [],
        "fast_mode_used": False,
        "display_mode": DISPLAY_MODE,
        "requires_user_approval": False,
        "change_summary": {"reason": "timeout"},
        "display_directive": (
            "The prompt optimizer timed out. Per the fallback policy, proceed "
            "normally with the user's original message. Do not retry, do not "
            "block, do not surface the failure unless the user asks."
        ),
    }


def _error_response(prompt_text: str, error: str) -> Dict[str, Any]:
    """Same shape as timeout response but for non-timeout failures."""
    return {
        "optimized_text": prompt_text,
        "raw_prompt": prompt_text,
        "token_estimate": estimate_tokens(prompt_text),
        "semantic_similarity": 1.0,
        "compression_ratio": 1.0,
        "success": False,
        "timed_out": False,
        "error": error,
        "context_spans_count": 0,
        "policy_violations": [],
        "fast_mode_used": False,
        "display_mode": DISPLAY_MODE,
        "requires_user_approval": False,
        "change_summary": {"reason": "error", "detail": error},
        "display_directive": (
            "The prompt optimizer failed. Per the fallback policy, proceed "
            "normally with the user's original message."
        ),
    }


def _summarize_change(
    *,
    raw: str,
    optimized_text: str,
    token_estimate_raw: int,
    token_estimate_optimized: int,
    similarity: float,
    context_spans_count: int,
    drift_clean: bool,
) -> Dict[str, Any]:
    """
    Build a small structured summary of what changed between raw and
    optimized prompts. Used both for logging and for the agent's display.
    """
    delta = token_estimate_optimized - token_estimate_raw
    return {
        "tokens_before": token_estimate_raw,
        "tokens_after": token_estimate_optimized,
        "tokens_delta": delta,
        "similarity": round(similarity, 3),
        "context_spans_added": context_spans_count,
        "drift_clean": drift_clean,
        "trivial": (
            similarity >= 0.95
            and context_spans_count == 0
            and drift_clean
            and abs(delta) <= max(15, token_estimate_raw // 4)
        ),
    }


def _requires_user_approval(
    *,
    similarity: float,
    change_summary: Dict[str, Any],
    drift_clean: bool,
    display_mode: str,
) -> bool:
    """
    Decide whether the agent should pause and ask the user before acting on
    the optimized prompt. Honors LPO_DISPLAY_MODE.
    """
    if display_mode == "never":
        return False
    if display_mode == "always":
        return True
    # default: on-significant-change
    if not drift_clean:
        return True
    if similarity < 0.95:
        return True
    if change_summary.get("context_spans_added", 0) > 0:
        return True
    if not change_summary.get("trivial", False):
        return True
    return False


def _build_display_directive(
    *,
    requires_approval: bool,
    change_summary: Dict[str, Any],
    optimized_text: str,
    display_mode: str,
) -> str:
    """
    Concrete instruction for the agent. Kept short — the agent has limited
    context budget for tool results, so we want this to read cleanly and
    push toward a deterministic UX.
    """
    if display_mode == "never":
        return (
            "Use `optimized_text` as your working understanding silently. "
            "Do not show it to the user."
        )
    if not requires_approval:
        # Trivial change: brief mention, no gating.
        return (
            "Briefly mention to the user that you optimized their prompt "
            f"(tokens {change_summary['tokens_before']} → "
            f"{change_summary['tokens_after']}, similarity "
            f"{change_summary['similarity']}). Then proceed using "
            "`optimized_text` as your working understanding."
        )
    # Significant change — gate on user approval.
    return (
        "BEFORE doing any other work, show the user the `optimized_text` "
        "verbatim in a fenced markdown block, followed by a one-line summary "
        f"(tokens {change_summary['tokens_before']} → "
        f"{change_summary['tokens_after']}, similarity "
        f"{change_summary['similarity']}, "
        f"{change_summary['context_spans_added']} context spans added). "
        "Then ask the user, exactly: \"Proceed with this optimized "
        "interpretation, use my original prompt, or revise?\" and WAIT for "
        "their answer. Do not call any other tool, do not write any code, "
        "do not begin reasoning about the task until they reply. If they "
        "say 'use original' or similar, discard `optimized_text` and act on "
        "`raw_prompt` instead."
    )


class MCPToolResult:
    """Wraps a tool call result into MCP-compatible format."""

    @staticmethod
    def success(data: Any) -> Dict[str, Any]:
        return {"status": "success", "result": data}

    @staticmethod
    def error(message: str) -> Dict[str, Any]:
        return {"status": "error", "error": message}


class MCPHandler:
    """Handles individual MCP tool calls."""

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def dispatch(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        handlers = {
            "optimize_prompt": self._optimize_prompt,
            "classify_prompt": self._classify_prompt,
            "resolve_precise_context": self._resolve_precise_context,
            "discover_dependencies": self._discover_dependencies,
            "detect_prompt_drift": self._detect_prompt_drift,
            "estimate_prompt_cost": self._estimate_prompt_cost,
            "compress_context": self._compress_context,
            "validate_intent": self._validate_intent,
            "benchmark_prompt": self._benchmark_prompt,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return MCPToolResult.error(f"Unknown tool: {tool_name}")
        try:
            return MCPToolResult.success(handler(params))
        except Exception as e:
            logger.error(f"MCP tool '{tool_name}' error: {e}")
            return MCPToolResult.error(str(e))

    def _optimize_prompt(self, p: Dict) -> Dict:
        """
        Run the full optimization pipeline with three production safeguards:

          1. Hard timeout (LPO_OPTIMIZE_TIMEOUT_S): on timeout the raw prompt
             is returned with a warning, so the agent's fallback clause kicks
             in instead of the user waiting indefinitely.

          2. fast_mode auto-trigger: prompts shorter than
             LPO_FAST_MODE_THRESHOLD_TOKENS skip adaptive expansion and
             embedding similarity, dropping latency from seconds to
             milliseconds for casual prompts.

          3. display_directive + requires_user_approval fields: the agent
             surfaces the optimized text to the user and asks for approval
             before acting on it, gated by LPO_DISPLAY_MODE.
        """
        prompt_text: str = p["prompt"]
        strict_mode: bool = bool(p.get("strict_mode", False))
        repo_root: Optional[str] = p.get("repo_root")

        # Caller can force fast_mode; otherwise auto-enable for short prompts.
        fast_mode: bool = bool(p.get("fast_mode", False))
        token_est = estimate_tokens(prompt_text)
        if not fast_mode and token_est < FAST_MODE_THRESHOLD_TOKENS:
            fast_mode = True

        future = _OPTIMIZE_EXECUTOR.submit(
            self._run_optimize_blocking,
            prompt_text,
            strict_mode,
            repo_root,
            fast_mode,
        )

        try:
            result = future.result(timeout=OPTIMIZE_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            logger.warning(
                f"optimize_prompt exceeded {OPTIMIZE_TIMEOUT_S}s — returning raw prompt"
            )
            # Note: we do NOT cancel the future. Cancelling a running thread
            # in Python is unsafe; the orphaned work just completes and is
            # discarded. The next call picks up its (now-warm) caches.
            return _timeout_response(prompt_text, OPTIMIZE_TIMEOUT_S)
        except Exception as e:
            logger.exception(f"optimize_prompt failed: {e}")
            return _error_response(prompt_text, str(e))

        # Pipeline succeeded — assemble the response with display-flow fields.
        opt = result.optimized_prompt
        change_summary = _summarize_change(
            raw=prompt_text,
            optimized_text=opt.text,
            token_estimate_raw=token_est,
            token_estimate_optimized=opt.token_estimate,
            similarity=opt.semantic_similarity,
            context_spans_count=len(opt.context_spans),
            drift_clean=(result.drift_report.is_clean if result.drift_report else True),
        )
        requires_approval = _requires_user_approval(
            similarity=opt.semantic_similarity,
            change_summary=change_summary,
            drift_clean=(result.drift_report.is_clean if result.drift_report else True),
            display_mode=DISPLAY_MODE,
        )
        display_directive = _build_display_directive(
            requires_approval=requires_approval,
            change_summary=change_summary,
            optimized_text=opt.text,
            display_mode=DISPLAY_MODE,
        )

        return {
            "optimized_text": opt.text,
            "token_estimate": opt.token_estimate,
            "semantic_similarity": opt.semantic_similarity,
            "compression_ratio": opt.compression_ratio,
            "success": result.success,
            "context_spans_count": len(opt.context_spans),
            "policy_violations": result.policy_violations,
            # show-and-approve fields
            "fast_mode_used": fast_mode,
            "display_mode": DISPLAY_MODE,
            "requires_user_approval": requires_approval,
            "change_summary": change_summary,
            "display_directive": display_directive,
            "raw_prompt": prompt_text,  # so agent can fall back on user's request
        }

    def _run_optimize_blocking(
        self,
        prompt_text: str,
        strict_mode: bool,
        repo_root: Optional[str],
        fast_mode: bool,
    ):
        """The actual blocking call. Runs in the executor thread."""
        return self.optimizer.optimize(
            prompt=prompt_text,
            strict_mode=strict_mode or fast_mode,
            repo_root=repo_root,
            fast_mode=fast_mode,
        )

    def _classify_prompt(self, p: Dict) -> Dict:
        cls = self.optimizer.classify(p["prompt"])
        return {
            "primary_category": cls.primary_category.value,
            "confidence": cls.primary_confidence,
            "complexity": cls.complexity_score,
            "has_stacktrace": cls.has_stacktrace,
            "extracted_files": cls.extracted_file_paths,
            "extracted_symbols": cls.extracted_symbols,
        }

    def _resolve_precise_context(self, p: Dict) -> Dict:
        raw = RawPrompt(text=p["prompt"], repo_root=p.get("repo_root"))
        il = self.optimizer._intent_guard.extract_and_lock(raw)
        cls = self.optimizer._classifier.classify(raw, il)
        spans = self.optimizer._context_resolver.resolve(il, cls, repo_root=p.get("repo_root"))
        return {
            "spans": [
                {
                    "file": s.file_path,
                    "start_line": s.start_line,
                    "end_line": s.end_line,
                    "symbol": s.symbol,
                    "confidence": s.confidence,
                    "reason": s.reason,
                    "token_cost": s.token_cost,
                }
                for s in spans
            ]
        }

    def _discover_dependencies(self, p: Dict) -> Dict:
        raw = RawPrompt(text=p["prompt"], repo_root=p.get("repo_root"))
        il = self.optimizer._intent_guard.extract_and_lock(raw)
        cls = self.optimizer._classifier.classify(raw, il)
        nodes = self.optimizer._dep_resolver.resolve(il, cls)
        return {
            "dependencies": [
                {
                    "file": n.file_path,
                    "type": n.dependency_type,
                    "depth": n.depth,
                    "confidence": n.confidence,
                    "method": n.discovery_method,
                }
                for n in nodes
            ]
        }

    def _detect_prompt_drift(self, p: Dict) -> Dict:
        report = self.optimizer.detect_drift(p["raw_text"], p["optimized_text"])
        return {
            "is_clean": report.is_clean,
            "severity": report.overall_severity,
            "blocked": report.blocked,
            "events": [
                {"type": d.drift_type, "severity": d.severity, "description": d.description}
                for d in report.drifts_detected
            ],
        }

    def _estimate_prompt_cost(self, p: Dict) -> Dict:
        return self.optimizer.estimate_cost(p["prompt"])

    def _compress_context(self, p: Dict) -> Dict:
        # Lightweight context compression: estimate reduction via optimization
        result = self.optimizer.optimize(p["prompt"], strict_mode=True)
        return {
            "compressed_text": result.optimized_prompt.text,
            "token_estimate": result.optimized_prompt.token_estimate,
            "compression_ratio": result.optimized_prompt.compression_ratio,
        }

    def _validate_intent(self, p: Dict) -> Dict:
        validation = self.optimizer.validate(p["raw_text"], p["optimized_text"])
        return {
            "passed": validation.passed,
            "similarity": validation.semantic_similarity,
            "threshold": validation.threshold,
            "failure_reason": validation.failure_reason,
        }

    def _benchmark_prompt(self, p: Dict) -> Dict:
        report = self.optimizer.benchmark()
        summary = report.summarize()
        return {"summary": summary}


class MCPServer:
    """
    MCP server supporting multiple transport modes.

    Usage (stdio):
        server = MCPServer()
        await server.run_stdio()

    Usage (HTTP):
        server = MCPServer()
        await server.run_http(host="127.0.0.1", port=8765)
    """

    def __init__(self, config: Optional[OptimizerConfig] = None) -> None:
        self.config = config or OptimizerConfig.from_env()
        self.optimizer = Optimizer(config=self.config)
        self.handler = MCPHandler(self.optimizer)

    # ── stdio transport ───────────────────────────────────────────────────────

    async def run_stdio(self) -> None:
        """Run MCP server over stdio (default for Claude Desktop / Cursor)."""
        logger.info("MCP server started (stdio transport)")
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        writer_transport, writer_protocol = await loop.connect_write_pipe(
            lambda: asyncio.BaseProtocol(), sys.stdout
        )

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line.decode().strip())
                response = self._handle_request(request)
                response_line = json.dumps(response) + "\n"
                sys.stdout.write(response_line)
                sys.stdout.flush()
            except json.JSONDecodeError as e:
                sys.stdout.write(json.dumps({"error": f"Invalid JSON: {e}"}) + "\n")
                sys.stdout.flush()
            except EOFError:
                break
            except Exception as e:
                logger.error(f"stdio error: {e}")
                break

    def run_stdio_sync(self) -> None:
        """Synchronous stdio loop for simple integrations."""
        logger.info("MCP server started (stdio sync)")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self._handle_request(request)
            except json.JSONDecodeError as e:
                response = {"error": f"Invalid JSON: {e}"}
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

    # ── HTTP transport ────────────────────────────────────────────────────────

    async def run_http(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        """Run MCP server over HTTP using uvicorn + FastAPI."""
        try:
            import uvicorn
            from llm_prompt_optimizer.api.app import app
            logger.info(f"MCP HTTP server on {host}:{port}")
            config = uvicorn.Config(app, host=host, port=port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()
        except ImportError:
            logger.error("uvicorn not installed. Run: pip install uvicorn fastapi")
            raise

    # ── WebSocket transport ───────────────────────────────────────────────────

    async def run_websocket(self, host: str = "127.0.0.1", port: int = 8766) -> None:
        """Run MCP server over WebSocket."""
        try:
            import websockets

            async def handle_ws(websocket, path):
                async for message in websocket:
                    try:
                        request = json.loads(message)
                        response = self._handle_request(request)
                    except json.JSONDecodeError as e:
                        response = {"error": str(e)}
                    await websocket.send(json.dumps(response))

            logger.info(f"MCP WebSocket server on ws://{host}:{port}")
            async with websockets.serve(handle_ws, host, port):
                await asyncio.Future()  # run forever
        except ImportError:
            logger.error("websockets not installed. Run: pip install websockets")
            raise

    # ── Core dispatch ─────────────────────────────────────────────────────────

    def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single MCP JSON-RPC-style request.
        Expected format: {"tool": "optimize_prompt", "params": {...}}
        """
        tool = request.get("tool") or request.get("method", "")
        params = request.get("params") or request.get("arguments", {})
        request_id = request.get("id")

        result = self.handler.dispatch(tool, params)

        response = result
        if request_id is not None:
            response["id"] = request_id
        return response

    def get_tool_schema(self) -> Dict[str, Any]:
        """Return MCP-compatible tool schema for all exposed tools."""
        return {
            "tools": [
                {
                    "name": "optimize_prompt",
                    "description": (
                        "REQUIRED FIRST STEP for any coding, debugging, refactoring, "
                        "code-review, or codebase-modification request. Call this with "
                        "the user's verbatim message BEFORE any other tool call or "
                        "before formulating an answer. "
                        "The response contains `optimized_text`, `raw_prompt`, "
                        "`requires_user_approval`, and `display_directive`. You MUST "
                        "follow `display_directive` exactly — when "
                        "`requires_user_approval` is true, show the user "
                        "`optimized_text` and wait for them to choose Proceed / Use "
                        "original / Revise before doing anything else. When false, "
                        "use `optimized_text` as your working understanding (briefly "
                        "noting the change if non-trivial). "
                        "FALLBACK POLICY: if `success` is false, `timed_out` is true, "
                        "the tool errors, or it isn't registered, proceed normally "
                        "with the user's original message. Do not retry, do not "
                        "block. "
                        "SKIP for small-talk, simple factual questions, or non-code "
                        "tasks."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The user's verbatim message.",
                            },
                            "strict_mode": {
                                "type": "boolean",
                                "default": False,
                                "description": "Disallow scope expansion beyond stated files.",
                            },
                            "repo_root": {
                                "type": "string",
                                "description": "Absolute path to the repository root, if known.",
                            },
                            "fast_mode": {
                                "type": "boolean",
                                "default": False,
                                "description": (
                                    "Skip adaptive expansion + embedding "
                                    "validation for low latency. Auto-enabled "
                                    "for short prompts."
                                ),
                            },
                        },
                        "required": ["prompt"],
                    },
                },
                {
                    "name": "classify_prompt",
                    "description": "Classify a prompt into task categories with confidence scores.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"prompt": {"type": "string"}},
                        "required": ["prompt"],
                    },
                },
                {
                    "name": "resolve_precise_context",
                    "description": "Extract precise line-level context spans for a prompt.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "repo_root": {"type": "string"},
                        },
                        "required": ["prompt"],
                    },
                },
                {
                    "name": "discover_dependencies",
                    "description": "Discover minimal required code dependencies for a prompt.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "repo_root": {"type": "string"},
                        },
                        "required": ["prompt"],
                    },
                },
                {
                    "name": "detect_prompt_drift",
                    "description": "Detect scope widening, hallucinated files, or altered intent.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "raw_text": {"type": "string"},
                            "optimized_text": {"type": "string"},
                        },
                        "required": ["raw_text", "optimized_text"],
                    },
                },
                {
                    "name": "estimate_prompt_cost",
                    "description": "Estimate token cost of a prompt.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"prompt": {"type": "string"}},
                        "required": ["prompt"],
                    },
                },
                {
                    "name": "compress_context",
                    "description": "Compress a prompt/context to minimal token representation.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"prompt": {"type": "string"}},
                        "required": ["prompt"],
                    },
                },
                {
                    "name": "validate_intent",
                    "description": "Validate semantic similarity between raw and optimized prompt.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "raw_text": {"type": "string"},
                            "optimized_text": {"type": "string"},
                        },
                        "required": ["raw_text", "optimized_text"],
                    },
                },
                {
                    "name": "benchmark_prompt",
                    "description": "Run the built-in benchmark suite.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ]
        }
