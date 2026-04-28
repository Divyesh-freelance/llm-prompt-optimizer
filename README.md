# llm-prompt-optimizer

> **Deterministic prompt optimization middleware for AI coding agents.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-purple.svg)](https://modelcontextprotocol.io)

A universal, installable, MCP-compatible **Prompt Compiler + Context Governance Engine** that acts as a deterministic middleware layer for AI coding agents.

Works as a **standalone engine** while optionally integrating with graph providers or external tools when available.

---

## Core Philosophy

| The optimizer MUST | The optimizer MUST NEVER |
|---|---|
| Preserve semantic intent | Hallucinate user goals |
| Remain deterministic | Invent broader tasks |
| Discover minimal required context | Rewrite prompts creatively |
| Explain why dependencies were included | Widen repo scope unnecessarily |
| Gracefully degrade when tools unavailable | Assume architecture redesign |
| Optimize at line-level granularity | Depend on hardcoded limits |
| Remain agent-agnostic | Require external graph tools to function |

**Highest priority rule: Intent preservation > Compression. Always.**

---

## Pipeline Architecture

```
User Prompt
      ↓
Intent Guard           ← extracts & seals user intent (immutable)
      ↓
Prompt Classification  ← 7-layer multi-label classification (no keyword-only)
      ↓
Dependency Resolution  ← FallbackGraphEngine (AST, imports, symbols, git)
      ↓
Adaptive Context Expansion ← value-based: (relevance × confidence × proximity) / token_cost
      ↓
Precise Context Extraction ← line-level, never full files
      ↓
Prompt Optimization    ← compression + constraint injection (no creative rewriting)
      ↓
Semantic Validation    ← reject if similarity < 0.90
      ↓
Drift Detection        ← scope widening, hallucinated files, altered intent
      ↓
Prompt Compiler        ← final assembly for LLM
      ↓
AI Agent / LLM
```

---

## Quick Start

### Install

```bash
pip install llm-prompt-optimizer

# With ML-powered semantic similarity (recommended):
pip install "llm-prompt-optimizer[ml]"

# With Anthropic adapter:
pip install "llm-prompt-optimizer[anthropic]"

# Everything:
pip install "llm-prompt-optimizer[all]"
```

### SDK Usage

```python
from llm_prompt_optimizer import Optimizer

optimizer = Optimizer()

result = optimizer.optimize(
    prompt="Debug condition mismatch in the provided /path/repository/file. No code changes.",
    strict_mode=True,
    repo_root="/path/to/your/repo",
)

print(result.optimized_prompt.text)
print(f"Tokens: {result.optimized_prompt.token_estimate}")
print(f"Similarity: {result.optimized_prompt.semantic_similarity:.3f}")
print(f"Category: {result.classification.primary_category.value}")
print(f"Intent: {result.intent_lock.intent_summary}")
print(f"Drift clean: {result.drift_report.is_clean}")
```

### CLI Usage

```bash
# Optimize a prompt
llm-prompt-optimizer optimize "debug condition mismatch in path/to/folder"

# Strict mode (no dependency expansion)
llm-prompt-optimizer optimize "debug condition mismatch" --strict

# JSON output
llm-prompt-optimizer optimize "debug condition mismatch" --json

# Classify a prompt
llm-prompt-optimizer classify "implement /health endpoint in api/routes.py"

# Validate semantic similarity
llm-prompt-optimizer validate \
  --raw "debug condition mismatch" \
  --optimized "# Task\ndebug condition mismatch\n## Constraints\n- No code changes."

# Detect drift
llm-prompt-optimizer detect-drift \
  --raw "debug condition mismatch" \
  --optimized "debug condition mismatch. Also refactor everything."

# Estimate token cost
llm-prompt-optimizer estimate-cost "debug condition mismatch"

# Run benchmark suite
llm-prompt-optimizer benchmark

# Start MCP server (stdio — for Claude Desktop / Cursor)
llm-prompt-optimizer serve --transport stdio

# Start HTTP server
llm-prompt-optimizer serve --transport http --port 8765

# WebSocket server
llm-prompt-optimizer serve --transport websocket --port 8766
```

---

## MCP Server Integration

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
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
```

### Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "lpo": {
      "command": "lpo",
      "args": ["serve", "--transport", "stdio"]
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `optimize_prompt` | Full pipeline optimization |
| `classify_prompt` | Multi-label task classification |
| `resolve_precise_context` | Line-level context extraction |
| `discover_dependencies` | Minimal dependency discovery |
| `detect_prompt_drift` | Scope/intent drift detection |
| `estimate_prompt_cost` | Token cost estimation |
| `compress_context` | Context compression |
| `validate_intent` | Semantic similarity validation |
| `benchmark_prompt` | Run built-in benchmarks |

> Registering the MCP server only makes these tools **available** to the agent — it does not force the agent to call them on every prompt. To make `optimize_prompt` run automatically before each coding turn, see the next section.

---

## Default-On Optimization (install / uninstall / reinstall)

By default, registering the MCP server makes `optimize_prompt` available to your agent but doesn't force it to be called. To make every coding-related prompt auto-route through the optimizer, the package ships an installer that writes a small, sentinel-bracketed instruction block into your agent's host configuration files.

### TL;DR

```bash
# 1) install the package
pip install llm-prompt-optimizer

# 2) register the MCP server in your host (see "MCP Server Integration" above)

# 3) install the auto-routing rule
llm-prompt-optimizer install-rules

# inspect / undo at any time
llm-prompt-optimizer rules-status
llm-prompt-optimizer uninstall-rules
```

### What `install-rules` actually does

It auto-detects which agent hosts are present on your machine (Claude Code, Cursor, Continue) and writes a rule block into each one's user-global config. The rule tells the agent to call `optimize_prompt` first for any coding-related message and includes explicit fail-open wording: *if the optimizer is unavailable, errors, or times out, proceed normally with the user's original message.* So a server outage or a missing tool degrades to "no optimization", never to "broken assistant".

A health check runs first and **refuses to install** if the MCP server is not registered in any host config. This prevents shipping a rule that points at a tool that demonstrably doesn't exist. Pass `--force` to override.

### Where rule files are written

| Host | User-global path | Project-local path (with `--scope project`) |
|------|------------------|----------------------------------------------|
| Claude Code | `~/.claude/CLAUDE.md` | `<repo>/CLAUDE.md` |
| Cursor | `~/.cursor/rules/llm-prompt-optimizer.mdc` | `<repo>/.cursorrules` |
| Continue | `~/.continue/.continuerules` | `<repo>/.continuerules` |

The install manifest itself lives at `~/.config/llm-prompt-optimizer/install-manifest.json` (or `%APPDATA%\llm-prompt-optimizer\install-manifest.json` on Windows). Override with `$LPO_INSTALL_HOME`.

Each inserted block is wrapped in sentinels — only this region is ever touched on uninstall:

```text
<!-- BEGIN llm-prompt-optimizer v1 -->
…rule text…
<!-- END llm-prompt-optimizer -->
```

### Inspect what's installed

```bash
llm-prompt-optimizer rules-status
```

Reports the block version, health-check result, which hosts are detected on the machine, which already have the MCP server registered, and the full list of currently managed files.

### Reinstall / upgrade

Just run `install-rules` again. The installer detects an existing block (any prior version, e.g. `v0`, `v1`) and replaces it in place. There is no duplication, and the original `file_existed_before` state is preserved across re-installs — which means a later `uninstall-rules` still correctly distinguishes files we created from files you already owned.

```bash
llm-prompt-optimizer install-rules    # safe to run multiple times
```

### Uninstall

```bash
llm-prompt-optimizer uninstall-rules
```

Manifest-driven and surgical:

- Only the BEGIN/END-bracketed region is removed; everything you wrote above and below stays byte-for-byte intact.
- Files the installer created from scratch are deleted (so you don't get an orphaned empty `.mdc` lying around).
- Files you already owned are kept with just the block stripped.
- Running it twice is a no-op — no errors, nothing changed.

### Recovering from a stale install

If you removed the package without first running `uninstall-rules` — e.g. `pip uninstall llm-prompt-optimizer` then notice an orphaned block weeks later — you can sweep host-standard paths for any leftover BEGIN/END region, even with no manifest entry:

```bash
llm-prompt-optimizer uninstall-rules --purge-unknown
```

If you don't have the package installed anywhere, the block is plain text in a known file — open the rule file in any editor and delete everything between the `<!-- BEGIN llm-prompt-optimizer … -->` and `<!-- END llm-prompt-optimizer -->` markers (inclusive).

### Flag reference

```bash
llm-prompt-optimizer install-rules \
  [--host claude-code|cursor|continue] \  # repeat for multiple; default = auto-detect
  [--scope user-global|project] \         # default: user-global
  [--project-root <path>] \               # required if --scope project
  [--force] \                             # skip MCP-registration health check
  [--dry-run] \                           # show what would change, don't write
  [--json]                                # machine-readable output

llm-prompt-optimizer uninstall-rules \
  [--purge-unknown] \                     # sweep orphans not in the manifest
  [--dry-run] \
  [--json]

llm-prompt-optimizer rules-status [--json]
```

### What happens if the MCP server is disconnected or removed?

| Scenario | Behavior |
|----------|----------|
| **Server registered but down** (crashed, port conflict, dependency upgrade broke it) | Host marks the tool unavailable. The rule's fail-open clause kicks in: agent answers normally with the user's original message. No retry loop, no error surfaced. |
| **Server fully uninstalled** (entry removed from `claude_desktop_config.json` / `mcp.json`) | The tool isn't visible. Fail-open clause still applies, but you should also run `uninstall-rules` to remove the now-pointless instruction. |
| **Package removed without cleanup** (`pip uninstall` without `uninstall-rules` first) | Orphaned block remains in rule files. Run `uninstall-rules --purge-unknown` from any environment that still has the package, or delete the BEGIN/END region by hand. |
| **Re-installing on top of an existing install** | Old block (any version) is found and replaced in place. No duplication. |
| **Manifest file is missing or corrupt** | Uninstall does not crash. Use `--purge-unknown` to clean orphans on a best-effort basis. |
| **You manually edited the rule file between install and uninstall** | Edits outside the BEGIN/END region are preserved. Edits *inside* the region are lost on uninstall (this is by design — it's our managed region). |

### Project-scope install (advanced)

For teams that want every developer who clones the repo to get auto-routing, you can write a project-local rule file that gets committed to git:

```bash
llm-prompt-optimizer install-rules --scope project --project-root .
```

Be aware: this writes `<repo>/CLAUDE.md`, `<repo>/.cursorrules`, etc. Anyone who clones the repo without the optimizer installed will see the rule but the tool won't resolve — the fail-open clause means their agent still works, but they'll incur a failed tool call per turn until they install the package or remove the rule. Use this only when the team is committed to the workflow.

---

## REST API

```bash
# Start API server
uvicorn llm_prompt_optimizer.api.app:app --host 0.0.0.0 --port 8765
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/optimize` | Optimize a prompt |
| `POST` | `/classify` | Classify a prompt |
| `POST` | `/validate` | Semantic validation |
| `POST` | `/detect-drift` | Drift detection |
| `POST` | `/estimate-cost` | Token cost estimation |
| `POST` | `/resolve-context` | Precise context resolution |
| `GET`  | `/plugins` | List loaded plugins |
| `POST` | `/benchmark` | Run benchmarks |

Interactive docs: `http://localhost:8765/docs`

---

## Docker

```bash
# Build and run the API server
docker compose up optimizer-api

# Run tests
docker compose --profile test up optimizer-test

# MCP stdio mode (for agent integration)
docker compose --profile mcp up optimizer-mcp
```

---

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Policy
LPO_STRICT_INTENT=true
LPO_SEMANTIC_THRESHOLD=0.90
LPO_TOKEN_BUDGET=8000

# MCP Server
LPO_MCP_HOST=127.0.0.1
LPO_MCP_PORT=8765
LPO_MCP_TRANSPORT=stdio

# Logging
LPO_LOG_LEVEL=INFO

# Graph Provider (optional — uses FallbackGraphEngine by default)
LPO_GRAPH_PROVIDER=
```

### Python Configuration

```python
from llm_prompt_optimizer import Optimizer, OptimizerConfig
from llm_prompt_optimizer.config.settings import PolicyConfig, TokenBudgetConfig

config = OptimizerConfig()
config.policy.strict_intent_mode = True
config.policy.semantic_similarity_threshold = 0.90
config.policy.allow_scope_expansion = "controlled"
config.token_budget.default_budget_tokens = 8000
config.token_budget.adaptive_budgeting = True

optimizer = Optimizer(config=config)
```

---

## Plugin System

```python
from llm_prompt_optimizer.plugins.system import PolicyPlugin, OptimizerPlugin, PluginSystem
from llm_prompt_optimizer import Optimizer

class MyEnterprisePolicy(PolicyPlugin):
    name = "enterprise_policy"
    version = "1.0.0"
    plugin_type = "policy"

    def initialize(self, config):
        self.blocked_terms = config.get("blocked_terms", [])

    def evaluate(self, context):
        violations = []
        intent = context.get("intent_lock")
        if intent:
            for term in self.blocked_terms:
                if term in intent.intent_summary.lower():
                    violations.append(f"Blocked term: {term}")
        return violations

plugins = PluginSystem()
plugins.register(MyEnterprisePolicy(), config={"blocked_terms": ["prod_db"]})

optimizer = Optimizer(plugin_system=plugins)
```

### Plugin types

| Type | Base Class | Purpose |
|------|-----------|---------|
| `optimizer` | `OptimizerPlugin` | Pre/post prompt middleware |
| `policy` | `PolicyPlugin` | Custom governance rules |
| `classifier` | `ClassifierPlugin` | Custom classification signals |
| `graph` | `GraphPlugin` | Alternative dependency discovery |
| `telemetry` | `TelemetryPlugin` | Custom telemetry backends |

---

## Dependency Resolution

Priority order (graceful degradation):

1. **External Graph MCP** — Graphify, Code Review Graph (when available)
2. **Local AST Graph** — Python AST import chain analysis
3. **Import Resolver** — dotted module → file path resolution
4. **Symbol Resolver** — function/class definition finding
5. **Execution Path Discovery** — call graph tracing
6. **Git Context** — co-changed file analysis
7. **Folder Heuristic** — directory proximity scoring
8. **Strict User Scope** — only explicitly stated files

The system **always works standalone** — external tools only enhance it.

---

## Adaptive Context Expansion

Context expansion uses a value formula — **not fixed limits**:

```
context_value_score = (relevance × confidence × execution_proximity) / token_cost
```

**Stop conditions** (not arbitrary limits):
- Confidence gain below threshold
- Token cost exceeds marginal value
- Execution relevance weakens
- Semantic confidence decreases
- Budget exhausted

---

## Precise Context Extraction

Never sends entire files. Returns exact line spans:

```json
{
  "file": "utils/condition.py",
  "start_line": 61,
  "end_line": 95,
  "symbol": "calculate_condition",
  "confidence": 0.93,
  "reason": "Symbol definition: calculate_condition"
}
```

**Principle: Relevant Lines > Relevant Files**

---

## Benchmarks

```bash
python benchmarks/run_benchmarks.py
python benchmarks/run_benchmarks.py --json
python benchmarks/run_benchmarks.py --case debug_condition_mismatch
```

Metrics measured:
- Semantic similarity preservation
- Token reduction %
- Classification accuracy
- Context usefulness
- Dependency precision
- Adaptive expansion efficiency
- Constraint preservation

---

## Development

```bash
# Clone
git clone https://github.com/your-org/llm-prompt-optimizer
cd llm-prompt-optimizer

# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=llm_prompt_optimizer --cov-report=html

# Lint
ruff check llm_prompt_optimizer/

# Type check
mypy llm_prompt_optimizer/

# Run benchmarks
python benchmarks/run_benchmarks.py
```

---

## Project Structure

```
llm-prompt-optimizer/
├── llm_prompt_optimizer/
│   ├── core/
│   │   ├── intent_guard/         # IntentGuard — locks user intent
│   │   ├── classifier/           # Multi-layer prompt classification
│   │   ├── dependency_resolution/ # DependencyResolver orchestrator
│   │   ├── adaptive_context_expansion/ # Value-based expansion
│   │   ├── precise_context/      # Line-level extraction
│   │   ├── optimizer/            # PromptOptimizer + PromptCompiler
│   │   ├── semantic_validator/   # Similarity validation
│   │   ├── drift_detection/      # Drift detector
│   │   ├── fallback_graph/       # Standalone AST graph engine
│   │   ├── token_budget/         # Token budget management
│   │   ├── policy/               # PolicyEngine
│   │   ├── telemetry/            # TelemetryEngine + AuditLogger
│   │   └── benchmarking/         # BenchmarkEngine
│   ├── adapters/                 # LLM adapters (Anthropic, OpenAI, Ollama…)
│   ├── mcp_server/               # MCP server (stdio, HTTP, WebSocket, IPC)
│   ├── installer/                # Host-rule install/uninstall (CLAUDE.md, .cursorrules, …)
│   ├── api/                      # FastAPI REST endpoints
│   ├── sdk/                      # Optimizer SDK (main entry point)
│   ├── plugins/                  # Plugin system
│   ├── models/                   # Typed data models
│   ├── config/                   # Configuration
│   └── utils/                    # Helpers, token estimation, AST tools
├── examples/
│   ├── claude_mcp/               # Claude Desktop MCP integration
│   ├── langchain/                # LangChain integration
│   └── local_agents/             # Custom plugin examples
├── tests/
│   ├── core/                     # Unit tests per component
│   ├── integration/              # Full pipeline + MCP tests
│   └── api/                      # API endpoint tests
├── benchmarks/                   # Benchmark runner + cases
├── docs/                         # Extended documentation
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## Supported Agent Ecosystems

- **Claude Desktop** (MCP stdio)
- **Cursor** (MCP stdio)
- **Continue.dev** (MCP stdio)
- **LangChain** (SDK middleware)
- **AutoGen** (SDK middleware)
- **CrewAI** (SDK middleware)
- **Any HTTP client** (REST API)
- **Any stdio MCP client** (MCP server)

---

## License

MIT — see [LICENSE](LICENSE)
