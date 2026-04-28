FROM python:3.12-slim AS base

LABEL maintainer="LLM Prompt Optimizer Contributors"
LABEL description="Deterministic prompt optimization middleware for AI coding agents"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Build stage ───────────────────────────────────────────────────────────────
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY llm_prompt_optimizer/ ./llm_prompt_optimizer/

RUN pip install --upgrade pip && \
    pip install ".[all]" --target /install

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM base AS runtime

COPY --from=builder /install /usr/local/lib/python3.12/site-packages/
COPY --from=builder /app /app

# Default environment
ENV LPO_MCP_TRANSPORT=http \
    LPO_MCP_HOST=0.0.0.0 \
    LPO_MCP_PORT=8765 \
    LPO_LOG_LEVEL=INFO \
    LPO_STRICT_INTENT=true \
    LPO_SEMANTIC_THRESHOLD=0.90 \
    LPO_TOKEN_BUDGET=8000

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/health')" || exit 1

CMD ["python", "-m", "uvicorn", "llm_prompt_optimizer.api.app:app", \
     "--host", "0.0.0.0", "--port", "8765", "--workers", "4"]

# ── Dev / test stage ──────────────────────────────────────────────────────────
FROM builder AS dev

RUN pip install ".[dev]"

COPY tests/ ./tests/
COPY benchmarks/ ./benchmarks/

CMD ["pytest", "tests/", "-v"]
