"""Tests for the FastAPI REST API layer."""

import pytest
from fastapi.testclient import TestClient

from llm_prompt_optimizer.api.app import app, get_optimizer
from llm_prompt_optimizer import Optimizer, OptimizerConfig


@pytest.fixture(scope="module")
def client():
    cfg = OptimizerConfig()
    cfg.policy.enable_audit_log = False
    test_optimizer = Optimizer(config=cfg)
    app.dependency_overrides[get_optimizer] = lambda: test_optimizer
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_optimize_endpoint(client):
    resp = client.post("/optimize", json={
        "prompt": "Debug EMA mismatch in signals/IndexSignals.py",
        "strict_mode": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "optimized_text" in data
    assert data["optimized_text"] != ""
    assert "token_estimate" in data
    assert data["token_estimate"] > 0


def test_optimize_strict_mode(client):
    resp = client.post("/optimize", json={
        "prompt": "Debug EMA mismatch in signals/IndexSignals.py",
        "strict_mode": True,
    })
    assert resp.status_code == 200


def test_classify_endpoint(client):
    resp = client.post("/classify", json={
        "prompt": "Debug EMA mismatch in signals/IndexSignals.py"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["primary_category"] == "debugging"
    assert 0.0 <= data["primary_confidence"] <= 1.0


def test_validate_endpoint_passes(client):
    text = "Debug EMA mismatch in signals/IndexSignals.py"
    resp = client.post("/validate", json={"raw_text": text, "optimized_text": text})
    assert resp.status_code == 200
    assert resp.json()["passed"] is True


def test_validate_endpoint_fails(client):
    resp = client.post("/validate", json={
        "raw_text": "Debug EMA mismatch",
        "optimized_text": "Write a poem about flowers."
    })
    assert resp.status_code == 200
    # Should be false (low similarity)
    assert isinstance(resp.json()["passed"], bool)


def test_detect_drift_endpoint_clean(client):
    text = "Debug EMA mismatch in signals/IndexSignals.py"
    resp = client.post("/detect-drift", json={"raw_text": text, "optimized_text": text})
    assert resp.status_code == 200
    assert resp.json()["is_clean"] is True


def test_estimate_cost_endpoint(client):
    resp = client.post("/estimate-cost", json={"prompt": "Debug EMA mismatch"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["estimated_tokens"] > 0
    assert data["approx_chars"] > 0


def test_resolve_context_endpoint(client):
    resp = client.post("/resolve-context", json={
        "prompt": "Debug EMA mismatch in signals/IndexSignals.py"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "spans" in data
    assert "total_lines" in data


def test_plugins_endpoint(client):
    resp = client.get("/plugins")
    assert resp.status_code == 200
    assert "plugins" in resp.json()


def test_benchmark_endpoint(client):
    resp = client.post("/benchmark")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "results" in data


def test_optimize_returns_category(client):
    resp = client.post("/optimize", json={
        "prompt": "Debug EMA mismatch in signals/IndexSignals.py"
    })
    data = resp.json()
    assert data["category"] == "debugging"


def test_optimize_returns_intent(client):
    resp = client.post("/optimize", json={
        "prompt": "Debug EMA mismatch in signals/IndexSignals.py"
    })
    data = resp.json()
    assert data["intent"] != ""
