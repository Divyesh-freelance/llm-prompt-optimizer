"""Ollama local LLM adapter."""
from __future__ import annotations
import json, urllib.request
from typing import Dict, Any

class OllamaAdapter:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url
        self.model = model

    def available(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2)
            return True
        except Exception:
            return False

    def send(self, prompt: str, **kwargs) -> Dict[str, Any]:
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        return {"content": data.get("response", ""), "model": self.model}
