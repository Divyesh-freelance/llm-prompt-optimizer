"""Anthropic Claude adapter."""
from __future__ import annotations
from typing import Optional, Dict, Any

class AnthropicAdapter:
    """Sends optimized prompts to Anthropic Claude API."""
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-opus-4-5"):
        self.model = model
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            self.client = None

    def available(self) -> bool:
        return self.client is not None

    def send(self, prompt: str, max_tokens: int = 4096) -> Dict[str, Any]:
        if not self.available():
            raise RuntimeError("anthropic package not installed")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return {"content": response.content[0].text, "model": self.model}
