"""OpenAI adapter."""
from __future__ import annotations
from typing import Optional, Dict, Any

class OpenAIAdapter:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.model = model
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            self.client = None

    def available(self) -> bool:
        return self.client is not None

    def send(self, prompt: str, max_tokens: int = 4096) -> Dict[str, Any]:
        if not self.available():
            raise RuntimeError("openai package not installed")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return {"content": response.choices[0].message.content, "model": self.model}
