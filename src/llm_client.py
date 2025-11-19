"""LLM client wrapper used across the orchestrator."""

import json
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LLMClient:
    """Thin wrapper around an OpenAI-compatible ChatCompletion API."""

    def __init__(self,
                 model: str,
                 temperature: float = 0.7,
                 seed: Optional[int] = None,
                 max_tokens: Optional[int] = None):
        if OpenAI is None:
            raise RuntimeError(
                "openai package not available. Install `openai` or plug your own client into LLMClient."
            )
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens

    def chat_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Ask the model for a JSON object response."""
        params: Dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.seed is not None:
            params["seed"] = self.seed

        completion = self.client.chat.completions.create(**params)
        content = completion.choices[0].message.content
        return json.loads(content)
