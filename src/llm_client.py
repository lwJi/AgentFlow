"""LLM client wrapper used across the orchestrator."""

import json
import os
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:  # pragma: no cover - requests usually available
    requests = None

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
        self.api_base = os.environ.get("LLM_BASE_URL")
        self.api_key = os.environ.get("LLM_API_KEY")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens

        if self.api_base:
            self.client = None
        else:
            if OpenAI is None:
                raise RuntimeError(
                    "openai package not available. Install `openai` or set LLM_BASE_URL/LLM_API_KEY."
                )
            self.client = OpenAI()

    def _build_payload(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
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
        return params

    def chat_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Ask the model for a JSON object response."""
        params = self._build_payload(system_prompt, user_prompt)

        if self.api_base:
            if requests is None:
                raise RuntimeError("requests package required when LLM_BASE_URL is set.")
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            resp = requests.post(self.api_base, json=params, headers=headers, timeout=60)
            resp.raise_for_status()
            completion = resp.json()
            content = completion["choices"][0]["message"]["content"]
        else:
            completion = self.client.chat.completions.create(**params)
            content = completion.choices[0].message.content

        return json.loads(content)
