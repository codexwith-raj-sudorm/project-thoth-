"""Zero-extra-dependency HTTP adapters for supported model providers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from pydantic import BaseModel

from ..rate_limit import TokenBucket, retry_transient
from ..schemas import Usage
from .base import ModelAdapter

_TRANSIENT = {429, 502, 503, 504}


def _is_transient(exc: Exception) -> bool:
    return isinstance(exc, urllib.error.HTTPError) and exc.code in _TRANSIENT


class JSONHTTPAdapter(ModelAdapter):
    def __init__(self, *, requests_per_minute: float | None = 12, timeout: float = 45) -> None:
        self.limiter = TokenBucket(requests_per_minute) if requests_per_minute else None
        self.timeout = timeout

    def _post(self, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")

        def request() -> dict[str, Any]:
            if self.limiter:
                self.limiter.acquire()
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json", **(headers or {})},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        return retry_transient(request, is_transient=_is_transient)


class OllamaAdapter(JSONHTTPAdapter):
    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: float = 45) -> None:
        super().__init__(requests_per_minute=None, timeout=timeout)
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _generate_text(self, system_prompt: str, user_prompt: str, schema: type[BaseModel]) -> str:
        data = self._post(
            f"{self.base_url}/api/chat",
            {
                "model": self.model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        counts = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
        self.last_usage = Usage(prompt_tokens=counts[0], completion_tokens=counts[1])
        return data["message"]["content"]


class OpenAICompatibleAdapter(JSONHTTPAdapter):
    """Works with Groq and local OpenAI-compatible vLLM endpoints."""

    def __init__(self, model: str, base_url: str, api_key: str = "", requests_per_minute: float | None = 12) -> None:
        super().__init__(requests_per_minute=requests_per_minute)
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key

    def _generate_text(self, system_prompt: str, user_prompt: str, schema: type[BaseModel]) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = self._post(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
            headers,
        )
        usage = data.get("usage", {})
        self.last_usage = Usage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
        return data["choices"][0]["message"]["content"]


class GeminiAdapter(JSONHTTPAdapter):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        super().__init__(requests_per_minute=12)
        self.api_key, self.model = api_key, model

    def _generate_text(self, system_prompt: str, user_prompt: str, schema: type[BaseModel]) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        data = self._post(
            url,
            {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "responseMimeType": "application/json",
                    "responseJsonSchema": schema.model_json_schema(),
                },
            },
        )
        metadata = data.get("usageMetadata", {})
        self.last_usage = Usage(
            prompt_tokens=metadata.get("promptTokenCount", 0),
            completion_tokens=metadata.get("candidatesTokenCount", 0),
        )
        return data["candidates"][0]["content"]["parts"][0]["text"]
