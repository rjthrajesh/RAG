from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

_GENERATE_PATH = "/api/generate"
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    async def stream_completion(self, prompt: str) -> AsyncIterator[str]:
        url = f"{_GROQ_BASE_URL}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:]
                        if raw.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        text = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if text:
                            yield text
        except httpx.ConnectError:
            yield "[Error: Could not reach Groq API]"
        except httpx.HTTPStatusError as exc:
            yield f"[Error: Groq returned {exc.response.status_code}]"

    async def completion(self, prompt: str) -> str:
        url = f"{_GROQ_BASE_URL}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except httpx.ConnectError:
            return "[Error: Could not reach Groq API]"
        except httpx.HTTPStatusError as exc:
            return f"[Error: Groq returned {exc.response.status_code}]"

    def completion_sync(self, prompt: str) -> str:
        """Synchronous variant — uses httpx.Client to avoid event loop conflicts in scripts."""
        return self.completion_sync_messages([{"role": "user", "content": prompt}])

    def completion_sync_messages(self, messages: list[dict[str, str]], _retries: int = 5) -> str:
        """Synchronous chat completion preserving full message roles (system/user/assistant).

        Retries up to _retries times with exponential backoff on 429 responses.
        """
        url = f"{_GROQ_BASE_URL}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "messages": messages, "stream": False}
        wait = 10.0
        for attempt in range(_retries + 1):
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(url, json=payload, headers=headers)
                    if response.status_code == 429 and attempt < _retries:
                        retry_after = float(response.headers.get("retry-after", wait))
                        logger.warning("Groq 429 — waiting %.0fs (attempt %d/%d)", retry_after, attempt + 1, _retries)
                        time.sleep(retry_after)
                        wait = min(wait * 2, 60.0)
                        continue
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
            except httpx.ConnectError:
                return "[Error: Could not reach Groq API]"
            except httpx.HTTPStatusError as exc:
                return f"[Error: Groq returned {exc.response.status_code}]"
        return "[Error: Groq rate limit exceeded after retries]"


class OllamaClient:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def stream_completion(self, prompt: str) -> AsyncIterator[str]:
        url = self.base_url + _GENERATE_PATH
        payload = {"model": self.model, "prompt": prompt, "stream": True}

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = chunk.get("response", "")
                        if text:
                            yield text
                        if chunk.get("done"):
                            break
        except httpx.ConnectError:
            yield f"[Error: Ollama not reachable at {self.base_url}]"
        except httpx.HTTPStatusError as exc:
            yield f"[Error: Ollama returned {exc.response.status_code}]"

    async def completion(self, prompt: str) -> str:
        url = self.base_url + _GENERATE_PATH
        payload = {"model": self.model, "prompt": prompt, "stream": False}

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json().get("response", "")
        except httpx.ConnectError:
            return f"[Error: Ollama not reachable at {self.base_url}]"
        except httpx.HTTPStatusError as exc:
            return f"[Error: Ollama returned {exc.response.status_code}]"

    def completion_sync(self, prompt: str) -> str:
        """Synchronous variant — uses httpx.Client to avoid event loop conflicts in scripts."""
        return self.completion_sync_messages([{"role": "user", "content": prompt}])

    def completion_sync_messages(self, messages: list[dict[str, str]]) -> str:
        """Synchronous chat completion via /api/chat, preserving full message roles."""
        url = self.base_url + "/api/chat"
        payload = {"model": self.model, "messages": messages, "stream": False}
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json().get("message", {}).get("content", "")
        except httpx.ConnectError:
            return f"[Error: Ollama not reachable at {self.base_url}]"
        except httpx.HTTPStatusError as exc:
            return f"[Error: Ollama returned {exc.response.status_code}]"
