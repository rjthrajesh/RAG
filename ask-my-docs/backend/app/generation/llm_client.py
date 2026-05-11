from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

_GENERATE_PATH = "/api/generate"


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
