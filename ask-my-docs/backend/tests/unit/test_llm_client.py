"""Tests for OllamaClient — httpx is mocked so no Ollama server needed."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.generation.llm_client import OllamaClient


BASE_URL = "http://localhost:11434"
MODEL = "llama3.1:8b"


@pytest.fixture
def client():
    return OllamaClient(base_url=BASE_URL, model=MODEL)


# ------------------------------------------------------------------
# Helpers to build fake NDJSON Ollama responses
# ------------------------------------------------------------------

def _ndjson_lines(tokens: list[str], *, done_at_end: bool = True) -> list[str]:
    lines = [json.dumps({"response": t, "done": False}) for t in tokens]
    if done_at_end:
        lines.append(json.dumps({"response": "", "done": True}))
    return lines


# ------------------------------------------------------------------
# stream_completion
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_completion_yields_tokens(client):
    tokens = ["Hello", " world", "!"]

    async def fake_aiter_lines():
        for line in _ndjson_lines(tokens):
            yield line

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = fake_aiter_lines

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_ctx)

    mock_outer = AsyncMock()
    mock_outer.__aenter__ = AsyncMock(return_value=mock_client)
    mock_outer.__aexit__ = AsyncMock(return_value=False)

    with patch("app.generation.llm_client.httpx.AsyncClient", return_value=mock_outer):
        collected = []
        async for delta in client.stream_completion("prompt"):
            collected.append(delta)

    assert collected == tokens


@pytest.mark.asyncio
async def test_stream_completion_skips_empty_response_field(client):
    lines = [
        json.dumps({"response": "Real", "done": False}),
        json.dumps({"response": "", "done": False}),   # empty — must be skipped
        json.dumps({"response": " text", "done": True}),
    ]

    async def fake_aiter_lines():
        for line in lines:
            yield line

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = fake_aiter_lines

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_ctx)

    mock_outer = AsyncMock()
    mock_outer.__aenter__ = AsyncMock(return_value=mock_client)
    mock_outer.__aexit__ = AsyncMock(return_value=False)

    with patch("app.generation.llm_client.httpx.AsyncClient", return_value=mock_outer):
        collected = [d async for d in client.stream_completion("prompt")]

    assert "" not in collected
    assert "Real" in collected
    assert " text" in collected


@pytest.mark.asyncio
async def test_stream_completion_connect_error_returns_message(client):
    import httpx as _httpx

    mock_outer = AsyncMock()
    mock_outer.__aenter__ = AsyncMock(side_effect=_httpx.ConnectError("refused"))
    mock_outer.__aexit__ = AsyncMock(return_value=False)

    with patch("app.generation.llm_client.httpx.AsyncClient", return_value=mock_outer):
        collected = [d async for d in client.stream_completion("prompt")]

    assert len(collected) == 1
    assert "not reachable" in collected[0]
    assert BASE_URL in collected[0]


# ------------------------------------------------------------------
# completion (non-streaming)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_completion_returns_full_text(client):
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=MagicMock(
        raise_for_status=MagicMock(),
        json=MagicMock(return_value={"response": "Full answer here."}),
    ))

    mock_outer = AsyncMock()
    mock_outer.__aenter__ = AsyncMock(return_value=mock_http)
    mock_outer.__aexit__ = AsyncMock(return_value=False)

    with patch("app.generation.llm_client.httpx.AsyncClient", return_value=mock_outer):
        result = await client.completion("prompt")

    assert result == "Full answer here."


@pytest.mark.asyncio
async def test_completion_connect_error_returns_message(client):
    import httpx as _httpx

    mock_outer = AsyncMock()
    mock_outer.__aenter__ = AsyncMock(side_effect=_httpx.ConnectError("refused"))
    mock_outer.__aexit__ = AsyncMock(return_value=False)

    with patch("app.generation.llm_client.httpx.AsyncClient", return_value=mock_outer):
        result = await client.completion("prompt")

    assert "not reachable" in result
    assert BASE_URL in result
