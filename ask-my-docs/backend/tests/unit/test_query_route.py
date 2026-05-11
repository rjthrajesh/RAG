"""Tests for the /query SSE route — retriever, reranker, and LLM are mocked."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.retrieval.vector_store import RetrievalResult


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

def _result(page: int, section: str = "Ch 1") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"page_{page}_chunk_0",
        text=f"Content on page {page} about important topics.",
        page_number=page,
        section_title=section,
        parent_text=f"Broader context around page {page} content.",
        score=0.85,
        retrieval_source="both",
        rerank_score=6.0,
    )


MOCK_RESULTS = [_result(10), _result(20), _result(30)]


async def _fake_stream(prompt: str):
    """Simulates Ollama streaming three tokens then stopping."""
    for token in ["RAG ", "retrieves ", f"documents [p.10]."]:
        yield token


@pytest.fixture(autouse=True)
def patch_singletons():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = MOCK_RESULTS

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = MOCK_RESULTS

    mock_llm = MagicMock()
    mock_llm.stream_completion = _fake_stream

    main_module._jobs.clear()

    with (
        patch.object(main_module, "_get_retriever", return_value=mock_retriever),
        patch.object(main_module, "_get_reranker", return_value=mock_reranker),
        patch.object(main_module, "_get_llm_client", return_value=mock_llm),
    ):
        yield


@pytest.fixture
def client():
    return TestClient(app)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_sse(raw: str) -> list[dict]:
    events = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def _query(client, question: str = "What is RAG?") -> list[dict]:
    with client.stream("POST", "/query", json={"question": question}) as resp:
        assert resp.status_code == 200
        resp.read()  # buffer the full SSE stream before the context exits
        return _parse_sse(resp.text)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_query_returns_200_with_sse_content_type(client):
    with client.stream("POST", "/query", json={"question": "What is RAG?"}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]


def test_query_stream_has_delta_events(client):
    events = _query(client)
    delta_events = [e for e in events if e.get("type") == "delta"]
    assert delta_events, "Expected at least one delta event"
    assert all("text" in e for e in delta_events)


def test_query_stream_concatenated_text_correct(client):
    events = _query(client)
    full = "".join(e["text"] for e in events if e.get("type") == "delta")
    assert "RAG" in full
    assert "[p.10]" in full


def test_query_stream_has_sources_event(client):
    events = _query(client)
    source_events = [e for e in events if e.get("type") == "sources"]
    assert len(source_events) == 1
    sources = source_events[0]["sources"]
    assert len(sources) == len(MOCK_RESULTS)
    for s in sources:
        assert "page_number" in s
        assert "section_title" in s
        assert "text_preview" in s
        assert "rerank_score" in s
        assert "retrieval_source" in s


def test_query_stream_has_done_event(client):
    events = _query(client)
    done_events = [e for e in events if e.get("type") == "done"]
    assert len(done_events) == 1
    done = done_events[0]
    assert "citation_valid" in done
    assert "warning" in done


def test_query_stream_event_order(client):
    events = _query(client)
    types = [e["type"] for e in events]
    # All deltas come before sources, sources before done
    last_delta = max((i for i, t in enumerate(types) if t == "delta"), default=-1)
    sources_idx = next((i for i, t in enumerate(types) if t == "sources"), -1)
    done_idx = next((i for i, t in enumerate(types) if t == "done"), -1)
    assert last_delta < sources_idx < done_idx


def test_query_citation_valid_for_cited_answer(client):
    """The mocked answer includes [p.10] which maps to page 10 in MOCK_RESULTS."""
    events = _query(client)
    done = next(e for e in events if e.get("type") == "done")
    assert done["citation_valid"] is True


def test_query_uses_configured_top_k(client):
    with client.stream(
        "POST", "/query",
        json={"question": "Q?", "top_k_retrieve": 7, "top_k_rerank": 3},
    ) as resp:
        resp.read()

    # Verify the retriever was called with the custom top_k
    retriever = main_module._get_retriever()
    retriever.retrieve.assert_called_once_with("Q?", top_k=7)
    reranker = main_module._get_reranker()
    reranker.rerank.assert_called_once()
    _, kwargs = reranker.rerank.call_args
    assert kwargs.get("top_k") == 3 or reranker.rerank.call_args[0][2] == 3


def test_query_sources_text_preview_truncated(client):
    events = _query(client)
    sources = next(e for e in events if e.get("type") == "sources")["sources"]
    for s in sources:
        assert len(s["text_preview"]) <= 200
