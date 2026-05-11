from __future__ import annotations

import json
import os
import tempfile

import pytest

from app.ingestion.chunker import Chunk
from app.retrieval.bm25_store import BM25Store, _tokenize


def _make_chunk(chunk_id: str, text: str, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        page_number=page,
        section_title=None,
        char_count=len(text),
        parent_text=text,
    )


CHUNKS = [
    _make_chunk("page_1_chunk_0", "Retrieval augmented generation combines search with LLMs", 1),
    _make_chunk("page_1_chunk_1", "Prompt engineering improves LLM output quality", 1),
    _make_chunk("page_2_chunk_0", "Vector databases store dense embeddings for similarity search", 2),
    _make_chunk("page_2_chunk_1", "Fine-tuning adapts a model to a specific domain", 2),
]


@pytest.fixture
def built_store() -> BM25Store:
    store = BM25Store()
    store.build(CHUNKS)
    return store


def test_tokenize_lowercases_and_strips_punctuation():
    tokens = _tokenize("Hello, World! It's a test.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens
    assert all("," not in t and "!" not in t for t in tokens)


def test_tokenize_empty_string():
    assert _tokenize("") == []


def test_query_returns_top_k(built_store):
    results = built_store.query("retrieval augmented generation", top_k=2)
    assert len(results) <= 2


def test_query_top_result_is_relevant(built_store):
    results = built_store.query("retrieval augmented generation", top_k=4)
    assert results, "Expected at least one result"
    assert results[0].chunk_id == "page_1_chunk_0"


def test_query_scores_normalized_between_zero_and_one(built_store):
    results = built_store.query("LLM embedding search", top_k=4)
    for r in results:
        assert 0.0 <= r.score <= 1.0, f"Score out of range: {r.score}"


def test_query_retrieval_source_is_sparse(built_store):
    results = built_store.query("vector database", top_k=2)
    for r in results:
        assert r.retrieval_source == "sparse"


def test_query_zero_score_results_excluded(built_store):
    # Query with gibberish tokens that match nothing
    results = built_store.query("xyzzy quux frobnitz", top_k=4)
    for r in results:
        assert r.score > 0


def test_save_and_load_produces_same_results(built_store):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        built_store.save(path)
        loaded = BM25Store()
        loaded.load(path)
        original = built_store.query("retrieval augmented generation", top_k=3)
        reloaded = loaded.query("retrieval augmented generation", top_k=3)
        assert [r.chunk_id for r in original] == [r.chunk_id for r in reloaded]
    finally:
        os.unlink(path)


def test_save_json_structure(built_store):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    try:
        built_store.save(path)
        with open(path) as f:
            data = json.load(f)
        assert "corpus_tokens" in data
        assert "chunk_ids" in data
        assert "chunk_metadata" in data
        assert len(data["chunk_ids"]) == len(CHUNKS)
    finally:
        os.unlink(path)


def test_query_before_build_returns_empty():
    store = BM25Store()
    results = store.query("anything", top_k=5)
    assert results == []
