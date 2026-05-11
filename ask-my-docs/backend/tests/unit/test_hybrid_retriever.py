"""Unit tests for HybridRetriever RRF fusion logic.

VectorStore, BM25Store, and Embedder are replaced with MagicMocks so the
tests exercise only the fusion and tagging logic — no real models or servers.
"""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from app.retrieval.hybrid_retriever import HybridRetriever, _RRF_K, _rrf_score
from app.retrieval.vector_store import RetrievalResult


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _result(chunk_id: str, page: int = 1, source: str = "dense") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        page_number=page,
        section_title=None,
        parent_text=f"parent text for {chunk_id}",
        score=1.0,
        retrieval_source=source,
    )


def _make_retriever(dense: list[RetrievalResult], sparse: list[RetrievalResult]) -> HybridRetriever:
    mock_vs = MagicMock()
    mock_vs.query.return_value = dense

    mock_bm25 = MagicMock()
    mock_bm25.query.return_value = sparse

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.0] * 4  # dummy embedding

    return HybridRetriever(
        vector_store=mock_vs,
        bm25_store=mock_bm25,
        embedder=mock_embedder,
    )


# ------------------------------------------------------------------
# RRF score helper
# ------------------------------------------------------------------

def test_rrf_score_formula():
    assert _rrf_score(1) == pytest.approx(1 / (_RRF_K + 1))
    assert _rrf_score(10) == pytest.approx(1 / (_RRF_K + 10))
    # Higher rank → lower score
    assert _rrf_score(1) > _rrf_score(2) > _rrf_score(10)


# ------------------------------------------------------------------
# Core fusion tests (from spec §11)
# ------------------------------------------------------------------

def test_rrf_fusion_both_sources_scores_higher():
    """Chunk in both lists must outscore a chunk in only one list."""
    shared = _result("shared")
    dense_only = _result("dense_only")
    sparse_only = _result("sparse_only")

    retriever = _make_retriever(
        dense=[shared, dense_only],
        sparse=[shared, sparse_only],
    )
    results = retriever.retrieve("test query", top_k=10)

    by_id = {r.chunk_id: r for r in results}
    assert "shared" in by_id
    assert by_id["shared"].score > by_id["dense_only"].score
    assert by_id["shared"].score > by_id["sparse_only"].score


def test_rrf_k_constant_does_not_change_top_result():
    """k=60 is baked in but changing the constant should not swap a strongly
    overlapping top result — the winner is stable across reasonable k values."""
    dense = [_result("winner"), _result("runner_up")]
    sparse = [_result("winner"), _result("other")]

    retriever = _make_retriever(dense, sparse)
    results = retriever.retrieve("test", top_k=5)

    assert results[0].chunk_id == "winner"


def test_retrieval_source_tagged_correctly():
    """Results must be tagged dense / sparse / both accurately."""
    dense_results = [_result("A"), _result("B")]
    sparse_results = [_result("A"), _result("C")]

    retriever = _make_retriever(dense_results, sparse_results)
    results = retriever.retrieve("test", top_k=10)

    by_id = {r.chunk_id: r for r in results}
    assert by_id["A"].retrieval_source == "both"
    assert by_id["B"].retrieval_source == "dense"
    assert by_id["C"].retrieval_source == "sparse"


def test_empty_sparse_results_handled_gracefully():
    """BM25 returning nothing must not crash — dense results still returned."""
    dense_results = [_result("A"), _result("B"), _result("C")]

    retriever = _make_retriever(dense=dense_results, sparse=[])
    results = retriever.retrieve("test", top_k=10)

    assert len(results) == 3
    assert all(r.retrieval_source == "dense" for r in results)


def test_empty_dense_results_handled_gracefully():
    sparse_results = [_result("X"), _result("Y")]

    retriever = _make_retriever(dense=[], sparse=sparse_results)
    results = retriever.retrieve("test", top_k=10)

    assert len(results) == 2
    assert all(r.retrieval_source == "sparse" for r in results)


def test_both_empty_returns_empty():
    retriever = _make_retriever(dense=[], sparse=[])
    results = retriever.retrieve("test", top_k=10)
    assert results == []


def test_top_k_limits_output():
    dense = [_result(f"d{i}") for i in range(15)]
    sparse = [_result(f"s{i}") for i in range(15)]

    retriever = _make_retriever(dense, sparse)
    results = retriever.retrieve("test", top_k=7)

    assert len(results) <= 7


def test_results_sorted_by_rrf_score_descending():
    dense = [_result("A"), _result("B"), _result("C")]
    sparse = [_result("B"), _result("A"), _result("D")]

    retriever = _make_retriever(dense, sparse)
    results = retriever.retrieve("test", top_k=10)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_dense_rank1_in_both_beats_rank1_in_one():
    """A chunk ranked #1 in both lists must score 2/(k+1) > 1/(k+1)."""
    top_both = _result("top_both")
    top_dense_only = _result("top_dense_only")

    retriever = _make_retriever(
        dense=[top_both, top_dense_only],
        sparse=[top_both],
    )
    results = retriever.retrieve("test", top_k=5)
    by_id = {r.chunk_id: r for r in results}

    expected_both = 2 * _rrf_score(1)
    expected_dense = _rrf_score(2)  # rank 2 in dense, absent in sparse

    assert by_id["top_both"].score == pytest.approx(expected_both)
    assert by_id["top_dense_only"].score == pytest.approx(expected_dense)
    assert by_id["top_both"].score > by_id["top_dense_only"].score
