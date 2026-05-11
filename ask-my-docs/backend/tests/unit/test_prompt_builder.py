"""Tests for PromptBuilder — pure logic, no external deps."""
from __future__ import annotations

import pytest

from app.generation.prompt_builder import PromptBuilder, _CHUNK_PREVIEW_LIMIT
from app.retrieval.vector_store import RetrievalResult


def _result(page: int, section: str | None = "Intro", text: str = "Some passage text.") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"page_{page}_chunk_0",
        text=text,
        page_number=page,
        section_title=section,
        parent_text=text,
        score=0.9,
        retrieval_source="both",
        rerank_score=5.0,
    )


@pytest.fixture
def builder():
    return PromptBuilder()


def test_prompt_contains_question(builder):
    q = "What is retrieval-augmented generation?"
    prompt = builder.build_rag_prompt(q, [_result(1)])
    assert q in prompt


def test_prompt_contains_answer_marker(builder):
    prompt = builder.build_rag_prompt("Q?", [_result(1)])
    assert "ANSWER (cite every claim with [p.X]):" in prompt


def test_prompt_contains_system_instructions(builder):
    prompt = builder.build_rag_prompt("Q?", [_result(1)])
    assert "AI Engineering" in prompt
    assert "[p.{page_number}]" in prompt
    assert "I cannot find information" in prompt


def test_passages_numbered_from_one(builder):
    results = [_result(p) for p in [5, 10, 15]]
    prompt = builder.build_rag_prompt("Q?", results)
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "[3]" in prompt
    assert "[4]" not in prompt


def test_page_numbers_in_prompt(builder):
    results = [_result(42), _result(99)]
    prompt = builder.build_rag_prompt("Q?", results)
    assert "Page 42" in prompt
    assert "Page 99" in prompt


def test_section_title_shown(builder):
    prompt = builder.build_rag_prompt("Q?", [_result(1, section="Chapter 3: RAG")])
    assert "Chapter 3: RAG" in prompt


def test_none_section_title_shows_general(builder):
    prompt = builder.build_rag_prompt("Q?", [_result(1, section=None)])
    assert "General" in prompt


def test_chunk_truncated_to_limit(builder):
    long_text = "word " * 300  # well over 600 chars
    prompt = builder.build_rag_prompt("Q?", [_result(1, text=long_text)])
    # Each passage in the prompt must be at most _CHUNK_PREVIEW_LIMIT chars
    # Verify the long text was cut by checking the prompt doesn't contain more
    # than limit chars of the passage text
    passage_text_in_prompt = long_text[:_CHUNK_PREVIEW_LIMIT]
    assert passage_text_in_prompt in prompt
    assert long_text not in prompt  # full text must NOT appear


def test_all_results_included(builder):
    results = [_result(i) for i in range(1, 6)]
    prompt = builder.build_rag_prompt("Q?", results)
    for i in range(1, 6):
        assert f"Page {i}" in prompt


def test_ordering_question_after_context(builder):
    prompt = builder.build_rag_prompt("My question", [_result(1)])
    context_pos = prompt.index("CONTEXT PASSAGES")
    question_pos = prompt.index("QUESTION:")
    answer_pos = prompt.index("ANSWER (cite")
    assert context_pos < question_pos < answer_pos
