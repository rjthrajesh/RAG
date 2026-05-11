from __future__ import annotations

import pytest

from app.generation.citation_validator import CitationValidator, _CANNOT_FIND
from app.retrieval.vector_store import RetrievalResult


def _result(page: int) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"page_{page}_chunk_0",
        text=f"Text on page {page}.",
        page_number=page,
        section_title=None,
        parent_text=f"Parent text on page {page}.",
        score=1.0,
    )


RESULTS = [_result(10), _result(20), _result(30)]


@pytest.fixture
def validator():
    return CitationValidator()


# ------------------------------------------------------------------
# validate()
# ------------------------------------------------------------------

def test_valid_answer_passes(validator):
    answer = (
        "Foundation models are large neural networks trained on broad data [p.10]. "
        "They can be adapted via fine-tuning or prompting [p.20]. "
        "RAG grounds the model in retrieved evidence [p.30]."
    )
    result = validator.validate(answer, RESULTS)
    assert result.is_valid
    assert set(result.citations_found) == {10, 20, 30}
    assert result.warning is None


def test_uncited_answer_fails(validator):
    # All sentences are long (>15 words) and have no citations
    long_sentence = "This is a sentence that is definitely longer than fifteen words and has absolutely no citation attached to it whatsoever right here. "
    answer = long_sentence * 4  # 4 uncited substantive sentences → 100% uncited
    result = validator.validate(answer, RESULTS)
    assert not result.is_valid
    assert result.warning is not None
    assert len(result.uncited_sentences) > 0


def test_hallucinated_page_stripped(validator):
    answer = "The model uses attention [p.10]. RAG was described on [p.999]."
    valid_pages = {10, 20, 30}
    cleaned = validator.strip_hallucinated_citations(answer, valid_pages)
    assert "[p.999]" not in cleaned
    assert "[p.10]" in cleaned


def test_cannot_find_answer_always_valid(validator):
    answer = f"{_CANNOT_FIND} about this in the provided context."
    result = validator.validate(answer, RESULTS)
    assert result.is_valid
    assert result.warning is None
    assert result.uncited_sentences == []


def test_short_sentences_exempt(validator):
    # Sentences under 15 words are not flagged even without citations
    answer = "RAG is useful. It retrieves documents. Answers are grounded."
    result = validator.validate(answer, RESULTS)
    # All sentences are short — none flagged, validation passes
    assert result.is_valid
    assert result.uncited_sentences == []


def test_citations_found_deduplicates(validator):
    answer = "Same page cited twice [p.10] and again [p.10] and also [p.20]."
    result = validator.validate(answer, RESULTS)
    assert result.citations_found.count(10) == 1  # deduplicated


def test_mixed_cited_uncited_below_threshold_passes(validator):
    # 1 uncited out of 2 substantive = 50% — exactly at threshold, should pass
    cited = "Foundation models are trained on massive internet-scale corpora of text and code [p.10]."
    uncited = "This is another sentence about foundation models that is long enough to matter here today."
    result = validator.validate(cited + " " + uncited, RESULTS)
    # 50% uncited is NOT > threshold (> 0.5), so is_valid=True
    assert result.is_valid


def test_above_threshold_fails(validator):
    # 2 uncited out of 3 substantive sentences (each >15 words) ≈ 67% → invalid
    cited = (
        "Foundation models are very large neural networks pretrained on broad internet-scale "
        "data for generalization across many downstream tasks [p.10]."
    )
    uncited1 = (
        "This extremely long sentence definitely lacks any citation whatsoever and is "
        "certainly well over fifteen words in its total word count."
    )
    uncited2 = (
        "Another very long sentence about retrieval augmented generation pipelines that "
        "does not contain any page citation reference at all."
    )
    result = validator.validate(f"{cited} {uncited1} {uncited2}", RESULTS)
    assert not result.is_valid


def test_strip_removes_only_invalid_pages(validator):
    answer = "Claim A [p.10]. Claim B [p.999]. Claim C [p.20]. Claim D [p.888]."
    cleaned = validator.strip_hallucinated_citations(answer, {10, 20, 30})
    assert "[p.10]" in cleaned
    assert "[p.20]" in cleaned
    assert "[p.999]" not in cleaned
    assert "[p.888]" not in cleaned


def test_empty_results_all_citations_hallucinated(validator):
    answer = "Something happened [p.1]."
    cleaned = validator.strip_hallucinated_citations(answer, set())
    assert "[p.1]" not in cleaned


def test_validate_extracts_correct_page_numbers(validator):
    answer = "Point one [p.10]. Point two [p.30]."
    result = validator.validate(answer, RESULTS)
    assert 10 in result.citations_found
    assert 30 in result.citations_found
