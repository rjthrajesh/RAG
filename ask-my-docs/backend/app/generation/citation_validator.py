from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.retrieval.vector_store import RetrievalResult

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[p\.(\d+)\]")
_CANNOT_FIND = "I cannot find information"
_MIN_WORDS_FOR_CITATION = 15
_UNCITED_THRESHOLD = 0.5  # >50% uncited → invalid


@dataclass
class ValidationResult:
    is_valid: bool
    citations_found: list[int] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    warning: str | None = None


class CitationValidator:
    def validate(
        self, answer: str, results: list[RetrievalResult]
    ) -> ValidationResult:
        # Rule 4: "cannot find" answer is always valid — skip all checks
        if _CANNOT_FIND in answer:
            return ValidationResult(is_valid=True)

        valid_pages = {r.page_number for r in results}

        # Rule 1: extract all cited page numbers
        cited_pages = [int(m) for m in _CITATION_RE.findall(answer)]

        # Rule 2: flag any cited page not in results (hallucinated)
        hallucinated = [p for p in cited_pages if p not in valid_pages]
        if hallucinated:
            logger.warning("Hallucinated page citations: %s", hallucinated)

        # Rule 3: find substantive sentences with no inline citation
        sentences = [s.strip() for s in answer.split(". ") if s.strip()]
        uncited: list[str] = []
        for sentence in sentences:
            is_long = len(sentence.split()) > _MIN_WORDS_FOR_CITATION
            has_citation = bool(_CITATION_RE.search(sentence))
            if is_long and not has_citation:
                uncited.append(sentence)

        # Rule 5: >50% of substantive sentences uncited → invalid
        substantive = [s for s in sentences if len(s.split()) > _MIN_WORDS_FOR_CITATION]
        uncited_ratio = len(uncited) / len(substantive) if substantive else 0.0
        is_valid = uncited_ratio <= _UNCITED_THRESHOLD

        warning: str | None = None
        if not is_valid:
            warning = (
                f"{len(uncited)} of {len(substantive)} substantive sentences "
                f"lack citations ({uncited_ratio:.0%})"
            )
            logger.warning("Citation validation failed: %s", warning)

        return ValidationResult(
            is_valid=is_valid,
            citations_found=list(dict.fromkeys(cited_pages)),  # deduplicated, order-preserving
            uncited_sentences=uncited,
            warning=warning,
        )

    def strip_hallucinated_citations(self, answer: str, valid_pages: set[int]) -> str:
        def _remove_if_invalid(m: re.Match) -> str:
            page = int(m.group(1))
            return m.group(0) if page in valid_pages else ""

        return _CITATION_RE.sub(_remove_if_invalid, answer)
