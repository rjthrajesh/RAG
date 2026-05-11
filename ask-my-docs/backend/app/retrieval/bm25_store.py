from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.ingestion.chunker import Chunk
from app.retrieval.vector_store import RetrievalResult

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


class BM25Store:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] = []
        self._chunk_ids: list[str] = []
        self._chunk_metadata: list[dict] = []

    def build(self, chunks: list[Chunk]) -> None:
        self._corpus_tokens = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._chunk_metadata = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "page_number": c.page_number,
                "section_title": c.section_title,
                "parent_text": c.parent_text,
            }
            for c in chunks
        ]
        logger.info("Built BM25 index with %d chunks", len(chunks))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "corpus_tokens": self._corpus_tokens,
                    "chunk_ids": self._chunk_ids,
                    "chunk_metadata": self._chunk_metadata,
                },
                f,
            )
        logger.info("Saved BM25 index to %s", path)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self._corpus_tokens = data["corpus_tokens"]
        self._chunk_ids = data["chunk_ids"]
        self._chunk_metadata = data["chunk_metadata"]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        logger.info("Loaded BM25 index from %s (%d chunks)", path, len(self._chunk_ids))

    def query(self, query_text: str, top_k: int) -> list[RetrievalResult]:
        if self._bm25 is None:
            return []

        tokens = _tokenize(query_text)
        scores = self._bm25.get_scores(tokens)

        max_score = float(max(scores)) if len(scores) > 0 and max(scores) > 0 else 1.0
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: list[RetrievalResult] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            meta = self._chunk_metadata[idx]
            results.append(
                RetrievalResult(
                    chunk_id=meta["chunk_id"],
                    text=meta["text"],
                    page_number=meta["page_number"],
                    section_title=meta["section_title"],
                    parent_text=meta["parent_text"],
                    score=float(scores[idx]) / max_score,
                    retrieval_source="sparse",
                )
            )
        return results
