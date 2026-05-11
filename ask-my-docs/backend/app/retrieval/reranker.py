from __future__ import annotations

import logging
import time
from dataclasses import replace

from sentence_transformers import CrossEncoder

from app.retrieval.vector_store import RetrievalResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            logger.info("Loading cross-encoder model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        # Use parent_text for wider context — better reranking signal than chunk text alone
        pairs = [(query, r.parent_text) for r in results]

        t0 = time.time()
        scores: list[float] = self.model.predict(pairs).tolist()
        elapsed = time.time() - t0

        logger.info(
            "Reranked %d results in %.2fs — scores min=%.3f max=%.3f mean=%.3f",
            len(results),
            elapsed,
            min(scores),
            max(scores),
            sum(scores) / len(scores),
        )

        # Attach rerank_score and sort descending
        scored = [
            replace(result, rerank_score=score)
            for result, score in zip(results, scores)
        ]
        scored.sort(key=lambda r: r.rerank_score, reverse=True)  # type: ignore[arg-type]

        return scored[:top_k]
