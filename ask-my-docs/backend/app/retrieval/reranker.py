from __future__ import annotations

import logging
import time
from dataclasses import replace

from flashrank import Ranker, RerankRequest

from app.retrieval.vector_store import RetrievalResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "ms-marco-MiniLM-L-12-v2"
_CACHE_DIR = "/root/.cache/flashrank"


class CrossEncoderReranker:
    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._ranker: Ranker | None = None

    @property
    def ranker(self) -> Ranker:
        if self._ranker is None:
            logger.info("Loading reranker model: %s", self.model_name)
            self._ranker = Ranker(model_name=self.model_name, cache_dir=_CACHE_DIR)
        return self._ranker

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        passages = [{"id": i, "text": r.parent_text} for i, r in enumerate(results)]

        t0 = time.time()
        reranked_passages = self.ranker.rerank(RerankRequest(query=query, passages=passages))
        elapsed = time.time() - t0

        scores = [p["score"] for p in reranked_passages]
        logger.info(
            "Reranked %d results in %.2fs — scores min=%.3f max=%.3f mean=%.3f",
            len(results),
            elapsed,
            min(scores),
            max(scores),
            sum(scores) / len(scores),
        )

        # flashrank returns passages sorted by score descending; map back by id
        scored = [
            replace(results[p["id"]], rerank_score=float(p["score"]))
            for p in reranked_passages
        ]
        return scored[:top_k]
