from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from app.ingestion.embedder import Embedder
from app.retrieval.bm25_store import BM25Store
from app.retrieval.vector_store import RetrievalResult, VectorStore

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard constant — insensitive to score scale, robust across datasets


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank)


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        embedder: Embedder,
    ) -> None:
        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        # 1. Run both retrievers in rank order
        query_embedding = self._embedder.embed_query(query)
        dense_results: list[RetrievalResult] = self._vector_store.query(query_embedding, top_k=top_k)
        sparse_results: list[RetrievalResult] = self._bm25_store.query(query, top_k=top_k)

        # 2. Build rank maps — chunk_id → (rank, result)
        dense_rank: dict[str, tuple[int, RetrievalResult]] = {
            r.chunk_id: (i + 1, r) for i, r in enumerate(dense_results)
        }
        sparse_rank: dict[str, tuple[int, RetrievalResult]] = {
            r.chunk_id: (i + 1, r) for i, r in enumerate(sparse_results)
        }

        # 3. Accumulate RRF scores across all unique chunk_ids
        all_ids = set(dense_rank) | set(sparse_rank)
        rrf_scores: dict[str, float] = {}
        for cid in all_ids:
            score = 0.0
            if cid in dense_rank:
                score += _rrf_score(dense_rank[cid][0])
            if cid in sparse_rank:
                score += _rrf_score(sparse_rank[cid][0])
            rrf_scores[cid] = score

        # 4. Tag retrieval_source and attach fused score
        fused: list[RetrievalResult] = []
        for cid in all_ids:
            in_dense = cid in dense_rank
            in_sparse = cid in sparse_rank
            source = "both" if (in_dense and in_sparse) else ("dense" if in_dense else "sparse")
            # Take the result object from whichever list has it
            base_result = (dense_rank[cid][1] if in_dense else sparse_rank[cid][1])
            fused.append(replace(base_result, score=rrf_scores[cid], retrieval_source=source))

        # 5. Sort by RRF score descending and take top_k
        fused.sort(key=lambda r: r.score, reverse=True)
        top = fused[:top_k]

        # 6. Log source breakdown
        n_both   = sum(1 for r in top if r.retrieval_source == "both")
        n_dense  = sum(1 for r in top if r.retrieval_source == "dense")
        n_sparse = sum(1 for r in top if r.retrieval_source == "sparse")
        logger.info(
            "RRF retrieval: %d results — both=%d, dense-only=%d, sparse-only=%d",
            len(top), n_both, n_dense, n_sparse,
        )

        return top
