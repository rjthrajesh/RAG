from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "ask_my_docs"


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    page_number: int
    section_title: str | None
    parent_text: str
    score: float
    retrieval_source: str = "dense"  # "dense" | "sparse" | "both"
    rerank_score: float | None = None


def _point_id(chunk_id: str) -> str:
    """Deterministic UUID from a chunk_id string — Qdrant supports UUID point IDs."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


class VectorStore:
    def __init__(self) -> None:
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        self._vector_size = settings.qdrant_vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self._client.get_collection(_COLLECTION_NAME)
        except Exception:
            self._client.create_collection(
                collection_name=_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d, distance=cosine)",
                _COLLECTION_NAME,
                self._vector_size,
            )

    _WRITE_BATCH = 100

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=_point_id(c.chunk_id),
                vector=emb,
                payload={
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "page_number": c.page_number,
                    "section_title": c.section_title or "",
                    "parent_text": c.parent_text,
                },
            )
            for c, emb in zip(chunks, embeddings)
        ]
        for i in range(0, len(points), self._WRITE_BATCH):
            self._client.upsert(
                collection_name=_COLLECTION_NAME,
                points=points[i : i + self._WRITE_BATCH],
            )
        logger.info("Upserted %d chunks into Qdrant", len(chunks))

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        hits = self._client.search(
            collection_name=_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        return [
            RetrievalResult(
                chunk_id=h.payload["chunk_id"],
                text=h.payload["text"],
                page_number=h.payload["page_number"],
                section_title=h.payload["section_title"] or None,
                parent_text=h.payload["parent_text"],
                score=h.score,
            )
            for h in hits
        ]

    def delete_all(self) -> None:
        self._client.delete_collection(_COLLECTION_NAME)
        self._client.create_collection(
            collection_name=_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=self._vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Deleted and recreated Qdrant collection '%s'", _COLLECTION_NAME)
