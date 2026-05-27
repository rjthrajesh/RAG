from __future__ import annotations

import logging
from dataclasses import dataclass

import chromadb

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


class VectorStore:
    def __init__(self) -> None:
        self._client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    _WRITE_BATCH = 500

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        metadatas = [
            {
                "page_number": c.page_number,
                "section_title": c.section_title or "",
                "chunk_id": c.chunk_id,
                "parent_text": c.parent_text,
            }
            for c in chunks
        ]
        for i in range(0, len(chunks), self._WRITE_BATCH):
            batch_chunks = chunks[i : i + self._WRITE_BATCH]
            self._collection.add(
                ids=[c.chunk_id for c in batch_chunks],
                documents=[c.text for c in batch_chunks],
                embeddings=embeddings[i : i + self._WRITE_BATCH],
                metadatas=metadatas[i : i + self._WRITE_BATCH],
            )
        logger.info("Stored %d chunks in ChromaDB", len(chunks))

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        output: list[RetrievalResult] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(
                RetrievalResult(
                    chunk_id=meta["chunk_id"],
                    text=doc,
                    page_number=meta["page_number"],
                    section_title=meta["section_title"] or None,
                    parent_text=meta["parent_text"],
                    score=1.0 - dist,
                )
            )
        return output

    def delete_all(self) -> None:
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._get_or_create_collection()
        logger.info("Deleted and recreated ChromaDB collection")
