from __future__ import annotations

import logging
import time

from sentence_transformers import SentenceTransformer

from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_BATCH_SIZE = 128


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_chunks(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        texts = [c.text for c in chunks]
        t0 = time.time()

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            vecs = self.model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.extend(vecs.tolist())

        elapsed = time.time() - t0
        throughput = len(chunks) / elapsed if elapsed > 0 else 0.0
        logger.info(
            "Embedded %d chunks in %.2fs (%.1f chunks/sec)",
            len(chunks),
            elapsed,
            throughput,
        )
        return list(zip(chunks, all_embeddings))

    def embed_query(self, query: str) -> list[float]:
        is_bge = "bge" in self.model_name.lower()
        text = (_BGE_QUERY_PREFIX + query) if is_bge else query
        vec = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.tolist()
