from __future__ import annotations

import logging
import time

from fastembed import TextEmbedding

from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_chunks(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        texts = [c.text for c in chunks]
        t0 = time.time()

        # fastembed handles batching internally and returns a generator of numpy arrays
        embeddings = [e.tolist() for e in self.model.embed(texts)]

        elapsed = time.time() - t0
        throughput = len(chunks) / elapsed if elapsed > 0 else 0.0
        logger.info(
            "Embedded %d chunks in %.2fs (%.1f chunks/sec)",
            len(chunks),
            elapsed,
            throughput,
        )
        return list(zip(chunks, embeddings))

    def embed_query(self, query: str) -> list[float]:
        # query_embed applies the BGE query prefix automatically
        return next(iter(self.model.query_embed([query]))).tolist()
