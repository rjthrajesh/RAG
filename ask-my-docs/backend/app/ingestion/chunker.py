from __future__ import annotations

import logging
from dataclasses import dataclass

from app.ingestion.pdf_parser import PageDocument

logger = logging.getLogger(__name__)

_SEPARATORS = ["\n\n", "\n", ". ", " "]
_MIN_CHUNK_CHARS = 100


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_number: int
    section_title: str | None
    char_count: int
    parent_text: str  # 128-char window before+after chunk, capped at page boundaries


class RecursiveChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, pages: list[PageDocument]) -> list[Chunk]:
        all_chunks: list[Chunk] = []

        for page in pages:
            if not page.text:
                continue
            splits = self._split_text(page.text, _SEPARATORS)
            merged = self._merge_with_overlap(splits)

            for idx, text in enumerate(merged):
                if len(text) < _MIN_CHUNK_CHARS:
                    continue
                all_chunks.append(
                    Chunk(
                        chunk_id=f"page_{page.page_number}_chunk_{idx}",
                        text=text,
                        page_number=page.page_number,
                        section_title=page.section_title,
                        char_count=len(text),
                        parent_text=self._build_parent_text(page.text, text),
                    )
                )

        if all_chunks:
            sizes = [c.char_count for c in all_chunks]
            logger.info(
                "Chunking complete: %d chunks — avg=%d, min=%d, max=%d",
                len(all_chunks),
                sum(sizes) // len(sizes),
                min(sizes),
                max(sizes),
            )
        return all_chunks

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if not separators or len(text) <= self.chunk_size:
            return [text] if text else []

        sep, rest_seps = separators[0], separators[1:]
        parts = text.split(sep)
        result: list[str] = []
        for part in parts:
            if not part:
                continue
            if len(part) > self.chunk_size:
                result.extend(self._split_text(part, rest_seps))
            else:
                result.append(part)
        return result

    def _merge_with_overlap(self, splits: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for split in splits:
            split_len = len(split)
            sep_cost = 1 if current else 0  # single space used as join separator

            if current and current_len + sep_cost + split_len > self.chunk_size:
                chunks.append(" ".join(current))
                # Trim from the front until we're within the overlap budget
                while current and current_len > self.chunk_overlap:
                    removed = current.pop(0)
                    current_len -= len(removed) + 1

            current.append(split)
            current_len += split_len + sep_cost

        if current:
            chunks.append(" ".join(current))

        return chunks

    # ------------------------------------------------------------------
    # Parent context window
    # ------------------------------------------------------------------

    def _build_parent_text(self, page_text: str, chunk_text: str) -> str:
        pos = page_text.find(chunk_text)
        if pos == -1:
            return chunk_text
        start = max(0, pos - 128)
        end = min(len(page_text), pos + len(chunk_text) + 128)
        return page_text[start:end]
