from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

_PAGE_NUMBER_RE = re.compile(r"^\d+$")


@dataclass
class PageDocument:
    page_number: int  # 1-indexed
    text: str
    char_count: int
    has_tables: bool
    has_figures: bool
    section_title: str | None


class PDFParser:
    def parse(self, pdf_path: str) -> list[PageDocument]:
        doc = fitz.open(pdf_path)
        pages: list[PageDocument] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_number = page_idx + 1

            raw_blocks = page.get_text("blocks")
            has_figures = any(b[6] == 1 for b in raw_blocks)
            has_tables = self._detect_tables(page)
            section_title = self._extract_section_title(page)
            cleaned = self._clean_text(page.get_text("text"))

            if len(cleaned) < 50:
                logger.warning(
                    "Page %d yielded only %d chars after cleaning", page_number, len(cleaned)
                )

            pages.append(
                PageDocument(
                    page_number=page_number,
                    text=cleaned,
                    char_count=len(cleaned),
                    has_tables=has_tables,
                    has_figures=has_figures,
                    section_title=section_title,
                )
            )

        doc.close()
        return sorted(pages, key=lambda p: p.page_number)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean_text(self, raw: str) -> str:
        lines = raw.split("\n")
        kept: list[str] = []
        for line in lines:
            stripped = line.strip()
            if _PAGE_NUMBER_RE.match(stripped):
                continue
            if stripped == "AI Engineering":
                continue
            kept.append(stripped)
        return re.sub(r"\s+", " ", " ".join(kept)).strip()

    def _detect_tables(self, page: fitz.Page) -> bool:
        y_coords: list[float] = []
        for drawing in page.get_drawings():
            for item in drawing.get("items", []):
                if item[0] != "l":
                    continue
                p1, p2 = item[1], item[2]
                # Horizontal line: y coords nearly equal, x span meaningful
                if abs(p1.y - p2.y) < 2 and abs(p1.x - p2.x) > 20:
                    y_coords.append(p1.y)

        if len(y_coords) < 3:
            return False

        y_coords.sort()
        for i in range(len(y_coords) - 2):
            if y_coords[i + 2] - y_coords[i] <= 100:
                return True
        return False

    def _extract_section_title(self, page: fitz.Page) -> str | None:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:  # 0 = text
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("size", 0) >= 14:
                        title = span.get("text", "").strip()
                        if title:
                            return title
        return None
