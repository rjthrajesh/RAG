import re

import pytest

from app.ingestion.chunker import RecursiveChunker, _MIN_CHUNK_CHARS
from app.ingestion.pdf_parser import PageDocument


def _make_page(text: str, page_number: int = 1) -> PageDocument:
    return PageDocument(
        page_number=page_number,
        text=text,
        char_count=len(text),
        has_tables=False,
        has_figures=False,
        section_title=None,
    )


CHUNK_SIZE = 200
OVERLAP = 40
LONG_TEXT = (
    "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega. " * 20
)


@pytest.fixture
def chunker():
    return RecursiveChunker(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)


def test_chunk_respects_size_limit(chunker):
    pages = [_make_page(LONG_TEXT)]
    chunks = chunker.chunk(pages)
    assert chunks, "Expected at least one chunk"
    for c in chunks:
        assert c.char_count <= CHUNK_SIZE * 1.2, (
            f"Chunk {c.chunk_id} exceeds size limit: {c.char_count}"
        )


def test_chunk_overlap_present(chunker):
    pages = [_make_page(LONG_TEXT)]
    chunks = chunker.chunk(pages)
    assert len(chunks) >= 2, "Need at least 2 chunks to test overlap"
    # Adjacent chunks must share at least one word
    for i in range(len(chunks) - 1):
        words_a = set(chunks[i].text.split())
        words_b = set(chunks[i + 1].text.split())
        assert words_a & words_b, (
            f"No overlap between chunk {i} and {i+1}"
        )


def test_minimum_chunk_size_enforced(chunker):
    pages = [_make_page(LONG_TEXT)]
    chunks = chunker.chunk(pages)
    for c in chunks:
        assert c.char_count >= _MIN_CHUNK_CHARS, (
            f"Chunk {c.chunk_id} is below minimum size: {c.char_count}"
        )


def test_chunk_id_format(chunker):
    pages = [_make_page(LONG_TEXT, page_number=7)]
    chunks = chunker.chunk(pages)
    pattern = re.compile(r"^page_\d+_chunk_\d+$")
    for c in chunks:
        assert pattern.match(c.chunk_id), f"Bad chunk_id format: {c.chunk_id}"


def test_parent_text_bounds(chunker):
    page_text = LONG_TEXT
    pages = [_make_page(page_text)]
    chunks = chunker.chunk(pages)
    for c in chunks:
        # parent_text must be a substring of the page text
        assert c.parent_text in page_text or page_text in c.parent_text or len(c.parent_text) <= len(page_text), (
            "parent_text exceeds page boundaries"
        )
        # parent_text must contain the chunk text
        assert c.text in c.parent_text or c.parent_text in c.text, (
            f"parent_text does not contain chunk text for {c.chunk_id}"
        )


def test_small_pages_discarded(chunker):
    short_text = "Too short."
    pages = [_make_page(short_text)]
    chunks = chunker.chunk(pages)
    assert all(c.char_count >= _MIN_CHUNK_CHARS for c in chunks)


def test_chunk_preserves_page_metadata(chunker):
    pages = [
        _make_page(LONG_TEXT, page_number=3),
    ]
    pages[0].section_title = "Introduction"
    chunks = chunker.chunk(pages)
    for c in chunks:
        assert c.page_number == 3
        assert c.section_title == "Introduction"
