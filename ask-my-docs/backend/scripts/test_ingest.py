#!/usr/bin/env python
"""
Manual ingestion smoke-test. Runs the full pipeline in-process:
  PDFParser → RecursiveChunker → Embedder → ChromaDB (in-memory) + BM25

Usage:
  .rag/bin/python scripts/test_ingest.py path/to/file.pdf
  .rag/bin/python scripts/test_ingest.py path/to/file.pdf --query "What is RAG?"

No Docker or server needed — uses an ephemeral (in-memory) ChromaDB client.

Note: The BGE embedding model (~1.3 GB) is downloaded on first run from HuggingFace.
Subsequent runs use the local cache (~/.cache/huggingface).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure app/ is importable when running from backend/
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb

from app.config import settings
from app.ingestion.chunker import RecursiveChunker
from app.ingestion.embedder import Embedder
from app.ingestion.pdf_parser import PDFParser
from app.retrieval.bm25_store import BM25Store
from app.retrieval.vector_store import RetrievalResult, VectorStore


# ------------------------------------------------------------------
# In-process ChromaDB that needs no server
# ------------------------------------------------------------------

class EphemeralVectorStore(VectorStore):
    """Override the HTTP client with an in-memory ephemeral client."""

    def __init__(self) -> None:
        self._client = chromadb.EphemeralClient()
        self._collection = self._get_or_create_collection()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _hr(char: str = "─", width: int = 60) -> None:
    print(char * width)


def _print_page_summary(pages) -> None:
    _hr()
    print(f"  PARSED  {len(pages)} pages")
    _hr()
    for p in pages:
        title = p.section_title or "(no title)"
        flags = []
        if p.has_tables:
            flags.append("tables")
        if p.has_figures:
            flags.append("figures")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  p{p.page_number:>3}  {p.char_count:>5} chars  {title[:45]}{flag_str}")


def _print_chunk_summary(chunks) -> None:
    _hr()
    sizes = [c.char_count for c in chunks]
    print(
        f"  CHUNKED  {len(chunks)} chunks — "
        f"avg {sum(sizes)//len(sizes)}, "
        f"min {min(sizes)}, "
        f"max {max(sizes)} chars"
    )
    _hr()
    for c in chunks[:10]:
        preview = c.text[:60].replace("\n", " ")
        print(f"  {c.chunk_id:<30}  {c.char_count:>4} chars  \"{preview}...\"")
    if len(chunks) > 10:
        print(f"  … and {len(chunks) - 10} more chunks")


def _print_retrieval(query: str, dense: list, sparse: list) -> None:
    _hr()
    print(f'  QUERY  "{query}"')
    _hr("·")
    print("  Dense (vector) top-3:")
    for r in dense[:3]:
        print(f"    [p.{r.page_number}] score={r.score:.3f}  {r.text[:70].replace(chr(10), ' ')}...")
    print()
    print("  Sparse (BM25) top-3:")
    for r in sparse[:3]:
        print(f"    [p.{r.page_number}] score={r.score:.3f}  {r.text[:70].replace(chr(10), ' ')}...")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", help="Path to the PDF file to ingest")
    parser.add_argument("--query", "-q", default=None, help="Optional query to test retrieval after ingestion")
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    t_start = time.time()

    # 1. Parse
    print(f"\nParsing {pdf_path.name} …")
    pages = PDFParser().parse(str(pdf_path))
    _print_page_summary(pages)

    # 2. Chunk
    print(f"\nChunking (size={args.chunk_size}, overlap={args.chunk_overlap}) …")
    chunks = RecursiveChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    ).chunk(pages)
    _print_chunk_summary(chunks)

    # 3. Embed
    print(f"\nEmbedding {len(chunks)} chunks with {settings.embed_model_name} …")
    print("  (downloads model on first run — may take a few minutes)")
    embedder = Embedder(model_name=settings.embed_model_name)
    chunk_embeddings = embedder.embed_chunks(chunks)

    # 4. Store — vector (in-memory)
    print("\nStoring in ephemeral ChromaDB …")
    vs = EphemeralVectorStore()
    vs.add_chunks(
        [ce[0] for ce in chunk_embeddings],
        [ce[1] for ce in chunk_embeddings],
    )

    # 5. Store — BM25 (in-memory, no disk write)
    print("Building BM25 index …")
    bm25 = BM25Store()
    bm25.build(chunks)

    elapsed = time.time() - t_start
    _hr("═")
    print(f"  Done in {elapsed:.1f}s — {len(pages)} pages → {len(chunks)} chunks indexed")
    _hr("═")

    # 6. Optional retrieval test
    if args.query:
        q_emb = embedder.embed_query(args.query)
        dense_results = vs.query(q_emb, top_k=args.top_k)
        sparse_results = bm25.query(args.query, top_k=args.top_k)
        _print_retrieval(args.query, dense_results, sparse_results)
    else:
        print("\nTip: add --query \"your question\" to test retrieval immediately.\n")


if __name__ == "__main__":
    main()
