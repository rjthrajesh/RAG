from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.generation.citation_validator import CitationValidator
from app.generation.llm_client import GroqClient, OllamaClient
from app.generation.prompt_builder import PromptBuilder
from app.ingestion.chunker import RecursiveChunker
from app.ingestion.embedder import Embedder
from app.ingestion.pdf_parser import PDFParser
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

app = FastAPI(title="Ask My Docs", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://rag-xi-ashy.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job registry — single-user system, no persistence needed
_jobs: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Lazy singletons — models load once, reused across requests
# ---------------------------------------------------------------------------
_vector_store: VectorStore | None = None
_bm25_store: BM25Store | None = None
_embedder: Embedder | None = None
_retriever: HybridRetriever | None = None
_reranker: CrossEncoderReranker | None = None
_llm_client: OllamaClient | None = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _get_bm25_store() -> BM25Store:
    global _bm25_store
    if _bm25_store is None:
        store = BM25Store()
        # Auto-load persisted index so queries work after server restart
        if Path(settings.bm25_index_path).exists():
            store.load(settings.bm25_index_path)
        _bm25_store = store
    return _bm25_store


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder(model_name=settings.embed_model_name)
    return _embedder


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(
            vector_store=_get_vector_store(),
            bm25_store=_get_bm25_store(),
            embedder=_get_embedder(),
        )
    return _retriever


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def _get_llm_client() -> OllamaClient | GroqClient:
    global _llm_client
    if _llm_client is None:
        if settings.llm_provider == "groq":
            _llm_client = GroqClient(
                api_key=settings.groq_api_key,
                model=settings.groq_model,
            )
        else:
            _llm_client = OllamaClient(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
            )
    return _llm_client


# ---------------------------------------------------------------------------
# Background ingestion task
# ---------------------------------------------------------------------------

def _run_ingestion(job_id: str, pdf_path: str) -> None:
    global _retriever  # reset so the next query picks up the new index
    try:
        t0 = time.time()
        _jobs[job_id]["status"] = "processing"

        _jobs[job_id]["progress"] = 0.10
        pages = PDFParser().parse(pdf_path)

        _jobs[job_id]["progress"] = 0.30
        chunks = RecursiveChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ).chunk(pages)

        _jobs[job_id]["progress"] = 0.50
        embedder = _get_embedder()
        chunk_embeddings = embedder.embed_chunks(chunks)

        _jobs[job_id]["progress"] = 0.70
        vs = _get_vector_store()
        vs.delete_all()
        vs.add_chunks(
            [ce[0] for ce in chunk_embeddings],
            [ce[1] for ce in chunk_embeddings],
        )

        _jobs[job_id]["progress"] = 0.90
        bm25 = _get_bm25_store()
        bm25.build(chunks)
        bm25.save(settings.bm25_index_path)
        _retriever = None  # force rebuild with fresh stores

        elapsed = round(time.time() - t0, 2)
        _jobs[job_id].update(
            {
                "status": "done",
                "progress": 1.0,
                "pages": len(pages),
                "chunks": len(chunks),
                "duration_seconds": elapsed,
            }
        )
        logger.info(
            "Ingestion job %s complete: %d pages, %d chunks in %.2fs",
            job_id, len(pages), len(chunks), elapsed,
        )
    except Exception:
        logger.exception("Ingestion job %s failed", job_id)
        _jobs[job_id]["status"] = "failed"
    finally:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    top_k_retrieve: int = settings.top_k_retrieve
    top_k_rerank: int = settings.top_k_rerank


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _query_stream(req: QueryRequest) -> AsyncIterator[str]:
    try:
        retriever = _get_retriever()
        results = retriever.retrieve(req.question, top_k=req.top_k_retrieve)

        reranker = _get_reranker()
        reranked = reranker.rerank(req.question, results, top_k=req.top_k_rerank)

        prompt = PromptBuilder().build_rag_prompt(req.question, reranked)

        full_answer = ""
        async for delta in _get_llm_client().stream_completion(prompt):
            full_answer += delta
            yield _sse({"type": "delta", "text": delta})

        validation = CitationValidator().validate(full_answer, reranked)

        sources = [
            {
                "page_number": r.page_number,
                "section_title": r.section_title,
                "text_preview": r.text[:200],
                "rerank_score": r.rerank_score,
                "retrieval_source": r.retrieval_source,
            }
            for r in reranked
        ]
        yield _sse({"type": "sources", "sources": sources})
        yield _sse({
            "type": "done",
            "citation_valid": validation.is_valid,
            "warning": validation.warning,
        })

    except Exception as exc:
        logger.exception("Query streaming failed")
        yield _sse({"type": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/query")
async def query(req: QueryRequest) -> StreamingResponse:
    return StreamingResponse(
        _query_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
        },
    )


@app.post("/ingest", status_code=202)
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a .pdf")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 50 MB limit")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(content)
    tmp.close()

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "processing", "progress": 0.0}
    background_tasks.add_task(_run_ingestion, job_id, tmp.name)

    return {"job_id": job_id, "status": "processing"}


@app.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    response: dict[str, Any] = {
        "status": job["status"],
        "progress": job.get("progress", 0.0),
    }
    if job["status"] == "done":
        response.update({
            "pages": job["pages"],
            "chunks": job["chunks"],
            "duration_seconds": job["duration_seconds"],
        })
    elif job["status"] == "failed":
        response["error"] = job.get("error", "Unknown error")
    return response


@app.get("/health")
async def health() -> dict:
    chroma_ok = False
    try:
        _get_vector_store()._client.heartbeat()
        chroma_ok = True
    except Exception:
        pass

    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        pass

    return {"status": "ok", "ollama": ollama_ok, "chroma": chroma_ok}
