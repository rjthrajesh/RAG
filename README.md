# Ask My Docs

A production-grade Retrieval-Augmented Generation (RAG) system for grounded question-answering over PDF documents. Upload a document, ask questions, and get answers with page-level citations — backed by hybrid retrieval, cross-encoder reranking, and an automated RAGAS evaluation pipeline.

---

## Features

- **PDF ingestion** via drag-and-drop with real-time progress
- **Hybrid retrieval** — BM25 sparse search + BGE dense embeddings fused with Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking** using `ms-marco-MiniLM-L-6-v2` over parent context windows
- **Streaming answers** with mandatory page citations (`[p.N]`) via Server-Sent Events
- **Citation panel** — click any source chip to expand the retrieved passage
- **RAGAS evaluation pipeline** with automated CI quality gates (faithfulness ≥ 0.75, answer relevancy ≥ 0.70)
- **GitHub Actions CI/CD** — lint, type-check, unit tests, and eval on every PR

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Next.js Frontend                    │
│   IngestionUploader → ChatWindow → CitationPanel        │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP / SSE
┌────────────────────────▼────────────────────────────────┐
│                    FastAPI Backend                       │
│                                                         │
│  POST /ingest          POST /query (SSE stream)         │
│       │                      │                          │
│  ┌────▼────┐           ┌─────▼──────┐                   │
│  │Ingestion│           │ Retrieval  │                   │
│  │ Pipeline│           │  Pipeline  │                   │
│  │         │           │            │                   │
│  │PDFParser│     ┌─────┤HybridRetri-│                   │
│  │Chunker  │     │     │ever (RRF)  │                   │
│  │Embedder │     │  ChromaDB  BM25  │                   │
│  │VectorSt.│     │     │            │                   │
│  │BM25Store│     └─────┤CrossEncoder│                   │
│  └─────────┘           │  Reranker  │                   │
│                        └─────┬──────┘                   │
│                        ┌─────▼──────┐                   │
│                        │Generation  │                   │
│                        │PromptBldr  │                   │
│                        │GroqClient  │                   │
│                        └────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI 0.111, Uvicorn, Python 3.11 |
| Frontend | Next.js 14.2, React 18, TypeScript, Tailwind CSS |
| PDF Parsing | PyMuPDF |
| Embeddings | sentence-transformers, `BAAI/bge-large-en-v1.5` |
| Vector Store | ChromaDB 0.5.3 (HTTP, Docker) |
| Sparse Retrieval | rank-bm25 |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM Inference | Groq Cloud (`llama-3.1-8b-instant`) |
| Evaluation | RAGAS 0.1.9, LangChain Core |
| HTTP Client | httpx |
| CI/CD | GitHub Actions |

---

## Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for ChromaDB)
- A [Groq API key](https://console.groq.com) (free tier works)

---

## Local Setup

### 1. Clone and configure environment

```bash
git clone <repo-url>
cd ask-my-docs
```

Create a `.env` file at the project root:

```env
# LLM
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8001

# Embeddings & chunking
EMBED_MODEL_NAME=BAAI/bge-base-en-v1.5
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# Retrieval
TOP_K_RETRIEVE=20
TOP_K_RERANK=5
BM25_INDEX_PATH=./data/bm25_index.json

# Evaluation thresholds
EVAL_FAITHFULNESS_THRESHOLD=0.75
EVAL_ANSWER_RELEVANCY_THRESHOLD=0.70
```

### 2. Start ChromaDB

```bash
docker compose up -d chromadb
```

### 3. Install dependencies

```bash
# Backend
python3.11 -m venv backend/.rag
source backend/.rag/bin/activate
pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install
```

### 4. Run the development server

```bash
make dev
```

This starts the FastAPI backend on `http://localhost:8000` and the Next.js frontend on `http://localhost:3000` in parallel.

---

## Usage

1. Open `http://localhost:3000`
2. Drag and drop a PDF onto the upload panel — ingestion runs automatically
3. Type a question in the chat input
4. Answers stream in with `[p.N]` citations; click a source chip to read the passage

---

## Project Structure

```
ask-my-docs/
├── backend/
│   ├── app/
│   │   ├── config.py               # Pydantic settings (reads .env)
│   │   ├── main.py                 # FastAPI app, /ingest and /query endpoints
│   │   ├── ingestion/
│   │   │   ├── pdf_parser.py       # PyMuPDF page extraction
│   │   │   ├── chunker.py          # Recursive chunker with parent context
│   │   │   └── embedder.py         # BGE embedder (batched, normalised)
│   │   ├── retrieval/
│   │   │   ├── vector_store.py     # ChromaDB HNSW cosine collection
│   │   │   ├── bm25_store.py       # BM25 index (serialised JSON)
│   │   │   ├── hybrid_retriever.py # RRF fusion of dense + sparse
│   │   │   └── reranker.py         # Cross-encoder reranker (parent_text)
│   │   ├── generation/
│   │   │   ├── llm_client.py       # GroqClient + OllamaClient (sync+async)
│   │   │   └── prompt_builder.py   # System prompt + citation instructions
│   │   └── evaluation/
│   │       ├── eval_dataset.py     # Golden dataset loader
│   │       └── ragas_runner.py     # RAGAS evaluation runner
│   └── tests/
│       ├── unit/                   # Unit tests (pytest)
│       └── eval/
│           ├── golden_dataset.json # 25 Q&A pairs with ground truth
│           └── test_ragas.py       # Pytest gate: asserts metric thresholds
├── frontend/
│   └── src/
│       ├── app/
│       │   ├── page.tsx            # Main layout (upload → chat)
│       │   └── api/query/route.ts  # Route Handler: pipes SSE stream
│       ├── components/
│       │   ├── ChatWindow.tsx
│       │   ├── CitationPanel.tsx
│       │   ├── IngestionUploader.tsx
│       │   └── MessageBubble.tsx
│       ├── hooks/useChat.ts        # SSE consumer hook
│       └── lib/api.ts              # Backend API calls
├── .github/workflows/
│   ├── ci.yml                      # Lint + type-check + unit tests
│   └── eval.yml                    # RAGAS eval gate on PR to main
├── docker-compose.yml              # ChromaDB + backend + frontend
└── Makefile                        # dev, test, eval targets
```

---

## Retrieval Pipeline

```
Query
  │
  ├─► BGE embed (query prefix) ──► ChromaDB cosine search ─► top-20 (dense)
  │
  └─► BM25 tokenise ─────────────► BM25 ranked list ────────► top-20 (sparse)
                                          │
                                   RRF fusion (k=60)
                                          │
                                     top-20 merged
                                          │
                              CrossEncoder(query, parent_text)
                                          │
                                      top-5 reranked
                                          │
                                   PromptBuilder
                                          │
                               Groq LLM (streaming)
```

**RRF score:** `1 / (k + rank)` summed across retrievers — scale-invariant, no score normalisation needed.

**Reranker input:** Uses the 256-char parent context window (before + after the chunk boundary) rather than the raw chunk text, giving the cross-encoder wider context for scoring.

---

## Evaluation

The evaluation pipeline scores 25 golden Q&A pairs end-to-end.

### Run evaluation

```bash
cd backend
PYTHONPATH=. .rag/bin/python -m app.evaluation.ragas_runner
```

Writes `backend/eval_report.json` with all four metric scores.

### Assert thresholds

```bash
PYTHONPATH=. .rag/bin/pytest tests/eval/test_ragas.py -v
```

### Latest scores

| Metric | Score | Threshold |
|---|---|---|
| Faithfulness | 0.773 | ≥ 0.75 ✅ |
| Answer Relevancy | 0.790 | ≥ 0.70 ✅ |
| Context Precision | 0.367 | — |
| Context Recall | 0.657 | — |

> The evaluation uses `meta-llama/llama-4-scout-17b-16e-instruct` as the RAGAS judge model (separate from the generation model) because the faithfulness metric requires two-step LLM calls with complex nested JSON schemas. The runner handles Groq free-tier 429 responses automatically with retry-after backoff.

---

## CI/CD

| Workflow | Trigger | Steps |
|---|---|---|
| `ci.yml` | Every push | ruff lint → mypy type-check → pytest unit tests → Codecov |
| `eval.yml` | PR to main | ChromaDB service → ingest PDF fixture → RAGAS eval → pytest gate → PR comment |

Set `GROQ_API_KEY` as a GitHub Actions secret to enable the eval workflow.

---

## Docker (Full Stack)

```bash
docker compose up --build
```

Starts ChromaDB, the FastAPI backend, and the Next.js frontend together. For GPU-accelerated Ollama, see the commented section in `docker-compose.yml`.

---

## Make Targets

```bash
make dev      # Run backend + frontend in parallel (hot reload)
make test     # Run unit tests with coverage
make eval     # Run RAGAS evaluation
```
