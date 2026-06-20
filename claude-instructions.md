# claude-instructions.md
# Ask My Docs — Domain-Specific RAG System
### Built on: "AI Engineering: Building Applications with Foundation Models" (Chip Huyen, O'Reilly 2024)

---

## 0. How to Use This File

This document is the single source of truth for building this project with Claude Code.
Each section is a self-contained phase. Work through them in order. At the start of each
phase, paste the relevant section into Claude Code and say: "Implement this phase."

**Key decisions are explained with a `> WHY:` callout so you understand the tradeoffs.**

---

## 1. Project Overview

Build a production-grade "Ask My Docs" system that lets a user upload a technical PDF
(the AI Engineering book) and ask natural language questions against it. The system returns
grounded, cited answers — every claim is traceable to a specific page and passage.

### Core Features
- **Hybrid retrieval**: BM25 (keyword) + dense vector search combined via Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking**: A second-pass model re-scores retrieved chunks for precision
- **Citation enforcement**: Answers are structurally required to include source references
- **Chat UI**: Streaming chat interface with source panel showing cited passages
- **CI-gated evaluation**: RAGAS metrics run in GitHub Actions on every PR; failing scores block merges
- **Cloud deployable**: Full Docker Compose stack, deployable to any VPS or AWS EC2

### Non-Goals (out of scope)
- Multi-user auth (single-user system)
- Support for multiple simultaneous PDFs
- Fine-tuning any models

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        USER BROWSER                         │
│                    Next.js Chat UI (port 3000)               │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP / SSE streaming
┌────────────────────────▼────────────────────────────────────┐
│                   FastAPI Backend (port 8000)                │
│                                                             │
│  /ingest  →  PDF Parser → Chunker → Embedder → Stores       │
│  /query   →  [BM25 Retriever + Vector Retriever]            │
│               → RRF Fusion → Cross-Encoder Reranker         │
│               → Prompt Builder → Ollama LLM                 │
│               → Citation Validator → Stream Response        │
└──────┬──────────────────────────┬───────────────────────────┘
       │                          │
┌──────▼──────┐          ┌────────▼────────┐
│  ChromaDB   │          │   BM25 Index    │
│ (vector DB) │          │ (rank_bm25,     │
│  port 8001  │          │  persisted JSON)│
└─────────────┘          └─────────────────┘
       │
┌──────▼──────┐
│   Ollama    │
│  llama3.1   │
│  port 11434 │
└─────────────┘
```

> **WHY ChromaDB over Pinecone here?** ChromaDB runs as a Docker container with zero cloud
> setup, persists to disk, and is Python-native. For a single-user system on one book, it's
> the right call. A `MIGRATION.md` in the repo will document the 3-step swap to Pinecone
> when you need cloud scale.

> **WHY Ollama (llama3.1) over Claude/GPT-4?** You chose local — this means zero API costs,
> full data privacy, and no rate limits. llama3.1:8b runs well on 16GB RAM; use llama3.1:70b
> on a GPU instance for better quality. The abstraction layer we build makes swapping trivial.

---

## 3. Repository Structure

```
ask-my-docs/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI app + routes
│   │   ├── config.py                # All settings via env vars
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── pdf_parser.py        # PyMuPDF extraction with metadata
│   │   │   ├── chunker.py           # Recursive + semantic chunking
│   │   │   └── embedder.py          # sentence-transformers wrapper
│   │   ├── retrieval/
│   │   │   ├── __init__.py
│   │   │   ├── vector_store.py      # ChromaDB client wrapper
│   │   │   ├── bm25_store.py        # BM25 index (rank_bm25)
│   │   │   ├── hybrid_retriever.py  # RRF fusion logic
│   │   │   └── reranker.py          # Cross-encoder reranking
│   │   ├── generation/
│   │   │   ├── __init__.py
│   │   │   ├── llm_client.py        # Ollama client (swappable)
│   │   │   ├── prompt_builder.py    # Citation-enforcing prompt templates
│   │   │   └── citation_validator.py # Post-generation citation checker
│   │   └── evaluation/
│   │       ├── __init__.py
│   │       ├── eval_dataset.py      # Golden Q&A dataset builder
│   │       └── ragas_runner.py      # RAGAS metric runner
│   ├── tests/
│   │   ├── unit/
│   │   │   ├── test_chunker.py
│   │   │   ├── test_hybrid_retriever.py
│   │   │   └── test_citation_validator.py
│   │   └── eval/
│   │       ├── golden_dataset.json  # 25 hand-crafted Q&A pairs
│   │       └── test_ragas.py        # CI eval gate
│   ├── Dockerfile
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx             # Chat page
│   │   │   └── layout.tsx
│   │   ├── components/
│   │   │   ├── ChatWindow.tsx       # Message thread
│   │   │   ├── MessageBubble.tsx    # User + assistant messages
│   │   │   ├── CitationPanel.tsx    # Slide-out source viewer
│   │   │   ├── SourceChip.tsx       # Inline citation badge [p.42]
│   │   │   └── IngestionUploader.tsx # PDF drag-and-drop
│   │   ├── hooks/
│   │   │   └── useChat.ts           # SSE streaming hook
│   │   └── lib/
│   │       └── api.ts               # API client
│   ├── Dockerfile
│   ├── package.json
│   └── next.config.ts
├── docker-compose.yml               # Full local stack
├── docker-compose.prod.yml          # Production overrides
├── .github/
│   └── workflows/
│       ├── ci.yml                   # Lint + unit tests
│       └── eval.yml                 # RAGAS eval gate on PR
├── .env.example
├── MIGRATION.md                     # ChromaDB → Pinecone guide
└── README.md
```

---

## 4. Phase 1 — Project Scaffold & Configuration

### 4.1 Instructions for Claude Code

```
Create the full repository structure listed in Section 3. Then implement the following:

1. backend/app/config.py
   - Use pydantic-settings BaseSettings
   - Settings: OLLAMA_BASE_URL, OLLAMA_MODEL, CHROMA_HOST, CHROMA_PORT,
     EMBED_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVE,
     TOP_K_RERANK, BM25_INDEX_PATH, EVAL_FAITHFULNESS_THRESHOLD,
     EVAL_ANSWER_RELEVANCY_THRESHOLD
   - All values must be overridable via environment variables
   - Provide defaults suitable for local development

2. .env.example with all variables documented with comments

3. backend/requirements.txt with pinned versions:
   fastapi==0.111.0, uvicorn[standard]==0.29.0, pydantic-settings==2.2.1,
   pymupdf==1.24.3, sentence-transformers==3.0.1, chromadb==0.5.3,
   rank-bm25==0.2.2, transformers==4.41.2, torch==2.3.1 (cpu),
   ragas==0.1.9, langchain-core==0.2.5, httpx==0.27.0, python-multipart==0.0.9

4. frontend/package.json with:
   next@14, react@18, typescript, tailwindcss, lucide-react, eventsource-parser

5. A root Makefile with targets:
   make install, make dev, make test, make eval, make docker-up, make docker-down
```

---

## 5. Phase 2 — PDF Ingestion Pipeline

### 5.1 PDF Parser (`backend/app/ingestion/pdf_parser.py`)

```
Implement a PDF parser with the following exact behaviour:

Class: PDFParser
Method: parse(pdf_path: str) -> list[PageDocument]

PageDocument is a dataclass with fields:
  - page_number: int (1-indexed)
  - text: str (cleaned text)
  - char_count: int
  - has_tables: bool
  - has_figures: bool
  - section_title: str | None  (detected from font size heuristic)

Rules:
- Use PyMuPDF (fitz) for extraction
- Strip headers/footers: skip lines matching page number patterns or
  the string "AI Engineering" (the book's running header)
- Detect section titles: blocks where font size >= 14pt → extract as section_title
- Detect tables: if page contains >=3 horizontal lines within 100px vertical span
- Detect figures: if page contains image blocks (fitz block type == 1)
- Normalize whitespace: collapse multiple spaces/newlines to single space
- Return pages sorted by page_number
- Log a warning for any page yielding < 50 characters after cleaning
```

### 5.2 Chunker (`backend/app/ingestion/chunker.py`)

> **WHY this chunking strategy?** Naive fixed-size chunking breaks sentences mid-thought
> and loses context. We use recursive character splitting with overlap to preserve
> sentence boundaries, then add a sliding-window parent chunk stored as metadata —
> this gives the reranker access to broader context without bloating vector search.

```
Implement a chunker with the following exact behaviour:

Class: RecursiveChunker
Constructor params: chunk_size: int = 512, chunk_overlap: int = 64

Method: chunk(pages: list[PageDocument]) -> list[Chunk]

Chunk is a dataclass with fields:
  - chunk_id: str  (format: "page_{page_num}_chunk_{idx}")
  - text: str
  - page_number: int
  - section_title: str | None
  - char_count: int
  - parent_text: str  (the surrounding 1024-char window for reranker context)

Rules:
- Split on separators in order: ["\n\n", "\n", ". ", " "]
- Respect chunk_size and chunk_overlap
- Minimum chunk size: 100 characters (discard smaller)
- For each chunk, build parent_text by taking 256 chars before and after
  the chunk within the same page (capped at page boundaries)
- Preserve page_number and section_title from the source PageDocument
- After chunking, log total chunks, average chunk size, min/max chunk size
```

### 5.3 Embedder (`backend/app/ingestion/embedder.py`)

```
Implement an embedder:

Class: Embedder
Constructor: model_name: str = "BAAI/bge-large-en-v1.5"

> WHY bge-large-en-v1.5? It consistently ranks top-3 on the MTEB benchmark for
> retrieval tasks, outperforms OpenAI ada-002 on domain-specific text, and runs
> locally for free. The "bge" prefix stands for BAAI General Embedding.

Method: embed_chunks(chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]
Method: embed_query(query: str) -> list[float]

Rules:
- Use sentence_transformers.SentenceTransformer
- For bge models, prepend "Represent this sentence for searching relevant passages: "
  to the query (NOT to documents) — this is the BGE instruction prefix
- Batch encode in groups of 32 to manage memory
- Normalize embeddings (normalize_embeddings=True)
- Log embedding time and throughput (chunks/sec)
```

### 5.4 Ingestion Route (`backend/app/main.py` — ingestion endpoint)

```
Add a POST /ingest endpoint to the FastAPI app:

- Accepts: multipart/form-data with field "file" (PDF)
- Validates: file must be .pdf, max 50MB
- Pipeline: PDFParser → RecursiveChunker → Embedder → store to ChromaDB + BM25
- Returns JSON: { "status": "ok", "pages": int, "chunks": int, "duration_seconds": float }
- If a previous index exists, DELETE all existing ChromaDB documents and rebuild
- Save BM25 index to disk at BM25_INDEX_PATH as JSON (serialize the corpus)
- Use BackgroundTasks for the heavy work; return 202 Accepted immediately with a job_id
- Add GET /ingest/status/{job_id} that returns { "status": "processing|done|failed", "progress": float }
```

---

## 6. Phase 3 — Hybrid Retrieval + Reranking

### 6.1 Vector Store (`backend/app/retrieval/vector_store.py`)

```
Implement a ChromaDB wrapper:

Class: VectorStore
Constructor: connects to ChromaDB at CHROMA_HOST:CHROMA_PORT
Collection name: "ask_my_docs"

Methods:
  - add_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None
    Stores: document=chunk.text, embedding=embedding,
    metadata={ page_number, section_title, chunk_id, parent_text }

  - query(query_embedding: list[float], top_k: int) -> list[RetrievalResult]
    Returns results with: chunk_id, text, page_number, section_title,
    parent_text, score (cosine distance converted to similarity: 1 - distance)

  - delete_all() -> None

RetrievalResult is a dataclass with all fields above.
```

### 6.2 BM25 Store (`backend/app/retrieval/bm25_store.py`)

```
Implement a BM25 retriever:

Class: BM25Store

Methods:
  - build(chunks: list[Chunk]) -> None
    Tokenize each chunk with simple whitespace + lowercase + punctuation strip.
    Build BM25Okapi index. Persist to disk as JSON:
    { "corpus_tokens": [...], "chunk_ids": [...], "chunk_metadata": [...] }

  - load(path: str) -> None (loads persisted index)

  - query(query_text: str, top_k: int) -> list[RetrievalResult]
    Tokenize query same way. Get BM25 scores. Return top_k with normalized
    scores (divide by max score). Populate RetrievalResult from stored metadata.
```

### 6.3 Hybrid Retriever with RRF (`backend/app/retrieval/hybrid_retriever.py`)

> **WHY Reciprocal Rank Fusion?** BM25 is great at exact keyword matches ("what is
> RLHF"). Dense vectors are great at semantic matches ("how do you align LLMs with
> human preferences"). RRF combines ranked lists without needing score calibration —
> it only uses rank positions, making it robust and parameter-free.

```
Implement RRF fusion:

Class: HybridRetriever
Constructor: vector_store: VectorStore, bm25_store: BM25Store, embedder: Embedder

Method: retrieve(query: str, top_k: int = 20) -> list[RetrievalResult]

RRF formula: score(d) = Σ 1 / (k + rank(d))  where k=60 (standard constant)

Steps:
1. Run vector_store.query(embed_query(query), top_k=top_k) → dense_results
2. Run bm25_store.query(query, top_k=top_k) → sparse_results
3. Assign ranks (1-indexed) to each result list
4. For each unique chunk_id, compute RRF score summing across both lists
   (if a chunk appears in only one list, it still gets a score from that list)
5. Sort by RRF score descending
6. Return top_k results with an added field: retrieval_source ("dense"|"sparse"|"both")
7. Log: how many results came from dense-only, sparse-only, and both
```

### 6.4 Cross-Encoder Reranker (`backend/app/retrieval/reranker.py`)

> **WHY a cross-encoder reranker?** Bi-encoders (like BGE) encode query and document
> separately — fast but imprecise. A cross-encoder sees the query AND document together,
> giving much higher precision. We use it only on the top-20 retrieved chunks (not the
> full corpus) to keep latency reasonable. This two-stage pattern is standard in
> production RAG systems.

```
Implement reranking:

Class: CrossEncoderReranker
Constructor: model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
Load model once at startup using sentence_transformers.CrossEncoder

Method: rerank(query: str, results: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]

Steps:
1. Build pairs: [(query, result.parent_text) for result in results]
   Use parent_text (wider context) not chunk text for better signal
2. Run cross_encoder.predict(pairs) → scores
3. Attach score to each result as rerank_score
4. Sort by rerank_score descending
5. Return top_k results
6. Log: reranking latency, score distribution (min/max/mean)
```

---

## 7. Phase 4 — Generation with Citation Enforcement

### 7.1 LLM Client (`backend/app/generation/llm_client.py`)

```
Implement an Ollama LLM client:

Class: OllamaClient
Constructor: base_url: str, model: str

Method: stream_completion(prompt: str) -> AsyncIterator[str]
  Uses httpx.AsyncClient to POST to {base_url}/api/generate
  Request body: { "model": model, "prompt": prompt, "stream": true }
  Parse NDJSON response lines, yield text deltas
  Handle connection errors with a clear message: "Ollama not reachable at {url}"

Method: completion(prompt: str) -> str  (non-streaming, for eval pipeline)

> WHY make the LLM client a thin wrapper with a clean interface? So that swapping
> Ollama for Claude API or OpenAI requires changing only this file. The interface
> contract: takes a prompt string, returns string or async stream of strings.
```

### 7.2 Prompt Builder (`backend/app/generation/prompt_builder.py`)

```
Implement citation-enforcing prompt construction:

Class: PromptBuilder

Method: build_rag_prompt(query: str, results: list[RetrievalResult]) -> str

Build a prompt with this EXACT structure:

---
You are an expert assistant for the book "AI Engineering: Building Applications
with Foundation Models" by Chip Huyen.

Answer the user's question using ONLY the provided context passages below.
You MUST cite every factual claim using the format [p.{page_number}].
If the answer cannot be found in the context, say exactly:
"I cannot find information about this in the provided context."
Do NOT use any knowledge outside the provided passages.

CONTEXT PASSAGES:
[1] (Page {page_number}, {section_title or "General"}):
{chunk_text}

[2] (Page {page_number}, ...):
{chunk_text}

... (include all top_k results)

QUESTION: {query}

ANSWER (cite every claim with [p.X]):
---

Rules:
- Number context passages starting from [1]
- Include all reranked results (default 5)
- section_title shown as "General" if None
- Truncate any single chunk to 600 chars in the prompt to control token count
```

### 7.3 Citation Validator (`backend/app/generation/citation_validator.py`)

```
Implement post-generation citation validation:

Class: CitationValidator

Method: validate(answer: str, results: list[RetrievalResult]) -> ValidationResult

ValidationResult dataclass:
  - is_valid: bool
  - citations_found: list[int]  (page numbers referenced)
  - uncited_sentences: list[str]  (sentences with no citation)
  - warning: str | None

Rules:
1. Extract all [p.X] patterns from the answer
2. Verify each cited page number exists in the results list
3. Flag any sentence (split on ". ") longer than 15 words with no [p.X] citation
4. If answer contains "I cannot find information" → is_valid=True, skip checks
5. If >50% of substantive sentences are uncited → is_valid=False
6. Log warnings for invalid citations but do NOT suppress the answer —
   attach the validation result to the response metadata instead

Method: strip_hallucinated_citations(answer: str, valid_pages: set[int]) -> str
  Remove any [p.X] where X is not in valid_pages
```

### 7.4 Query Route

```
Add POST /query to FastAPI:

Request body:
{
  "question": str,
  "session_id": str | None,   # for future conversation history
  "top_k_retrieve": int = 20,
  "top_k_rerank": int = 5
}

Pipeline:
1. HybridRetriever.retrieve(question, top_k=top_k_retrieve)
2. CrossEncoderReranker.rerank(question, results, top_k=top_k_rerank)
3. PromptBuilder.build_rag_prompt(question, reranked_results)
4. OllamaClient.stream_completion(prompt)
5. CitationValidator.validate(full_answer, reranked_results) [post-stream]

Response: Server-Sent Events (SSE) stream
  - Stream text delta events: data: {"type": "delta", "text": "..."}
  - After full answer: data: {"type": "sources", "sources": [{page_number,
    section_title, text_preview, rerank_score, retrieval_source}]}
  - Final event: data: {"type": "done", "citation_valid": bool, "warning": str|null}

Add GET /health that returns { "status": "ok", "ollama": bool, "chroma": bool }
```

---

## 8. Phase 5 — Frontend Chat UI

```
Build the Next.js frontend. Use Tailwind CSS. Target a clean, minimal two-panel layout:
  LEFT (60%): Chat thread
  RIGHT (40%): Sources panel (updates after each answer)

Implement these components:

1. useChat.ts hook:
   - Manages messages array: { role: "user"|"assistant", content: string,
     sources: Source[], citation_valid: bool }[]
   - sendMessage(question: string): POSTs to /query, reads SSE stream,
     appends delta text in real-time, sets sources on "sources" event
   - Exposes: messages, sendMessage, isLoading, error

2. ChatWindow.tsx:
   - Renders MessageBubble for each message
   - Auto-scrolls to bottom on new content
   - Shows a pulsing cursor while streaming
   - Input bar at bottom with send button (disabled while loading)

3. MessageBubble.tsx:
   - User messages: right-aligned, accent background
   - Assistant messages: left-aligned, white card
   - Inline SourceChip components for each [p.X] citation in the text
     (parse the answer text and render page references as clickable badges)
   - If citation_valid=false, show a subtle yellow banner: "Some claims may be uncited"

4. SourceChip.tsx:
   - Renders as "[p.42]" badge, clickable
   - On click: highlights the corresponding source card in the right panel

5. CitationPanel.tsx:
   - Shows source cards for the latest assistant message
   - Each card: page number badge, section title, text preview (first 200 chars)
   - Color-coded retrieval_source indicator: blue=vector, orange=BM25, green=both
   - Rerank score shown as a confidence bar (0–1 scale)

6. IngestionUploader.tsx:
   - Drag-and-drop zone for PDF upload
   - Shows progress bar during ingestion (polls GET /ingest/status/{job_id})
   - On completion: shows "Ready — {chunks} chunks indexed from {pages} pages"
   - Only shown when no index exists (check via GET /health)
```

---

## 9. Phase 6 — Evaluation Pipeline

> **WHY CI-gated evals?** RAG systems silently degrade — a config change, a new
> chunking strategy, or model update can tank retrieval quality without any errors.
> RAGAS metrics give you objective quality scores. Blocking merges on score drops
> means quality regressions never reach production.

### 9.1 Golden Dataset (`backend/tests/eval/golden_dataset.json`)

```
Create a golden dataset with 25 question-answer pairs about the AI Engineering book.
Structure each entry as:
{
  "question": "...",
  "ground_truth": "...",   # factual answer from the book
  "relevant_pages": [int]  # which pages should be retrieved
}

Include questions spanning:
- Definitions (5): e.g. "What is the difference between foundation models and traditional ML models?"
- How-to (5): e.g. "How does the author recommend structuring a RAG pipeline?"
- Comparative (5): e.g. "How does RLHF differ from RLAIF according to the book?"
- Specific facts (5): e.g. "What does the author say about context window limitations?"
- Failure modes (5): e.g. "What are the main causes of hallucination described in the book?"

Note: You will need to populate ground_truth values after ingesting the PDF.
Provide placeholder entries with TODO markers for ground_truth.
```

### 9.2 RAGAS Runner (`backend/app/evaluation/ragas_runner.py`)

```
Implement the evaluation runner:

Class: RAGASRunner

Method: run_eval(dataset_path: str) -> EvalReport

EvalReport dataclass:
  - faithfulness: float        # are claims supported by context?
  - answer_relevancy: float    # is the answer relevant to the question?
  - context_precision: float   # are retrieved chunks relevant?
  - context_recall: float      # are relevant chunks being retrieved?
  - passed: bool               # True if all metrics >= thresholds

Steps:
1. Load golden_dataset.json
2. For each entry: call the full pipeline (retrieve → rerank → generate)
   using OllamaClient.completion() (non-streaming)
3. Build ragas Dataset with columns: question, answer, contexts, ground_truth
4. Run ragas.evaluate() with metrics:
   [faithfulness, answer_relevancy, context_precision, context_recall]
5. Log per-question results and aggregate scores
6. Write report to eval_report.json

Thresholds (from config):
  EVAL_FAITHFULNESS_THRESHOLD = 0.75
  EVAL_ANSWER_RELEVANCY_THRESHOLD = 0.70
```

### 9.3 GitHub Actions Workflows

```
Create two workflow files:

.github/workflows/ci.yml:
  Trigger: push to any branch, PR to main
  Jobs:
    lint: ruff check backend/
    type-check: mypy backend/app
    unit-tests: pytest backend/tests/unit/ -v --cov=app --cov-report=xml
    Upload coverage to Codecov

.github/workflows/eval.yml:
  Trigger: PR to main ONLY
  Jobs:
    eval:
      runs-on: ubuntu-latest
      services:
        chromadb:
          image: chromadb/chroma:0.5.3
          ports: ["8001:8000"]
        ollama:
          image: ollama/ollama:latest
          ports: ["11434:11434"]
      Steps:
      1. Checkout + setup Python 3.11
      2. pip install -r backend/requirements.txt
      3. Pull ollama model: ollama pull llama3.1:8b
      4. Ingest the test PDF (use a 10-page excerpt, committed to repo as
         tests/fixtures/ai_engineering_excerpt.pdf)
      5. Run: python -m app.evaluation.ragas_runner
      6. Parse eval_report.json — fail the job if passed=false
      7. Post metric scores as a PR comment using actions/github-script
```

---

## 10. Phase 7 — Docker & Deployment

### 10.1 Docker Compose (Local Dev)

```
Create docker-compose.yml with these services:

services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes: ["./data:/data"]   # persists ChromaDB + BM25 index
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
      - BM25_INDEX_PATH=/data/bm25_index.json
    depends_on: [chromadb, ollama]
    healthcheck: test curl -f http://localhost:8000/health

  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on: [backend]

  chromadb:
    image: chromadb/chroma:0.5.3
    ports: ["8001:8000"]
    volumes: ["./data/chroma:/chroma/chroma"]

  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes: ["ollama_models:/root/.ollama"]
    # For GPU: add deploy.resources.reservations.devices (nvidia)

volumes:
  ollama_models:
```

### 10.2 Production Docker Compose

```
Create docker-compose.prod.yml (extends base):

Changes from dev:
- backend: remove volume mount, use env vars from .env file
- frontend: set NODE_ENV=production
- Add nginx service:
    image: nginx:alpine
    ports: ["80:80", "443:443"]
    volumes: ["./nginx.conf:/etc/nginx/nginx.conf:ro",
              "./certs:/etc/nginx/certs:ro"]
    depends_on: [frontend, backend]
- Add restart: unless-stopped to all services
- Remove exposed ports for chromadb and ollama (internal only)

Create nginx.conf:
- Route /api/* → http://backend:8000
- Route /* → http://frontend:3000
- Enable gzip compression
- Set proxy_read_timeout 120s (for LLM streaming)
```

### 10.3 AWS EC2 Deployment Checklist

```
Create DEPLOYMENT.md with step-by-step AWS deployment:

1. Instance recommendation:
   - Minimum: t3.xlarge (4 vCPU, 16GB RAM) for llama3.1:8b
   - Better: g4dn.xlarge (GPU) for llama3.1:70b
   - Storage: 50GB EBS (models are ~5GB each)

2. Setup steps:
   - Install Docker + Docker Compose on Amazon Linux 2023
   - Clone repo, copy .env.example to .env, fill values
   - Pull ollama model: docker compose exec ollama ollama pull llama3.1:8b
   - Run: docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

3. Security groups:
   - Allow inbound: 80, 443 (HTTP/HTTPS)
   - Block: 8000, 8001, 11434 (internal services only)

4. Domain + SSL:
   - Use Certbot + Let's Encrypt for SSL certs
   - Update nginx.conf for HTTPS redirect

5. Monitoring:
   - docker compose logs -f backend (watch for errors)
   - GET /health endpoint for uptime monitoring (set up UptimeRobot free)
```

---

## 11. Phase 8 — Unit Tests

```
Write the following unit tests. Use pytest. Mock external services (ChromaDB, Ollama).

backend/tests/unit/test_chunker.py:
  - test_chunk_respects_size_limit: no chunk exceeds chunk_size * 1.2
  - test_chunk_overlap_present: consecutive chunks share overlapping text
  - test_minimum_chunk_size_enforced: no chunk under 100 chars
  - test_chunk_id_format: matches "page_\d+_chunk_\d+"
  - test_parent_text_bounds: parent_text never exceeds page boundaries

backend/tests/unit/test_hybrid_retriever.py:
  - test_rrf_fusion_both_sources: chunk in both lists scores higher than chunk in one
  - test_rrf_k_constant: changing k=60 changes scores but not top result with strong overlap
  - test_retrieval_source_tagged: results tagged correctly as dense/sparse/both
  - test_empty_sparse_results: handles BM25 returning no results gracefully

backend/tests/unit/test_citation_validator.py:
  - test_valid_answer_passes: answer with [p.X] on every claim passes
  - test_uncited_answer_fails: answer with no citations fails validation
  - test_hallucinated_page_stripped: [p.999] removed when page 999 not in results
  - test_cannot_find_answer_always_valid: "I cannot find..." passes regardless
  - test_short_sentences_exempt: sentences under 15 words not flagged as uncited
```

---

## 12. Environment Variables Reference

```
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8001

# Embedding
EMBED_MODEL_NAME=BAAI/bge-large-en-v1.5

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# Retrieval
TOP_K_RETRIEVE=20
TOP_K_RERANK=5

# Storage
BM25_INDEX_PATH=./data/bm25_index.json

# Evaluation thresholds (0.0 - 1.0)
EVAL_FAITHFULNESS_THRESHOLD=0.75
EVAL_ANSWER_RELEVANCY_THRESHOLD=0.70
```

---

## 13. Key Technical Decisions Summary

| Decision | Choice | Why |
|---|---|---|
| Vector DB | ChromaDB | Zero infra, Python-native, Docker-ready |
| Sparse retrieval | BM25 (rank_bm25) | Best keyword recall, no infra needed |
| Fusion strategy | RRF (k=60) | Parameter-free, rank-based, robust |
| Embedding model | BGE-large-en-v1.5 | Top MTEB scores, free, local |
| Reranker | ms-marco-MiniLM-L-6-v2 | Fast cross-encoder, proven on passage ranking |
| LLM | Ollama (llama3.1:8b) | Local, free, no API keys, swappable |
| Eval framework | RAGAS | Industry standard for RAG quality metrics |
| CI gate | GitHub Actions | Free for public repos, blocks on quality regression |
| PDF parsing | PyMuPDF | Fastest Python PDF lib, good metadata extraction |
| Streaming | SSE (Server-Sent Events) | Simpler than WebSockets for unidirectional stream |

---

## 14. Recommended Build Order

```
Phase 1  →  Scaffold + config (30 min)
Phase 2  →  Ingestion pipeline (2 hours) — test with a single PDF chapter first
Phase 3  →  Hybrid retrieval (2 hours) — validate with manual queries before building UI
Phase 6  →  Golden dataset (1 hour) — do this early, drives quality awareness
Phase 4  →  Generation + citations (1.5 hours)
Phase 5  →  Frontend UI (2 hours)
Phase 8  →  Unit tests (1 hour)
Phase 7  →  Docker + deployment (1 hour)
Phase 6  →  CI eval pipeline (1 hour) — wire up after everything works
```

---

## 15. Validation Checkpoints

After each phase, verify with these manual checks before proceeding:

**After Phase 2 (Ingestion):**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@ai_engineering.pdf"
# Expected: 202 Accepted with job_id
# Poll /ingest/status/{id} until done
# Expected: ~800-1200 chunks for the full book
```

**After Phase 3 (Retrieval):**
```python
# Run in Python REPL
from app.retrieval.hybrid_retriever import HybridRetriever
r = HybridRetriever(...)
results = r.retrieve("What is RLHF?", top_k=10)
# Check: results from both dense and sparse sources
# Check: page numbers are plausible (not all the same page)
```

**After Phase 4 (Generation):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between RAG and fine-tuning?"}'
# Check: answer contains [p.X] citations
# Check: citation_valid=true in done event
```

**After Phase 7 (Docker):**
```bash
docker compose up -d
curl http://localhost:8000/health
# Expected: { "status": "ok", "ollama": true, "chroma": true }
```

---

*End of claude-instructions.md*
*Generated for: Ask My Docs — AI Engineering Book RAG System*
*Stack: FastAPI · Next.js · ChromaDB · BM25 · BGE · Ollama · RAGAS · Docker*
