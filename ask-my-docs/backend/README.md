---
title: Ask My Docs Backend
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Ask My Docs — Backend API

FastAPI RAG backend for the Ask My Docs portfolio project.

**Stack:** FastAPI · fastembed (BAAI/bge-small-en-v1.5) · FlashRank reranker · Qdrant Cloud · Groq (llama-3.1-8b-instant) · BM25 hybrid retrieval

**Note:** The BM25 index is stored in-container and resets on Space restart. Re-upload your PDF after a restart to rebuild the index.
