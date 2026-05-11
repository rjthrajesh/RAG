# ChromaDB → Pinecone Migration Guide

This system uses ChromaDB for local/self-hosted deployments. To migrate to Pinecone:

## Step 1 — Replace `vector_store.py`

Swap the ChromaDB client in `backend/app/retrieval/vector_store.py` for the Pinecone SDK.
The `VectorStore` interface (`add_chunks`, `query`, `delete_all`) stays identical.

```python
# pip install pinecone-client
import pinecone
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="us-east1-gcp")
```

## Step 2 — Update `config.py`

Remove `CHROMA_HOST` / `CHROMA_PORT`. Add:
```
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east1-gcp
PINECONE_INDEX_NAME=ask-my-docs
```

## Step 3 — Remove ChromaDB from Docker Compose

Delete the `chromadb` service from `docker-compose.yml`. Remove the `depends_on` reference
in the `backend` service.

That's it — BM25, reranking, generation, and the frontend are all unaffected.
