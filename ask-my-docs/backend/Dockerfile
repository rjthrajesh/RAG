FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ONNX models at build time so the first request isn't slow
# and so OOM-killed containers restart without re-downloading
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-base-en-v1.5').embed(['warmup']))"
RUN python -c "from flashrank import Ranker; Ranker('ms-marco-MiniLM-L-12-v2', cache_dir='/root/.cache/flashrank')"

COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
