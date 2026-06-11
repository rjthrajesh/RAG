from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Walk up from this file to find the nearest .env (covers both
# "run from backend/" and "run from ask-my-docs/" working directories).
_HERE = Path(__file__).resolve().parent
_ENV_CANDIDATES = [_HERE.parent / ".env", _HERE.parent.parent / ".env"]
_ENV_FILE = next((str(p) for p in _ENV_CANDIDATES if p.exists()), ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8")
    # LLM provider: "ollama" | "groq"
    llm_provider: str = "ollama"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    # Judge model for RAGAS evaluation (needs to handle complex nested JSON schemas)
    groq_judge_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # Embedding
    embed_model_name: str = "BAAI/bge-base-en-v1.5"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k_retrieve: int = 20
    top_k_rerank: int = 5

    # Storage
    bm25_index_path: str = "./data/bm25_index.json"

    # CORS — comma-separated list of allowed origins
    cors_origins: str = "http://localhost:3000,https://rag-xi-ashy.vercel.app"

    # Evaluation thresholds
    eval_faithfulness_threshold: float = 0.75
    eval_answer_relevancy_threshold: float = 0.70

settings = Settings()
