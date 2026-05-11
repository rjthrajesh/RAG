from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # Embedding
    embed_model_name: str = "BAAI/bge-large-en-v1.5"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k_retrieve: int = 20
    top_k_rerank: int = 5

    # Storage
    bm25_index_path: str = "./data/bm25_index.json"

    # Evaluation thresholds
    eval_faithfulness_threshold: float = 0.75
    eval_answer_relevancy_threshold: float = 0.70

settings = Settings()
