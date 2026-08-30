"""Application settings loaded from the .env file.

This is the FastAPI equivalent of Spring Boot's application.properties —
but type-safe and validated at startup via pydantic-settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # These fields map 1:1 to keys in backend/.env
    database_url: str
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    frontend_origin: str = "http://localhost:5173"

    # Gemini API (https://aistudio.google.com/apikey)
    gemini_api_key: str = ""

    # Default LLM model
    default_llm_model: str = "gemini-2.5-flash"

    # ── LlamaIndex RAG knobs ─────────────────────────────────
    # Vector dimension of the embedding model (bge-small-en-v1.5 = 384)
    embed_dim: int = 384
    # Chunking: a chunk = N whole sentences within a section
    sentences_per_chunk: int = 6
    sentence_overlap: int = 1
    # pgvector HNSW index tuning
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 40

    # ── LangSmith tracing (optional; OTLP via OpenLLMetry) ──
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-doc-chat"
    langsmith_tracing: bool = False

    # Tell pydantic-settings to read from backend/.env
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Single shared instance — import this anywhere: `from app.core.config import settings`
settings = Settings()
