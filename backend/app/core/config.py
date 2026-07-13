"""Application settings loaded from the .env file.

This is the FastAPI equivalent of Spring Boot's application.properties —
but type-safe and validated at startup via pydantic-settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # These fields map 1:1 to keys in backend/.env
    database_url: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    frontend_origin: str = "http://localhost:5173"

    # Ollama LLM settings
    ollama_base_url: str = "http://ollama:11434"

    # OpenAI settings
    openai_api_key: str = ""

    # Default LLM provider and model
    default_llm_provider: str = "ollama"  # Options: "ollama", "openai"
    default_llm_model: str = "llama3.2"

    # Tell pydantic-settings to read from backend/.env
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Single shared instance — import this anywhere: `from app.core.config import settings`
settings = Settings()
