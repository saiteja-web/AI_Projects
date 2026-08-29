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

    # Tell pydantic-settings to read from backend/.env
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Single shared instance — import this anywhere: `from app.core.config import settings`
settings = Settings()
