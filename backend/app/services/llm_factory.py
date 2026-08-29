"""Gemini model registry and factory.

Single provider (Google Gemini) with three selectable models. The frontend
model picker is driven by AVAILABLE_MODELS via GET /models/.
"""
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings


@dataclass
class ModelInfo:
    """Metadata about an available LLM model."""
    id: str
    name: str
    context_tokens: int | None = None
    description: str = ""


# Available models registry
AVAILABLE_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        context_tokens=1_048_576,
        description="Fast and capable default model",
    ),
    ModelInfo(
        id="gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        context_tokens=1_048_576,
        description="Higher quality for complex questions",
    ),
    ModelInfo(
        id="gemini-2.5-flash-lite",
        name="Gemini 2.5 Flash-Lite",
        context_tokens=1_048_576,
        description="Fastest and cheapest",
    ),
]


def get_model_by_id(model_id: str) -> ModelInfo | None:
    """Find a model by its ID."""
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model
    return None


def create_llm(model_id: str, temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Create a Gemini chat model instance.

    Args:
        model_id: Model identifier (must be in AVAILABLE_MODELS)
        temperature: Sampling temperature (0.0 to 1.0)

    Raises:
        ValueError: If model_id is unknown or GEMINI_API_KEY is not configured
    """
    if get_model_by_id(model_id) is None:
        raise ValueError(
            f"Unknown model: {model_id}. Available: {[m.id for m in AVAILABLE_MODELS]}"
        )
    if not settings.gemini_api_key:
        raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY in .env")
    return ChatGoogleGenerativeAI(
        model=model_id,
        temperature=temperature,
        google_api_key=settings.gemini_api_key,
    )


def get_default_model() -> str:
    """Get the default model id from settings."""
    return settings.default_llm_model
