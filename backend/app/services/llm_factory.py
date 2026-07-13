"""LLM factory and model registry.

Supports multiple LLM providers:
- Ollama: Local models (llama3.2, mistral, codellama, etc.)
- OpenAI: GPT-4, GPT-3.5-turbo, etc.

The factory pattern allows dynamic LLM instantiation based on provider and model.
"""
from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from app.core.config import settings


@dataclass
class ModelInfo:
    """Metadata about an available LLM model."""
    id: str
    name: str
    provider: str
    context_tokens: int | None = None
    description: str = ""


# Available models registry
AVAILABLE_MODELS: list[ModelInfo] = [
    # Ollama (local) models
    ModelInfo(
        id="llama3.2",
        name="Llama 3.2",
        provider="ollama",
        context_tokens=128000,
        description="Fast and capable local model"
    ),
    ModelInfo(
        id="mistral",
        name="Mistral 7B",
        provider="ollama",
        context_tokens=32000,
        description="Balanced performance and speed"
    ),
    ModelInfo(
        id="codellama",
        name="Code Llama",
        provider="ollama",
        context_tokens=100000,
        description="Specialized for code generation"
    ),
    ModelInfo(
        id="qwen2.5",
        name="Qwen 2.5",
        provider="ollama",
        context_tokens=32768,
        description="Strong reasoning capabilities"
    ),
    # OpenAI models
    ModelInfo(
        id="gpt-4o",
        name="GPT-4o",
        provider="openai",
        context_tokens=128000,
        description="OpenAI's fastest flagship model"
    ),
    ModelInfo(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        context_tokens=128000,
        description="Affordable and fast"
    ),
    ModelInfo(
        id="gpt-4-turbo",
        name="GPT-4 Turbo",
        provider="openai",
        context_tokens=128000,
        description="Advanced reasoning"
    ),
]


def get_models_for_provider(provider: str) -> list[ModelInfo]:
    """Get all available models for a specific provider."""
    return [m for m in AVAILABLE_MODELS if m.provider == provider]


def get_model_by_id(model_id: str) -> ModelInfo | None:
    """Find a model by its ID."""
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model
    return None


def create_llm(provider: str, model_id: str, temperature: float = 0.3) -> Any:
    """Factory function to create an LLM instance.

    Args:
        provider: The LLM provider ("ollama" or "openai")
        model_id: The model identifier
        temperature: Sampling temperature (0.0 to 1.0)

    Returns:
        A LangChain Chat model instance

    Raises:
        ValueError: If provider is unknown or credentials are missing
    """
    if provider == "ollama":
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=model_id,
            temperature=temperature,
        )
    elif provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY in .env")
        return ChatOpenAI(
            model=model_id,
            temperature=temperature,
            api_key=settings.openai_api_key,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_default_model() -> tuple[str, str]:
    """Get the default provider and model from settings."""
    return settings.default_llm_provider, settings.default_llm_model
