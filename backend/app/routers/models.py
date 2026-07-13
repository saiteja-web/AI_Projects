"""Endpoint for listing available LLM models.

Provides the frontend with available models organized by provider.
"""
from fastapi import APIRouter

from app.services.llm_factory import (
    AVAILABLE_MODELS,
    get_default_model,
    get_models_for_provider,
)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/")
async def list_models():
    """List all available LLM models grouped by provider.

    Returns:
        {
            "models": [...],
            "default_provider": "...",
            "default_model": "..."
        }
    """
    # Group models by provider
    models_by_provider = {}
    for model in AVAILABLE_MODELS:
        if model.provider not in models_by_provider:
            models_by_provider[model.provider] = []
        models_by_provider[model.provider].append({
            "id": model.id,
            "name": model.name,
            "context_tokens": model.context_tokens,
            "description": model.description,
        })

    default_provider, default_model = get_default_model()

    return {
        "providers": {
            "ollama": {
                "name": "Ollama (Local)",
                "models": models_by_provider.get("ollama", []),
            },
            "openai": {
                "name": "OpenAI",
                "models": models_by_provider.get("openai", []),
            },
        },
        "default_provider": default_provider,
        "default_model": default_model,
    }
