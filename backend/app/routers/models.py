"""Endpoint for listing available LLM models.

Provides the frontend with the flat Gemini model list.
"""
from fastapi import APIRouter

from app.services.llm_factory import AVAILABLE_MODELS, get_default_model

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/")
async def list_models():
    """List all available LLM models.

    Returns:
        {
            "models": [{"id", "name", "context_tokens", "description"}, ...],
            "default_model": "gemini-3.6-flash"
        }
    """
    return {
        "models": [
            {
                "id": model.id,
                "name": model.name,
                "context_tokens": model.context_tokens,
                "description": model.description,
            }
            for model in AVAILABLE_MODELS
        ],
        "default_model": get_default_model(),
    }
