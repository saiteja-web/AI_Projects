"""Gemini model registry and factory.

Single provider (Google Gemini) with three selectable models. The frontend
model picker is driven by AVAILABLE_MODELS via GET /models/.
"""
from dataclasses import dataclass

from llama_index.llms.google_genai import GoogleGenAI

from app.core.config import settings

# Output token limit shared by the current Gemini models (what the API's
# models.get metadata would report). Keeps construction free of network calls.
GEMINI_MAX_OUTPUT_TOKENS = 65_536


@dataclass
class ModelInfo:
    """Metadata about an available LLM model."""
    id: str
    name: str
    context_tokens: int | None = None
    description: str = ""


# Available models registry.
# gemini-2.5-* ids are rejected at invocation time for new API keys ("no
# longer available to new users"), so the registry tracks the current family;
# the -latest aliases are Google's stable pointers to the current pro/lite.
AVAILABLE_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gemini-3.6-flash",
        name="Gemini 3.6 Flash",
        context_tokens=1_048_576,
        description="Fast and capable default model",
    ),
    ModelInfo(
        id="gemini-pro-latest",
        name="Gemini Pro (latest)",
        context_tokens=1_048_576,
        description="Higher quality for complex questions",
    ),
    ModelInfo(
        id="gemini-flash-lite-latest",
        name="Gemini Flash-Lite (latest)",
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


def create_llm(model_id: str, temperature: float = 0.3) -> GoogleGenAI:
    """Create a Gemini chat model instance.

    Args:
        model_id: Model identifier (must be in AVAILABLE_MODELS)
        temperature: Sampling temperature (0.0 to 1.0)

    Raises:
        ValueError: If model_id is unknown or GEMINI_API_KEY is not configured
    """
    info = get_model_by_id(model_id)
    if info is None:
        raise ValueError(
            f"Unknown model: {model_id}. Available: {[m.id for m in AVAILABLE_MODELS]}"
        )
    if not settings.gemini_api_key:
        raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY in .env")
    # Passing both max_tokens and context_window keeps the constructor offline:
    # otherwise GoogleGenAI eagerly calls client.models.get() over the network,
    # which breaks unit tests and adds a startup API dependency. These match the
    # gemini-2.5 family limits the API would return from that metadata call.
    return GoogleGenAI(
        model=model_id,
        temperature=temperature,
        api_key=settings.gemini_api_key,
        context_window=info.context_tokens,
        max_tokens=GEMINI_MAX_OUTPUT_TOKENS,
    )


def get_default_model() -> str:
    """Get the default model id from settings."""
    return settings.default_llm_model
