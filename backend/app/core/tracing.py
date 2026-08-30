"""Env-gated LangSmith tracing via OpenLLMetry (OpenTelemetry OTLP).

LangSmith's first-class integrations are LangChain-shaped; for LlamaIndex
the supported path is OpenTelemetry: OpenLLMetry (traceloop-sdk) auto-
instruments LlamaIndex + the Gemini SDK and exports OTLP spans to LangSmith's
OTel endpoint, where they appear as retrieval spans (retrieved nodes, scores)
and generation spans (prompt, tokens, latency) in LANGSMITH_PROJECT.

Disabled by default (LANGSMITH_TRACING=false): zero overhead, no imports.
"""
import logging
import os

from app.core.config import settings

logger = logging.getLogger(__name__)

_LANGSMITH_OTLP_ENDPOINT = "https://api.smith.langchain.com/otel/v1/traces"


def setup_tracing() -> None:
    """Configure OTLP tracing to LangSmith. No-op unless LANGSMITH_TRACING=true."""
    if not settings.langsmith_tracing:
        return
    if not settings.langsmith_api_key:
        logger.warning(
            "LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty — tracing disabled"
        )
        return

    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", _LANGSMITH_OTLP_ENDPOINT)
    os.environ.setdefault(
        "OTEL_EXPORTER_OTLP_HEADERS",
        f"x-api-key={settings.langsmith_api_key},"
        f"Langsmith-Project={settings.langsmith_project}",
    )

    # Imported lazily: traceloop-sdk is heavy and only needed when tracing.
    from traceloop.sdk import Traceloop

    Traceloop.init(app_name=settings.langsmith_project)
    logger.info(
        "LangSmith tracing enabled (project=%s, endpoint=%s)",
        settings.langsmith_project,
        _LANGSMITH_OTLP_ENDPOINT,
    )
