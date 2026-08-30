"""Env-gated LangSmith tracing via OpenLLMetry (OpenTelemetry OTLP).

LangSmith's first-class integrations are LangChain-shaped; for LlamaIndex
the supported path is OpenTelemetry: OpenLLMetry (traceloop-sdk) auto-
instruments LlamaIndex + the Gemini SDK and exports OTLP spans to LangSmith's
OTel endpoint, where they appear as retrieval spans (retrieved nodes, scores)
and generation spans (prompt, tokens, latency) in LANGSMITH_PROJECT.

Disabled by default (LANGSMITH_TRACING=false): zero overhead, no imports.
"""
import logging

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

    # Explicit params: traceloop-sdk ignores the generic OTEL_EXPORTER_OTLP_*
    # env vars; api_endpoint/headers are its supported configuration surface.
    # Dict headers also sidestep comma/space escaping in header strings.
    from traceloop.sdk import Traceloop  # heavy; only when tracing

    try:
        Traceloop.init(
            app_name=settings.langsmith_project,
            api_endpoint=_LANGSMITH_OTLP_ENDPOINT,
            headers={
                "x-api-key": settings.langsmith_api_key,
                "Langsmith-Project": settings.langsmith_project,
            },
        )
    except Exception:
        # Observability must degrade, never take the app down.
        logger.exception(
            "LangSmith tracing failed to initialize — continuing without it"
        )
        return

    logger.info(
        "LangSmith tracing enabled (project=%s, endpoint=%s)",
        settings.langsmith_project,
        _LANGSMITH_OTLP_ENDPOINT,
    )
