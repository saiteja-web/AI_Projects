"""Regression tests for the LlamaIndex migration config knobs."""
from app.core.config import Settings


def _fresh_settings() -> Settings:
    """Instantiate Settings directly so env-file values don't mask defaults."""
    return Settings(database_url="postgresql+asyncpg://u:p@localhost:5432/db")


def test_rag_knobs_have_defaults():
    s = _fresh_settings()
    assert s.embed_dim == 384
    assert s.sentences_per_chunk == 6
    assert s.sentence_overlap == 1
    assert s.hnsw_m == 16
    assert s.hnsw_ef_construction == 64
    assert s.hnsw_ef_search == 40


def test_langsmith_tracing_off_by_default():
    s = _fresh_settings()
    assert s.langsmith_tracing is False
    assert s.langsmith_project == "rag-doc-chat"
    assert s.langsmith_api_key == ""
