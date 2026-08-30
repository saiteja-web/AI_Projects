"""Unit tests for rag_service guards (no DB, no model downloads, no API)."""
from types import SimpleNamespace

import pytest

from app.services import rag_service
from app.services.rag_service import SessionNotIndexedError


class _FakeRetriever:
    def __init__(self, nodes):
        self._nodes = nodes

    def retrieve(self, query):
        return self._nodes


class _FakeIndex:
    def __init__(self, retriever):
        self._retriever = retriever

    def as_retriever(self, **kwargs):
        return self._retriever


class _FakeNode:
    def __init__(self, page, section):
        self.metadata = {"page": page, "section": section}


def _patch_pipeline(monkeypatch, retrieved_nodes):
    """Stub everything between rag_service and the outside world."""
    monkeypatch.setattr(rag_service, "_configure_llamaindex", lambda: None)
    monkeypatch.setattr(rag_service, "_vector_store", lambda: object())
    monkeypatch.setattr(
        rag_service.VectorStoreIndex,
        "from_vector_store",
        lambda store, embed_model=None: _FakeIndex(_FakeRetriever(retrieved_nodes)),
    )
    monkeypatch.setattr(
        rag_service,
        "_synthesizer_for",
        lambda model_id: SimpleNamespace(
            synthesize=lambda query, nodes: SimpleNamespace(response=" 42 days ")
        ),
    )


def test_query_never_indexed_session_raises(monkeypatch):
    _patch_pipeline(monkeypatch, retrieved_nodes=[])

    with pytest.raises(SessionNotIndexedError):
        rag_service.query_rag("never-indexed-session", "hello")


def test_query_returns_answer_and_deduped_sources(monkeypatch):
    nodes = [
        _FakeNode(1, "EXPERIENCE"),
        _FakeNode(1, "EXPERIENCE"),
        _FakeNode(2, "SKILLS"),
    ]
    _patch_pipeline(monkeypatch, retrieved_nodes=nodes)

    result = rag_service.query_rag("s1", "how much leave?", model_id="gemini-2.5-flash")

    assert result["answer"] == "42 days"
    assert result["sources"] == [
        {"page": 1, "section": "EXPERIENCE"},
        {"page": 2, "section": "SKILLS"},
    ]
