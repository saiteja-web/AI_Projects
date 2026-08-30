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
    rag_service.rag_indexes.clear()
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

    result = rag_service.query_rag("s1", "how much leave?", model_id="gemini-3.6-flash")

    assert result["answer"] == "42 days"
    assert result["sources"] == [
        {"page": 1, "section": "EXPERIENCE"},
        {"page": 2, "section": "SKILLS"},
    ]


def test_build_rag_index_rejects_empty_parse(monkeypatch):
    _patch_pipeline(monkeypatch, retrieved_nodes=[])
    monkeypatch.setattr(rag_service, "parse_pdf_to_nodes", lambda p, s: [])

    with pytest.raises(ValueError, match="No extractable text"):
        rag_service.build_rag_index(
            "f.pdf", "empty-parse-session", model_id="gemini-3.6-flash"
        )


def test_build_then_query_hits_cache_without_rebuild(monkeypatch):
    builds = {"count": 0}

    class _CountingIndex:
        def __init__(self, nodes, vector_store=None, **kwargs):
            builds["count"] += 1

        def as_retriever(self, **kwargs):
            return _FakeRetriever([_FakeNode(1, "SKILLS")])

    _patch_pipeline(monkeypatch, retrieved_nodes=[_FakeNode(1, "SKILLS")])
    monkeypatch.setattr(rag_service, "VectorStoreIndex", _CountingIndex)
    monkeypatch.setattr(
        rag_service, "parse_pdf_to_nodes", lambda p, s: [_FakeNode(1, "SKILLS")]
    )

    rag_service.build_rag_index("f.pdf", "s1", model_id="m1")
    result = rag_service.query_rag("s1", "any question", model_id="m1")

    assert builds["count"] == 1  # cache hit: no index rebuild
    assert result["answer"] == "42 days"
    assert result["sources"] == [{"page": 1, "section": "SKILLS"}]


def test_build_rag_index_backs_index_with_our_store(monkeypatch):
    """Regression: the pgvector store must back the built index.

    Core 0.14 silently ignores VectorStoreIndex(vector_store=...) — the store
    must arrive via storage_context, else the index is backed by an in-memory
    SimpleVectorStore and nothing persists.
    """
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores import SimpleVectorStore

    fake_store = SimpleVectorStore()
    real_node = TextNode(
        text="SKILLS content", metadata={"page": 1, "section": "SKILLS"}
    )
    _patch_pipeline(monkeypatch, retrieved_nodes=[_FakeNode(1, "SKILLS")])
    monkeypatch.setattr(rag_service, "_vector_store", lambda: fake_store)
    monkeypatch.setattr(
        rag_service, "parse_pdf_to_nodes", lambda p, s: [real_node]
    )
    monkeypatch.setattr(
        rag_service.LlamaSettings, "_embed_model", MockEmbedding(embed_dim=384)
    )
    monkeypatch.setattr(
        rag_service,
        "_synthesizer_for",
        lambda model_id: SimpleNamespace(
            synthesize=lambda query, nodes: SimpleNamespace(response="ok")
        ),
    )
    rag_service.rag_indexes.clear()

    retriever, _ = rag_service.build_rag_index("f.pdf", "s1", model_id="m1")

    assert retriever._index._vector_store is fake_store
