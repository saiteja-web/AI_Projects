"""Tests for POST /api/chat

We test the contract (status codes + response shape) without burning real
LLM calls: the 404 and 422 paths need no LLM, and the happy path mocks
query_rag_chain so it stays fast, free, and deterministic.
"""
from unittest.mock import patch

import pytest

from app.services import db_service


@pytest.mark.asyncio
async def test_chat_returns_404_for_unknown_session(client):
    """A session_id that doesn't exist → 404 before any RAG call."""
    response = await client.post(
        "/api/chat",
        json={"session_id": "00000000-0000-0000-0000-000000000000", "query": "hi"},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_chat_returns_422_for_empty_query(db_session, client):
    """An empty query string fails Pydantic validation → 422."""
    async with db_session() as db:
        session = await db_service.create_session(db, "x.pdf")

    response = await client.post(
        "/api/chat",
        json={"session_id": str(session.id), "query": ""},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_returns_answer_and_persists_messages(db_session, client):
    """Happy path: mock the RAG call, verify response + DB persistence."""
    async with db_session() as db:
        session = await db_service.create_session(db, "doc.pdf")

    fake_rag = {
        "answer": "You get 24 paid leave days.",
        "sources": [{"page": 1, "section": "Leave Policy"}],
    }

    with patch(
        "app.routers.chat.rag_service.query_rag_chain",
        return_value=fake_rag,
    ):
        response = await client.post(
            "/api/chat",
            json={"session_id": str(session.id), "query": "How much leave do I get?"},
        )

    # 1. Response is correct
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "You get 24 paid leave days."
    assert body["sources"] == [{"page": 1, "section": "Leave Policy"}]
    assert body["session_id"] == str(session.id)

    # 2. Both messages were persisted
    history_resp = await client.get(f"/api/history/{session.id}")
    messages = history_resp.json()["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "How much leave do I get?"
    assert messages[1]["role"] == "ai"
    assert messages[1]["content"] == "You get 24 paid leave days."
    assert messages[1]["sources"] == [{"page": 1, "section": "Leave Policy"}]


@pytest.mark.asyncio
async def test_chat_returns_404_when_session_has_no_indexed_document(db_session, client):
    """A session that exists in the DB but has no indexed document → 404."""
    async with db_session() as db:
        session = await db_service.create_session(db, "never-indexed.pdf")

    with patch(
        "app.routers.chat.rag_service.query_rag_chain",
        side_effect=ValueError(f"No document indexed for session {session.id}"),
    ):
        response = await client.post(
            "/api/chat",
            json={"session_id": str(session.id), "query": "anything"},
        )

    assert response.status_code == 404
    assert "No document indexed" in response.json()["detail"]
