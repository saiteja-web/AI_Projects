"""Integration tests for GET /api/history/{session_id}.

Fully async: setup, HTTP call, and assertions all run in one event loop via
the `client` and `db_session` fixtures from conftest.py.
"""
import uuid

import pytest

from app.services import db_service


@pytest.mark.asyncio
async def test_history_returns_404_for_nonexistent_session(client):
    """A random UUID that doesn't exist should return 404."""
    response = await client.get(f"/api/history/{uuid.uuid4()}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_history_returns_empty_for_existing_session(db_session, client):
    """A newly created session with no messages should return an empty list."""
    async with db_session() as db:
        session = await db_service.create_session(db, "test.pdf")

    response = await client.get(f"/api/history/{session.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == str(session.id)
    assert body["messages"] == []


@pytest.mark.asyncio
async def test_history_returns_messages_in_order(db_session, client):
    """Saved messages should come back oldest-first with their sources."""
    async with db_session() as db:
        session = await db_service.create_session(db, "doc.pdf")
        await db_service.save_message(db, str(session.id), "user", "What is RAG?", [])
        await db_service.save_message(db, str(session.id), "ai", "RAG is...", [1, 2])

    response = await client.get(f"/api/history/{session.id}")
    assert response.status_code == 200
    messages = response.json()["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is RAG?"
    assert messages[1]["role"] == "ai"
    assert messages[1]["sources"] == [1, 2]


@pytest.mark.asyncio
async def test_health_check_still_works(client):
    """Sanity check that the app boots and /health responds."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
