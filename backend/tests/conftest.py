"""Shared pytest fixtures for the test suite.

Key idea: run the WHOLE test (setup + HTTP call + assertions) inside ONE
async event loop using httpx.AsyncClient. Mixing sync TestClient with manual
async DB calls collides on a shared asyncpg connection — so we go fully async.
"""
import asyncio

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from app.core.database import Base, async_session_factory, engine
from app.main import app
from app.models import db_models  # noqa: F401  (registers tables on Base.metadata)


@pytest.fixture(scope="session")
def event_loop():
    """One event loop for the entire test session — prevents loop-mixing bugs."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db_session():
    """Yield a clean AsyncSession and wipe data between tests."""
    # Ensure tables exist (ASGITransport skips the app's lifespan startup)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield async_session_factory

    # Truncate after each test so tests are fully isolated
    async with async_session_factory() as db:
        await db.execute(text("DELETE FROM messages"))
        await db.execute(text("DELETE FROM sessions"))
        await db.commit()


@pytest_asyncio.fixture
async def client(db_session):
    """Async HTTP client wired to the FastAPI app in-process."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
