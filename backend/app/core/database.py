"""Async SQLAlchemy engine + session factory.

- engine:      the connection pool (one per app, like Spring's DataSource)
- session:     per-request DB session (like JPA EntityManager)
- get_db():    FastAPI dependency that hands a session to a route and closes it
               when the request ends (FastAPI's Dependency Injection ≈ @Autowired)
"""
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

# echo=True logs SQL to the console — handy while learning; turn off in prod
engine = create_async_engine(settings.database_url, echo=True, future=True)

# factory that produces AsyncSession objects
async_session_factory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    """All SQLAlchemy models inherit from this. It tracks table metadata."""
    pass


async def get_db() -> AsyncSession:
    """FastAPI dependency: yields a session, closes it after the request.

    Usage in a router:
        async def my_route(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_factory() as session:
        yield session
