"""Database service layer — all CRUD operations for sessions and messages.

This is the FastAPI equivalent of a Spring @Service class: it isolates
DB logic from the HTTP router so routes stay thin and the service is unit-testable.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import Message, Session


async def create_session(db: AsyncSession, filename: str) -> Session:
    """Insert a new session row and return it (with the generated UUID)."""
    session = Session(filename=filename)
    db.add(session)
    await db.commit()
    await db.refresh(session)  # reload to get server-generated id + created_at
    return session


async def save_message(
    db: AsyncSession,
    session_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> Message:
    """Insert one message row (either 'user' or 'ai')."""
    message = Message(
        session_id=session_id,
        role=role,
        content=content,
        sources=sources or [],
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message


async def get_history(db: AsyncSession, session_id: str) -> list[Message]:
    """Return all messages for a session, oldest first."""
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
    )
    return list(result.scalars().all())
