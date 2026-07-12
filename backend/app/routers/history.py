"""History router — GET /api/history/{session_id}

Returns all messages for a chat session, oldest first.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.schemas import HistoryOut, MessageOut
from app.services import db_service

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/history/{session_id}", response_model=HistoryOut)
async def get_history(session_id: UUID, db: AsyncSession = Depends(get_db)) -> HistoryOut:
    """Return the full message history for one session.

    Raises 404 if the session doesn't exist (no messages AND no session row).
    """
    messages = await db_service.get_history(db, session_id)

    # If no messages, verify the session itself exists — otherwise 404
    if not messages:
        from sqlalchemy import select
        from app.models.db_models import Session

        exists = await db.execute(select(Session).where(Session.id == session_id))
        if exists.scalars().first() is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )

    return HistoryOut(
        session_id=session_id,
        messages=[MessageOut.model_validate(m) for m in messages],
    )
