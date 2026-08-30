"""Chat router — POST /api/chat

Takes a session_id + question, runs the RAG query against the session's
document, persists both the user question and the AI answer, and returns
the answer with source page citations.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.schemas import ChatRequest, ChatResponse
from app.services import db_service, rag_service
from app.services.rag_service import SessionNotIndexedError

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, db: AsyncSession = Depends(get_db)
) -> ChatResponse:
    """Answer a question using the session's indexed document."""
    # 1. Verify the session exists (404 if not)
    from sqlalchemy import select
    from app.models.db_models import Session

    exists = await db.execute(
        select(Session).where(Session.id == request.session_id)
    )
    if exists.scalars().first() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {request.session_id} not found",
        )

    # 2. Run the RAG query (retrieves chunks + asks the LLM). The sync
    #    LlamaIndex/Gemini calls run in the threadpool: they expect a thread
    #    with no running event loop, and this keeps the loop free meanwhile.
    try:
        result = await run_in_threadpool(
            rag_service.query_rag,
            str(request.session_id),
            request.query,
            request.model_id,
        )
    except SessionNotIndexedError as e:
        # e.g. session exists but no document was ever indexed for it
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query failed: {str(e) if str(e) else 'LLM API error - please check model configuration'}",
        ) from e

    # 3. Persist BOTH messages so /api/history can replay the conversation
    await db_service.save_message(db, request.session_id, "user", request.query, [])
    await db_service.save_message(
        db, request.session_id, "ai", result["answer"], result["sources"]
    )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=request.session_id,
    )
