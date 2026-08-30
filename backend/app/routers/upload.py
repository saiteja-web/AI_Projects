"""Upload router — POST /api/upload

Accepts a PDF, indexes it into pgvector, and creates a chat session.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.schemas import UploadResponse
from app.services import db_service, rag_service

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    """Upload a PDF, index it for RAG, and return a new session_id."""
    # 1. Validate the file is a PDF
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted",
        )

    # 2. Create the DB session first (so we have a session_id to name the collection)
    session = await db_service.create_session(db, file.filename)
    session_id = str(session.id)

    # 3. Save the uploaded PDF to /tmp/uploads
    try:
        file_path = rag_service.save_uploaded_pdf(file, session_id)
    finally:
        await file.close()

    # 4. Build the RAG index (parse + embed + store in pgvector)
    #    This is synchronous/heavy — brute-force OK for dev. Production would use
    #    a background task (FastAPI BackgroundTasks) so the HTTP call returns fast.
    try:
        rag_service.build_rag_index(file_path, session_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document: {e}",
        ) from e

    return UploadResponse(
        session_id=session.id, filename=file.filename, status="ready"
    )
