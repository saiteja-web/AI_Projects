"""Pydantic schemas = API request/response shapes (DTOs).

Per the project structure, these live here alongside the SQLAlchemy models.
Think of them as TypeScript interfaces / Java DTOs — they validate and
shape the JSON that crosses the API boundary.
"""
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


# ── History endpoint responses ────────────────────────────────
class MessageOut(BaseModel):
    """One message as returned by GET /api/history/{session_id}."""

    id: UUID
    role: str
    content: str
    sources: list[int]
    created_at: datetime

    # Allow building from SQLAlchemy objects directly (ORM mode)
    model_config = {"from_attributes": True}


class HistoryOut(BaseModel):
    """Full history response: a session id and its messages."""

    session_id: UUID
    messages: list[MessageOut]


# ── Upload endpoint response (used in Phase 2) ────────────────
class UploadResponse(BaseModel):
    session_id: UUID
    filename: str
    status: str


# ── Chat endpoint request/response (used in Phase 3) ──────────
class ChatRequest(BaseModel):
    session_id: str
    query: str = Field(min_length=1)  # empty string → 422
    provider: str | None = None  # LLM provider ("ollama" or "openai")
    model_id: str | None = None  # Model identifier


class ChatResponse(BaseModel):
    answer: str
    sources: list[int]
    session_id: str
