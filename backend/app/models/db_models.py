"""SQLAlchemy ORM models = the database table definitions.

These are the FastAPI equivalent of Spring Boot @Entity classes —
SQLAlchemy reads them and creates the tables on startup.

NOTE: Kept separate from Pydantic schemas (schemas.py) so DB shape and
API shape can evolve independently. This separation is the 'clean
architecture' the resume calls out.
"""
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Session(Base):
    """A chat session tied to one uploaded PDF document."""

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # One session has many messages (like @OneToMany in JPA)
    messages: Mapped[list["Message"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Message(Base):
    """One message in a chat session — either from the user or the AI."""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(10), nullable=False)  # "user" | "ai"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # JSONB = PostgreSQL's native JSON type. Stores the list of source page numbers.
    sources: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # The other side of the relationship (like @ManyToOne in JPA)
    session: Mapped["Session"] = relationship(back_populates="messages")
