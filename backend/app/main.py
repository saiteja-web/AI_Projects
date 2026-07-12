"""FastAPI application entry point.

Wires up:
- Table creation on startup (Phase 1)
- CORS middleware (allows the React frontend to call us)
- A basic /health endpoint
- Router registration (history added in Phase 1; upload/chat in later phases)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import Base, engine
# Import models so SQLAlchemy registers them with Base.metadata before create_all
from app.models import db_models  # noqa: F401
from app.routers import chat, history, upload


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables on startup. In production you'd use Alembic migrations."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="AI Document Chat API", version="0.1.0", lifespan=lifespan)

# Allow the Vite dev server to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(history.router)
app.include_router(upload.router)
app.include_router(chat.router)


@app.get("/health")
async def health_check() -> dict:
    """Smoke test — confirms the server is alive."""
    return {"status": "ok"}
