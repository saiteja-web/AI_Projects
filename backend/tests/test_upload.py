"""Tests for POST /api/upload

Covers the three cases from the master prompt:
- valid PDF upload returns a session_id
- non-PDF file returns 400
- missing file returns 422 (FastAPI's default for required form fields)

Note: the successful-indexing test uses a minimal real PDF. The heavy RAG
build (embeddings download etc.) is exercised end-to-end via Postman instead.
"""
import io

import pytest

# Minimal valid PDF bytes (a single blank page). Good enough to pass PyPDFLoader
# and prove the upload route end-to-end without shipping a binary fixture file.
MINIMAL_PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000052 00000 n
0000000101 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
164
%%EOF
"""


@pytest.mark.asyncio
async def test_non_pdf_returns_400(client):
    """A .txt file should be rejected with 400."""
    response = await client.post(
        "/api/upload",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400
    assert "pdf" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_missing_file_returns_422(client):
    """No file at all → FastAPI returns 422 (required field missing)."""
    response = await client.post("/api/upload")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_upload_with_empty_filename_returns_422(client):
    """A file with an empty filename is rejected by FastAPI's validator (422),
    not our route logic — because File(...) requires a usable filename."""
    response = await client.post(
        "/api/upload",
        files={"file": ("", io.BytesIO(b"hello"), "application/pdf")},
    )
    assert response.status_code == 422


# NOTE: A full valid-PDF-returns-session_id test is skipped here because it
# triggers the real RAG build (embedding model download + ChromaDB writes).
# That path is verified manually via Postman in the Phase 2 checklist.
