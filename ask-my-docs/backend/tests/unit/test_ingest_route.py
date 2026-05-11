"""Integration tests for the /ingest and /ingest/status routes.

VectorStore, BM25Store, and Embedder are patched so no ChromaDB or ML
model is required. sentence_transformers is stubbed in conftest.py.
"""
from __future__ import annotations

import io
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app

# ------------------------------------------------------------------
# Minimal valid one-page PDF (hand-crafted, ~500 bytes)
# ------------------------------------------------------------------
_MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
    b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    b"4 0 obj<</Length 44>>\nstream\nBT /F1 12 Tf 100 700 Td"
    b" (Hello World) Tj ET\nendstream\nendobj\n"
    b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"xref\n0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"0000000266 00000 n \n"
    b"0000000360 00000 n \n"
    b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n441\n%%EOF\n"
)


@pytest.fixture(autouse=True)
def patch_singletons():
    """Replace the three lazy singletons with lightweight mocks for every test."""
    mock_vs = MagicMock()
    mock_bm25 = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_chunks.return_value = []

    with (
        patch.object(main_module, "_get_vector_store", return_value=mock_vs),
        patch.object(main_module, "_get_bm25_store", return_value=mock_bm25),
        patch.object(main_module, "_get_embedder", return_value=mock_embedder),
    ):
        # Also reset the job registry between tests
        main_module._jobs.clear()
        yield


@pytest.fixture
def client():
    return TestClient(app)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_ingest_returns_202_with_job_id(client):
    resp = client.post(
        "/ingest",
        files={"file": ("book.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["status"] == "processing"


def test_ingest_rejects_non_pdf(client):
    resp = client.post(
        "/ingest",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 400
    assert "pdf" in resp.json()["detail"].lower()


def test_ingest_rejects_oversized_file(client):
    big = io.BytesIO(b"%" + b"x" * (51 * 1024 * 1024))
    resp = client.post(
        "/ingest",
        files={"file": ("big.pdf", big, "application/pdf")},
    )
    assert resp.status_code == 400
    assert "50" in resp.json()["detail"]


def test_status_unknown_job_returns_404(client):
    resp = client.get("/ingest/status/no-such-job")
    assert resp.status_code == 404


def test_status_tracks_known_job(client):
    resp = client.post(
        "/ingest",
        files={"file": ("book.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")},
    )
    job_id = resp.json()["job_id"]

    # Poll until the background task finishes (max 10s)
    for _ in range(20):
        status = client.get(f"/ingest/status/{job_id}").json()
        if status["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    assert status["status"] in ("done", "failed"), f"Job never finished: {status}"
    assert 0.0 <= status.get("progress", 0) <= 1.0


def test_status_done_includes_summary(client):
    resp = client.post(
        "/ingest",
        files={"file": ("book.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")},
    )
    job_id = resp.json()["job_id"]

    for _ in range(20):
        status = client.get(f"/ingest/status/{job_id}").json()
        if status["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    if status["status"] == "done":
        assert "pages" in status
        assert "chunks" in status
        assert "duration_seconds" in status


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "ollama" in body
    assert "chroma" in body
