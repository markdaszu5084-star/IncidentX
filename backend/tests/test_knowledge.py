"""Unit and integration tests for Knowledge Base ingestion pipeline (Stage 1)."""

import io
import shutil
import tempfile
from pathlib import Path
import pytest
import pypdf
from fastapi.testclient import TestClient

from app.knowledge.chunker import IntelligentChunker
from app.knowledge.exceptions import (
    CorruptedDocumentException,
    DocumentNotFoundException,
    FileSizeExceededException,
    InvalidFileTypeException,
)
from app.knowledge.extractor import (
    ExtractedDocumentContent,
    clean_text,
    extract_document_content,
    extract_from_pdf,
    extract_from_txt,
)
from app.knowledge.service import KnowledgeDocumentService
from app.knowledge.storage import DocumentStorageManager
from app.knowledge.validator import sanitize_filename, validate_document_upload
from app.main import app
from app.schemas.knowledge import (
    ChunkListResponse,
    ChunkingConfig,
    DocumentChunk,
    DocumentMetadata,
    DocumentUploadResponse,
)


def create_sample_pdf_bytes(pages_text: list[str]) -> bytes:
    """Helper to generate a valid PDF byte string in-memory with text."""
    writer = pypdf.PdfWriter()
    for text in pages_text:
        # Note: In pypdf, adding a page and writing a basic PDF structure
        page = writer.add_blank_page(width=300, height=300)
    
    # We can write out the PDF bytes
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def temp_storage():
    """Fixture providing an isolated temporary storage directory."""
    temp_dir = tempfile.mkdtemp(prefix="incidentx_kb_test_")
    manager = DocumentStorageManager(base_dir=temp_dir)
    yield manager, temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def client() -> TestClient:
    """Fixture providing FastAPI TestClient."""
    return TestClient(app)


# --- 1. Validator Tests ---

def test_sanitize_filename():
    """Verify filename sanitization removes malicious directory traversal."""
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("runbook v1.0 (final).txt") == "runbook_v1.0_final.txt"
    assert sanitize_filename("C:\\Windows\\System32\\cmd.exe") == "cmd.exe"
    assert sanitize_filename("") == "unnamed_document.txt"


def test_validate_document_txt_success():
    """Verify validation passes for valid text documents."""
    filename, file_type = validate_document_upload(
        filename="db_troubleshooting.txt",
        content=b"PostgreSQL connection pool exhaustion guide.",
    )
    assert filename == "db_troubleshooting.txt"
    assert file_type == "txt"


def test_validate_document_pdf_success():
    """Verify validation passes for standard PDF header."""
    fake_pdf = b"%PDF-1.7\nSample PDF payload"
    filename, file_type = validate_document_upload(
        filename="incident_playbook.pdf",
        content=fake_pdf,
    )
    assert filename == "incident_playbook.pdf"
    assert file_type == "pdf"


def test_validate_unsupported_extension():
    """Verify rejection of unsupported extensions."""
    with pytest.raises(InvalidFileTypeException) as exc_info:
        validate_document_upload(filename="script.sh", content=b"#!/bin/bash")
    assert "Unsupported file format" in str(exc_info.value)


def test_validate_empty_file():
    """Verify rejection of empty files (0 bytes)."""
    with pytest.raises(FileSizeExceededException) as exc_info:
        validate_document_upload(filename="empty.txt", content=b"")
    assert "empty" in str(exc_info.value).lower()


def test_validate_file_size_exceeded():
    """Verify rejection of files exceeding max limit."""
    content = b"X" * (1024 * 1024 + 10)  # 1MB + 10 bytes
    with pytest.raises(FileSizeExceededException) as exc_info:
        validate_document_upload(
            filename="large.txt",
            content=content,
            max_size_bytes=1024 * 1024,  # 1MB limit
        )
    assert "exceeds maximum allowed limit" in str(exc_info.value)


def test_validate_fake_pdf_header_rejected():
    """Verify rejection of .pdf extension without standard %PDF- header."""
    with pytest.raises(InvalidFileTypeException) as exc_info:
        validate_document_upload(
            filename="fake_doc.pdf",
            content=b"This is actually a plain text file renamed to .pdf",
        )
    assert "missing standard '%PDF-'" in str(exc_info.value)


def test_validate_disallowed_binary_in_text():
    """Verify rejection of executable binaries masquerading as .txt."""
    with pytest.raises(InvalidFileTypeException) as exc_info:
        validate_document_upload(
            filename="trojan.txt",
            content=b"MZ\x90\x00\x03\x00\x00\x00executable bytes",
        )
    assert "binary executable" in str(exc_info.value).lower()


# --- 2. Extractor Tests ---

def test_clean_text_normalization():
    """Verify text cleaning collapses whitespace and strips control codes."""
    raw = "Header line\r\n\r\nLine with   extra    spaces \x00\x07and tabs\t\n\n\n\nFooter"
    cleaned = clean_text(raw)
    assert "\r" not in cleaned
    assert "extra spaces" in cleaned
    assert "\x07" not in cleaned
    assert "\n\n\n" not in cleaned  # Max 2 consecutive newlines


def test_extract_from_txt():
    """Verify extraction and metrics from plain text."""
    content = "Incident Runbook:\n1. Check database connections.\n2. Restart pool."
    extracted = extract_from_txt(content.encode("utf-8"))
    assert extracted.character_count == len(content)
    assert extracted.word_count == 9
    assert extracted.page_count is None
    assert "Check database connections" in extracted.text


def test_extract_from_pdf():
    """Verify extraction and page metrics from PDF."""
    pdf_bytes = create_sample_pdf_bytes(["Page 1", "Page 2"])
    extracted = extract_from_pdf(pdf_bytes)
    assert extracted.page_count == 2
    assert len(extracted.page_details) == 2
    assert extracted.page_details[0]["page_number"] == 1


# --- 3. Storage Tests ---

def test_storage_lifecycle(temp_storage):
    """Verify document storage, retrieval, listing, and deletion."""
    storage, temp_dir = temp_storage

    doc_id = "doc_test12345"
    filename = "runbook.txt"
    content = b"Sample runbook content for testing."
    full_text = "Sample runbook content for testing."

    metadata = DocumentMetadata(
        document_id=doc_id,
        filename=filename,
        file_type="txt",
        file_size_bytes=len(content),
        character_count=len(full_text),
        word_count=5,
        page_count=None,
        sha256_hash=storage.compute_sha256(content),
        storage_path=f"raw/{doc_id}_{filename}",
    )

    # 1. Store
    storage.store_document(
        document_id=doc_id,
        filename=filename,
        content=content,
        metadata=metadata,
        extracted_text=full_text,
    )

    # 2. Get metadata
    retrieved_meta = storage.get_metadata(doc_id)
    assert retrieved_meta.document_id == doc_id
    assert retrieved_meta.filename == filename
    assert retrieved_meta.file_size_bytes == len(content)

    # 3. Get text
    retrieved_text = storage.get_text(doc_id)
    assert retrieved_text == full_text

    # 4. List documents
    docs = storage.list_documents()
    assert len(docs) == 1
    assert docs[0].document_id == doc_id

    # 5. Delete document
    assert storage.delete_document(doc_id) is True
    assert len(storage.list_documents()) == 0

    with pytest.raises(DocumentNotFoundException):
        storage.get_metadata(doc_id)


# --- 4. API Integration Tests ---

def test_api_upload_txt_success(client: TestClient):
    """Test POST /api/v1/knowledge/upload with valid text file."""
    txt_payload = b"Runbook: When Auth Service fails with HTTP 504, check Redis cache connectivity."
    files = {"file": ("auth_service_runbook.txt", txt_payload, "text/plain")}

    response = client.post("/api/v1/knowledge/upload", files=files)
    assert response.status_code == 201

    data = response.json()
    model = DocumentUploadResponse(**data)
    assert model.status == "success"
    assert model.document.file_type == "txt"
    assert model.document.filename == "auth_service_runbook.txt"
    assert "Redis cache" in model.extracted_text_preview

    doc_id = model.document.document_id

    # Verify document is in listing
    list_res = client.get("/api/v1/knowledge/documents")
    assert list_res.status_code == 200
    docs = list_res.json()["documents"]
    assert any(d["document_id"] == doc_id for d in docs)

    # Verify document details
    detail_res = client.get(f"/api/v1/knowledge/documents/{doc_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["metadata"]["document_id"] == doc_id
    assert "Redis cache" in detail_res.json()["extracted_text"]

    # Verify raw text endpoint
    text_res = client.get(f"/api/v1/knowledge/documents/{doc_id}/text")
    assert text_res.status_code == 200
    assert "text/plain" in text_res.headers.get("content-type", "")
    assert "Redis cache" in text_res.text

    # Clean up
    del_res = client.delete(f"/api/v1/knowledge/documents/{doc_id}")
    assert del_res.status_code == 200


def test_api_upload_pdf_success(client: TestClient):
    """Test POST /api/v1/knowledge/upload with valid PDF file."""
    pdf_bytes = create_sample_pdf_bytes(["Database Troubleshooting Guide", "Step 2: Check active connections"])
    files = {"file": ("database_runbook.pdf", pdf_bytes, "application/pdf")}

    response = client.post("/api/v1/knowledge/upload", files=files)
    assert response.status_code == 201

    data = response.json()
    model = DocumentUploadResponse(**data)
    assert model.status == "success"
    assert model.document.file_type == "pdf"
    assert model.document.page_count == 2
    assert model.document.filename == "database_runbook.pdf"

    # Clean up
    doc_id = model.document.document_id
    client.delete(f"/api/v1/knowledge/documents/{doc_id}")


def test_api_upload_unsupported_file_type(client: TestClient):
    """Test POST /api/v1/knowledge/upload rejects unsupported extensions with HTTP 400."""
    files = {"file": ("malicious.exe", b"MZexecutable bytes", "application/octet-stream")}
    response = client.post("/api/v1/knowledge/upload", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_api_upload_fake_pdf(client: TestClient):
    """Test POST /api/v1/knowledge/upload rejects fake PDFs with HTTP 400."""
    files = {"file": ("fake.pdf", b"This is plain text with no PDF header", "application/pdf")}
    response = client.post("/api/v1/knowledge/upload", files=files)
    assert response.status_code == 400
    assert "missing standard '%PDF-'" in response.json()["detail"]


def test_api_upload_empty_file(client: TestClient):
    """Test POST /api/v1/knowledge/upload rejects empty files."""
    files = {"file": ("empty.txt", b"", "text/plain")}
    response = client.post("/api/v1/knowledge/upload", files=files)
    assert response.status_code in (400, 413)


def test_api_get_nonexistent_document(client: TestClient):
    """Test GET /api/v1/knowledge/documents/{id} returns 404 for invalid ID."""
    response = client.get("/api/v1/knowledge/documents/doc_nonexistent_999")
    assert response.status_code == 404


# --- 5. Stage 2 Intelligent Chunking Tests ---

def test_chunking_config_validation():
    """Verify ChunkingConfig validates parameters correctly."""
    # Valid
    config = ChunkingConfig(chunk_size=1000, chunk_overlap=150)
    assert config.chunk_size == 1000
    assert config.chunk_overlap == 150

    # Overlap >= size raises ValueError
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=500, chunk_overlap=500)

    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=500, chunk_overlap=600)

    # Size too small
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=50, chunk_overlap=10)


def test_intelligent_chunker_empty_document():
    """Verify empty document produces 0 chunks."""
    chunker = IntelligentChunker(chunk_size=1000, chunk_overlap=150)
    chunks = chunker.chunk_text(
        text="",
        document_id="doc_empty",
        filename="empty.txt",
    )
    assert len(chunks) == 0

    # Whitespace only
    chunks_ws = chunker.chunk_text(
        text="   \n\n\t   ",
        document_id="doc_empty_ws",
        filename="empty_ws.txt",
    )
    assert len(chunks_ws) == 0


def test_intelligent_chunker_very_short_document():
    """Verify document shorter than chunk_size produces exactly 1 chunk."""
    chunker = IntelligentChunker(chunk_size=1000, chunk_overlap=150)
    short_text = "Postgres connection pool timeout: restart pgpool service."
    chunks = chunker.chunk_text(
        text=short_text,
        document_id="doc_short",
        filename="short.txt",
    )
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == short_text
    assert chunks[0].chunk_id == "doc_short_c0000"
    assert chunks[0].character_count == len(short_text)
    assert chunks[0].word_count == len(short_text.split())


def test_intelligent_chunker_normal_document():
    """Verify multi-paragraph document is split cleanly respecting boundaries."""
    chunker = IntelligentChunker(chunk_size=300, chunk_overlap=50)
    para1 = "Section 1: Database Connection Saturation.\nWhen connection slots reach 100%, clients receive connection refused."
    para2 = "Section 2: Immediate Remediation Steps.\n1. Check current connection count via pg_stat_activity.\n2. Terminate idle connections."
    para3 = "Section 3: Long-term Fix.\nIncrease max_connections in postgresql.conf and reload configuration."
    full_text = f"{para1}\n\n{para2}\n\n{para3}"

    chunks = chunker.chunk_text(
        text=full_text,
        document_id="doc_normal",
        filename="postgres_guide.txt",
    )
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.text) <= 350  # Respects chunk size within reasonable delimiter bounds
        assert chunk.document_id == "doc_normal"
        assert chunk.filename == "postgres_guide.txt"


def test_intelligent_chunker_large_document():
    """Verify large technical document splits into multiple well-formed chunks."""
    chunker = IntelligentChunker(chunk_size=500, chunk_overlap=100)
    # Generate 3000 chars of structured incident runbook text
    paragraphs = [
        f"Incident Step {i}: Check subsystem {i} for latency spikes and thread pool exhaustion. "
        f"Ensure memory utilization remains below threshold. Execute healthcheck /ping."
        for i in range(1, 25)
    ]
    large_text = "\n\n".join(paragraphs)

    chunks = chunker.chunk_text(
        text=large_text,
        document_id="doc_large",
        filename="large_runbook.txt",
    )
    assert len(chunks) >= 5
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.chunk_id == f"doc_large_c{i:04d}"
        assert len(c.text) > 0


def test_intelligent_chunker_overlap_verification():
    """Verify consecutive chunks share the configured overlapping content."""
    chunk_size = 200
    chunk_overlap = 80
    chunker = IntelligentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    sentence1 = "Alpha node reported packet loss of 25% at 22:30."
    sentence2 = "Beta node initiated failover sequence to redundant region."
    sentence3 = "Gamma gateway confirmed traffic reroute completed successfully."
    sentence4 = "Delta cluster monitored error rate until normal levels resumed."
    text = f"{sentence1}\n\n{sentence2}\n\n{sentence3}\n\n{sentence4}"

    chunks = chunker.chunk_text(
        text=text,
        document_id="doc_overlap",
        filename="network_incident.txt",
    )
    assert len(chunks) >= 2

    # Check that chunk 0 and chunk 1 share overlapping words
    words_chunk0 = set(chunks[0].text.split())
    words_chunk1 = set(chunks[1].text.split())
    shared_words = words_chunk0.intersection(words_chunk1)
    assert len(shared_words) > 0, "Consecutive chunks must share overlapping words"


def test_intelligent_chunker_metadata_preservation():
    """Verify every chunk retains document_id, filename, index, page_number, and custom attributes."""
    chunker = IntelligentChunker(chunk_size=400, chunk_overlap=50)
    content = ExtractedDocumentContent(
        text="Single text stream",
        character_count=18,
        word_count=3,
        page_count=None,
        page_details=[],
    )

    custom_meta = {"environment": "production", "service": "billing"}
    chunks = chunker.chunk_extracted_content(
        document_id="doc_meta_test",
        filename="billing_runbook.txt",
        content=content,
        custom_metadata=custom_meta,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.document_id == "doc_meta_test"
    assert chunk.filename == "billing_runbook.txt"
    assert chunk.chunk_index == 0
    assert chunk.metadata["source"] == "billing_runbook.txt"
    assert chunk.metadata["environment"] == "production"
    assert chunk.metadata["service"] == "billing"
    assert chunk.metadata["chunk_size"] == 400
    assert chunk.metadata["chunk_overlap"] == 50


def test_intelligent_chunker_multi_page_pdf_page_number():
    """Verify chunks from a multi-page PDF retain their respective page numbers."""
    chunker = IntelligentChunker(chunk_size=300, chunk_overlap=50)
    content = ExtractedDocumentContent(
        text="Page 1 text.\n\nPage 2 text.",
        character_count=26,
        word_count=6,
        page_count=2,
        page_details=[
            {"page_number": 1, "text": "Page 1: Auth service incident overview and initial diagnostics.", "character_count": 60, "word_count": 8},
            {"page_number": 2, "text": "Page 2: Remediation procedure and verification steps.", "character_count": 52, "word_count": 7},
        ],
    )

    chunks = chunker.chunk_extracted_content(
        document_id="doc_pdf_pages",
        filename="auth_playbook.pdf",
        content=content,
    )
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[0].metadata["page_number"] == 1
    assert "Page 1" in chunks[0].text

    assert chunks[1].page_number == 2
    assert chunks[1].metadata["page_number"] == 2
    assert "Page 2" in chunks[1].text


def test_storage_chunk_persistence(temp_storage):
    """Verify storing, retrieving, and individual chunk querying."""
    storage, _ = temp_storage
    doc_id = "doc_chunk_store_test"

    chunks = [
        DocumentChunk(
            chunk_id=f"{doc_id}_c0000",
            document_id=doc_id,
            filename="test.txt",
            chunk_index=0,
            text="Chunk 0 content",
            character_count=15,
            word_count=3,
            page_number=None,
            metadata={"source": "test.txt"},
        ),
        DocumentChunk(
            chunk_id=f"{doc_id}_c0001",
            document_id=doc_id,
            filename="test.txt",
            chunk_index=1,
            text="Chunk 1 content",
            character_count=15,
            word_count=3,
            page_number=None,
            metadata={"source": "test.txt"},
        ),
    ]

    # Save metadata first
    meta = DocumentMetadata(
        document_id=doc_id,
        filename="test.txt",
        file_type="txt",
        file_size_bytes=100,
        character_count=30,
        word_count=6,
        total_chunks=2,
        sha256_hash="dummyhash",
        storage_path=f"raw/{doc_id}_test.txt",
    )
    storage.store_document(doc_id, "test.txt", b"test", meta, "test")

    # Store chunks
    chunks_path = storage.store_chunks(doc_id, chunks)
    assert Path(chunks_path).exists()

    # Retrieve all chunks
    retrieved = storage.get_chunks(doc_id)
    assert len(retrieved) == 2
    assert retrieved[0].chunk_id == f"{doc_id}_c0000"
    assert retrieved[1].chunk_id == f"{doc_id}_c0001"

    # Retrieve single chunk
    chunk0 = storage.get_chunk(doc_id, 0)
    assert chunk0.text == "Chunk 0 content"

    # Out of range index
    with pytest.raises(DocumentNotFoundException):
        storage.get_chunk(doc_id, 99)


# --- 6. API Stage 2 Endpoints Integration Tests ---

def test_api_upload_with_custom_chunking_params(client: TestClient):
    """Test POST /api/v1/knowledge/upload with custom chunk_size and chunk_overlap query params."""
    txt_content = (
        "Runbook for Memory Leak in Cache Service.\n\n"
        "1. Identify cache node consuming > 90% memory.\n\n"
        "2. Trigger graceful evictions using redis-cli MEMORY PURGE.\n\n"
        "3. If leak persists, initiate failover to standby replica.\n\n"
        "4. Capture heap dump and notify platform on-call."
    )
    files = {"file": ("cache_leak.txt", txt_content.encode("utf-8"), "text/plain")}

    response = client.post(
        "/api/v1/knowledge/upload?chunk_size=150&chunk_overlap=30",
        files=files,
    )
    assert response.status_code == 201

    data = response.json()
    assert data["status"] == "success"
    assert data["chunk_size"] == 150
    assert data["chunk_overlap"] == 30
    assert data["total_chunks"] >= 2

    doc_id = data["document"]["document_id"]

    # Test GET /documents/{id}/chunks
    chunks_res = client.get(f"/api/v1/knowledge/documents/{doc_id}/chunks")
    assert chunks_res.status_code == 200
    chunk_data = chunks_res.json()
    assert chunk_data["document_id"] == doc_id
    assert chunk_data["total_chunks"] == data["total_chunks"]
    assert len(chunk_data["chunks"]) == data["total_chunks"]

    # Test GET /documents/{id}/chunks/0
    single_res = client.get(f"/api/v1/knowledge/documents/{doc_id}/chunks/0")
    assert single_res.status_code == 200
    assert single_res.json()["chunk_index"] == 0
    assert single_res.json()["document_id"] == doc_id

    # Test POST /documents/{id}/rechunk
    rechunk_res = client.post(f"/api/v1/knowledge/documents/{doc_id}/rechunk?chunk_size=500&chunk_overlap=50")
    assert rechunk_res.status_code == 200
    rechunked = rechunk_res.json()
    assert rechunked["chunk_size"] == 500
    assert rechunked["total_chunks"] < data["total_chunks"]  # Larger chunks = fewer chunks

    # Cleanup
    client.delete(f"/api/v1/knowledge/documents/{doc_id}")


def test_api_upload_invalid_chunk_overlap(client: TestClient):
    """Test POST /api/v1/knowledge/upload rejects chunk_overlap >= chunk_size."""
    files = {"file": ("test.txt", b"Valid text content", "text/plain")}
    response = client.post(
        "/api/v1/knowledge/upload?chunk_size=200&chunk_overlap=250",
        files=files,
    )
    assert response.status_code == 400
    assert "chunk_overlap" in response.json()["detail"]



# ===========================================================================
# Stage 3 — Embedding Generation Engine Tests
# ===========================================================================


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

import numpy as np
from app.knowledge.embeddings import EmbeddingService, get_embedding_service
from app.schemas.knowledge import (
    ChunkEmbedding,
    DocumentEmbeddingResponse,
    EmbeddingStatusResponse,
    QueryEmbeddingResponse,
)


@pytest.fixture(scope="module")
def embedding_service() -> EmbeddingService:
    """Module-scoped EmbeddingService; model loads only once for all embedding tests."""
    return EmbeddingService()


# ---------------------------------------------------------------------------
# 1. Model initialization — singleton caching
# ---------------------------------------------------------------------------

def test_embedding_service_singleton():
    """get_embedding_service() must return the identical object on repeated calls."""
    svc1 = get_embedding_service()
    svc2 = get_embedding_service()
    assert svc1 is svc2, "EmbeddingService singleton violated — two different instances returned."


def test_embedding_service_model_name(embedding_service: EmbeddingService):
    """The service must load with the configured model name."""
    assert "MiniLM" in embedding_service.model_name or "all-" in embedding_service.model_name


def test_embedding_service_expected_dimension(embedding_service: EmbeddingService):
    """Expected dimension must equal 384 for all-MiniLM-L6-v2."""
    assert embedding_service.expected_dimension == 384


# ---------------------------------------------------------------------------
# 2. Single text embedding
# ---------------------------------------------------------------------------

def test_embed_text_returns_list(embedding_service: EmbeddingService):
    """embed_text() must return a Python list of floats."""
    vector = embedding_service.embed_text("CPU utilisation exceeded 95% threshold.")
    assert isinstance(vector, list)
    assert all(isinstance(v, float) for v in vector)


def test_embed_text_correct_dimension(embedding_service: EmbeddingService):
    """embed_text() must produce a 384-element vector."""
    vector = embedding_service.embed_text("Disk I/O latency alert on /dev/sda.")
    assert len(vector) == 384


def test_embed_text_no_nan_or_inf(embedding_service: EmbeddingService):
    """The embedding vector must not contain NaN or Inf values."""
    vector = embedding_service.embed_text("Service degradation detected in auth-service.")
    arr = np.array(vector)
    assert not np.isnan(arr).any(), "NaN found in embedding vector."
    assert not np.isinf(arr).any(), "Inf found in embedding vector."


def test_embed_text_normalized(embedding_service: EmbeddingService):
    """With normalize_embeddings=True the L2 norm must be very close to 1.0."""
    vector = embedding_service.embed_text("Kubernetes pod restarted unexpectedly.")
    norm = float(np.linalg.norm(np.array(vector)))
    assert abs(norm - 1.0) < 1e-4, f"Expected unit vector (norm≈1), got {norm:.6f}."


def test_embed_text_different_texts_produce_different_vectors(embedding_service: EmbeddingService):
    """Two semantically different texts must produce distinct vectors."""
    v1 = embedding_service.embed_text("Database connection timeout.")
    v2 = embedding_service.embed_text("Pineapple pizza is controversial.")
    assert v1 != v2


def test_embed_text_similar_texts_closer_than_dissimilar(embedding_service: EmbeddingService):
    """Semantically similar texts should have higher cosine similarity than unrelated ones."""
    v_a = np.array(embedding_service.embed_text("Server CPU high utilisation."))
    v_b = np.array(embedding_service.embed_text("High CPU usage on the server."))
    v_c = np.array(embedding_service.embed_text("The cat sat on the mat."))
    sim_ab = float(np.dot(v_a, v_b))  # both normalised, dot == cosine sim
    sim_ac = float(np.dot(v_a, v_c))
    assert sim_ab > sim_ac, (
        f"Expected sim(a,b)={sim_ab:.4f} > sim(a,c)={sim_ac:.4f} but it was not."
    )


# ---------------------------------------------------------------------------
# 3. Empty / invalid input validation
# ---------------------------------------------------------------------------

def test_embed_text_empty_string_raises(embedding_service: EmbeddingService):
    """embed_text('') must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        embedding_service.embed_text("")


def test_embed_text_whitespace_only_raises(embedding_service: EmbeddingService):
    """embed_text('   ') must raise ValueError (only whitespace)."""
    with pytest.raises(ValueError, match="empty"):
        embedding_service.embed_text("   ")


def test_embed_text_newline_only_raises(embedding_service: EmbeddingService):
    """embed_text with only newlines must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        embedding_service.embed_text("\n\n\t")


# ---------------------------------------------------------------------------
# 4. Query embedding
# ---------------------------------------------------------------------------

def test_embed_query_returns_list(embedding_service: EmbeddingService):
    """embed_query() must return a Python list."""
    vector = embedding_service.embed_query("How do I restart the payment service?")
    assert isinstance(vector, list)


def test_embed_query_correct_dimension(embedding_service: EmbeddingService):
    """embed_query() must produce a 384-element vector."""
    vector = embedding_service.embed_query("What caused the memory leak?")
    assert len(vector) == 384


def test_embed_query_empty_raises(embedding_service: EmbeddingService):
    """embed_query('') must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        embedding_service.embed_query("")


def test_embed_query_same_space_as_documents(embedding_service: EmbeddingService):
    """Query and document embeddings must be in the same vector space (same dimension)."""
    q_vec = embedding_service.embed_query("CPU spike investigation")
    d_vec = embedding_service.embed_text("CPU utilization spiked to 98% during peak traffic.")
    assert len(q_vec) == len(d_vec) == 384


def test_embed_query_norm_near_one(embedding_service: EmbeddingService):
    """Query vector L2 norm must be close to 1.0."""
    vector = embedding_service.embed_query("Diagnose memory leak in Java service.")
    norm = float(np.linalg.norm(np.array(vector)))
    assert abs(norm - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# 5. Batch chunk embedding
# ---------------------------------------------------------------------------

def _make_chunks(n: int = 3) -> list:
    """Create n minimal DocumentChunk objects for testing."""
    from app.schemas.knowledge import DocumentChunk

    return [
        DocumentChunk(
            chunk_id=f"doc_test_c{i:04d}",
            document_id="doc_test",
            filename="test.txt",
            chunk_index=i,
            text=f"This is chunk number {i}. It describes an IT incident log entry.",
            character_count=60,
            word_count=12,
            metadata={"source": "test.txt"},
        )
        for i in range(n)
    ]


def test_embed_chunks_returns_correct_count(embedding_service: EmbeddingService):
    """embed_chunks() must return exactly one ChunkEmbedding per input chunk."""
    chunks = _make_chunks(3)
    result = embedding_service.embed_chunks(chunks)
    assert len(result) == 3


def test_embed_chunks_returns_chunk_embedding_objects(embedding_service: EmbeddingService):
    """embed_chunks() must return ChunkEmbedding instances."""
    chunks = _make_chunks(2)
    result = embedding_service.embed_chunks(chunks)
    for item in result:
        assert isinstance(item, ChunkEmbedding)


def test_embed_chunks_correct_dimension(embedding_service: EmbeddingService):
    """Every ChunkEmbedding must have dimension=384 and len(embedding)==384."""
    chunks = _make_chunks(2)
    result = embedding_service.embed_chunks(chunks)
    for item in result:
        assert item.dimension == 384
        assert len(item.embedding) == 384


def test_embed_chunks_metadata_preserved(embedding_service: EmbeddingService):
    """chunk_id, document_id, and chunk_index must be copied from the source chunk."""
    chunks = _make_chunks(3)
    result = embedding_service.embed_chunks(chunks)
    for original, embedded in zip(chunks, result):
        assert embedded.chunk_id == original.chunk_id
        assert embedded.document_id == original.document_id
        assert embedded.chunk_index == original.chunk_index


def test_embed_chunks_model_name_stored(embedding_service: EmbeddingService):
    """embedding_model must be populated in each ChunkEmbedding."""
    chunks = _make_chunks(1)
    result = embedding_service.embed_chunks(chunks)
    assert result[0].embedding_model == embedding_service.model_name


def test_embed_chunks_empty_list_returns_empty(embedding_service: EmbeddingService):
    """embed_chunks([]) must return an empty list without raising."""
    result = embedding_service.embed_chunks([])
    assert result == []


def test_embed_chunks_all_empty_text_raises(embedding_service: EmbeddingService):
    """embed_chunks with all-blank texts must raise ValueError."""
    from app.schemas.knowledge import DocumentChunk

    chunks = [
        DocumentChunk(
            chunk_id="doc_test_c0000",
            document_id="doc_test",
            filename="test.txt",
            chunk_index=0,
            text="   ",
            character_count=3,
            word_count=0,
            metadata={},
        )
    ]
    with pytest.raises(ValueError):
        embedding_service.embed_chunks(chunks)


def test_embed_chunks_no_nan_or_inf(embedding_service: EmbeddingService):
    """No embedding in a batch result may contain NaN or Inf."""
    chunks = _make_chunks(4)
    result = embedding_service.embed_chunks(chunks)
    for item in result:
        arr = np.array(item.embedding)
        assert not np.isnan(arr).any(), f"NaN in chunk {item.chunk_id}"
        assert not np.isinf(arr).any(), f"Inf in chunk {item.chunk_id}"


# ---------------------------------------------------------------------------
# 6. compute_vector_norm
# ---------------------------------------------------------------------------

def test_compute_vector_norm(embedding_service: EmbeddingService):
    """compute_vector_norm must return the correct L2 norm."""
    vector = [3.0, 4.0]  # norm = 5.0
    assert abs(embedding_service.compute_vector_norm(vector) - 5.0) < 1e-6


# ---------------------------------------------------------------------------
# 7. Storage — embedding persistence
# ---------------------------------------------------------------------------

def test_storage_store_and_retrieve_embeddings(tmp_path):
    """StorageManager must round-trip ChunkEmbedding objects through JSON."""
    from app.knowledge.storage import DocumentStorageManager

    storage = DocumentStorageManager(base_dir=str(tmp_path / "kb"))

    # Seed a document so get_metadata() does not raise
    from datetime import datetime, timezone
    from app.schemas.knowledge import DocumentMetadata

    meta = DocumentMetadata(
        document_id="doc_emb_test",
        filename="embed_test.txt",
        file_type="txt",
        file_size_bytes=100,
        character_count=50,
        word_count=10,
        sha256_hash="abc123",
        storage_path="raw/doc_emb_test_embed_test.txt",
        created_at=datetime.now(timezone.utc),
    )
    storage.save_metadata(meta)

    embeddings = [
        ChunkEmbedding(
            chunk_id="doc_emb_test_c0000",
            document_id="doc_emb_test",
            chunk_index=0,
            embedding=[0.1, 0.2, 0.3] + [0.0] * 381,
            dimension=384,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ),
    ]
    storage.store_embeddings("doc_emb_test", embeddings)

    loaded = storage.get_embeddings("doc_emb_test")
    assert len(loaded) == 1
    assert loaded[0].chunk_id == "doc_emb_test_c0000"
    assert loaded[0].embedding[0] == pytest.approx(0.1)


def test_storage_embeddings_exist(tmp_path):
    """embeddings_exist() must return False before and True after store_embeddings()."""
    from app.knowledge.storage import DocumentStorageManager
    from datetime import datetime, timezone
    from app.schemas.knowledge import DocumentMetadata

    storage = DocumentStorageManager(base_dir=str(tmp_path / "kb"))

    meta = DocumentMetadata(
        document_id="doc_exist_test",
        filename="exist.txt",
        file_type="txt",
        file_size_bytes=10,
        character_count=5,
        word_count=1,
        sha256_hash="xyz",
        storage_path="raw/doc_exist_test_exist.txt",
        created_at=datetime.now(timezone.utc),
    )
    storage.save_metadata(meta)

    assert not storage.embeddings_exist("doc_exist_test")

    storage.store_embeddings(
        "doc_exist_test",
        [
            ChunkEmbedding(
                chunk_id="doc_exist_test_c0000",
                document_id="doc_exist_test",
                chunk_index=0,
                embedding=[0.0] * 384,
                dimension=384,
                embedding_model="test-model",
            )
        ],
    )
    assert storage.embeddings_exist("doc_exist_test")


def test_storage_get_embeddings_missing_returns_empty(tmp_path):
    """get_embeddings() must return [] when no embeddings file exists."""
    from app.knowledge.storage import DocumentStorageManager
    from datetime import datetime, timezone
    from app.schemas.knowledge import DocumentMetadata

    storage = DocumentStorageManager(base_dir=str(tmp_path / "kb"))
    meta = DocumentMetadata(
        document_id="doc_no_emb",
        filename="no_emb.txt",
        file_type="txt",
        file_size_bytes=10,
        character_count=5,
        word_count=1,
        sha256_hash="zzz",
        storage_path="raw/doc_no_emb_no_emb.txt",
        created_at=datetime.now(timezone.utc),
    )
    storage.save_metadata(meta)
    result = storage.get_embeddings("doc_no_emb")
    assert result == []


# ---------------------------------------------------------------------------
# 8. API endpoint tests (embedding endpoints)
# ---------------------------------------------------------------------------

@pytest.fixture()
def client_with_doc(tmp_path, monkeypatch):
    """Return a TestClient with a pre-uploaded TXT document and its document_id."""
    from app.knowledge.service import KnowledgeDocumentService
    from app.knowledge.storage import DocumentStorageManager
    import app.api.knowledge as knowledge_router_module

    storage = DocumentStorageManager(base_dir=str(tmp_path / "kb"))
    svc = KnowledgeDocumentService(storage_manager=storage)
    monkeypatch.setattr(knowledge_router_module, "service", svc)

    client = TestClient(app)

    txt_content = (
        "Incident Report: High Memory Usage\n\n"
        "At 14:32 UTC the payment-service pod OOMKilled on node prod-worker-3.\n\n"
        "Root cause: unbounded cache growth in the session manager.\n\n"
        "Resolution: restarted pod; cache TTL set to 60 seconds.\n\n"
        "Follow-up: add memory limit alert at 85% threshold."
    )
    response = client.post(
        "/api/v1/knowledge/upload",
        files={"file": ("incident.txt", txt_content.encode(), "text/plain")},
    )
    assert response.status_code == 201
    doc_id = response.json()["document"]["document_id"]
    return client, doc_id


def test_api_embed_document_success(client_with_doc):
    """POST /embed must return 200 with total_embeddings_generated > 0."""
    client, doc_id = client_with_doc
    response = client.post(f"/api/v1/knowledge/documents/{doc_id}/embed")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["document_id"] == doc_id
    assert body["total_embeddings_generated"] > 0
    assert body["dimension"] == 384
    assert "MiniLM" in body["embedding_model"] or "all-" in body["embedding_model"]


def test_api_embed_document_not_found(client_with_doc):
    """POST /embed with unknown document ID must return 404."""
    client, _ = client_with_doc
    response = client.post("/api/v1/knowledge/documents/doc_nonexistent/embed")
    assert response.status_code == 404


def test_api_embedding_status_before_embedding(client_with_doc):
    """GET /embeddings/status before embedding must report embeddings_generated=False."""
    client, doc_id = client_with_doc
    response = client.get(f"/api/v1/knowledge/documents/{doc_id}/embeddings/status")
    assert response.status_code == 200
    body = response.json()
    assert body["embeddings_generated"] is False
    assert body["total_embeddings"] == 0


def test_api_embedding_status_after_embedding(client_with_doc):
    """GET /embeddings/status after embedding must report embeddings_generated=True."""
    client, doc_id = client_with_doc
    # Generate embeddings first
    embed_resp = client.post(f"/api/v1/knowledge/documents/{doc_id}/embed")
    assert embed_resp.status_code == 200

    response = client.get(f"/api/v1/knowledge/documents/{doc_id}/embeddings/status")
    assert response.status_code == 200
    body = response.json()
    assert body["embeddings_generated"] is True
    assert body["total_embeddings"] > 0
    assert body["dimension"] == 384


def test_api_embedding_status_not_found(client_with_doc):
    """GET /embeddings/status with unknown document ID must return 404."""
    client, _ = client_with_doc
    response = client.get("/api/v1/knowledge/documents/doc_ghost/embeddings/status")
    assert response.status_code == 404


def test_api_embed_query_success(client_with_doc):
    """POST /embed-query must return 200 with correct dimension and preview."""
    client, _ = client_with_doc
    response = client.post(
        "/api/v1/knowledge/embed-query",
        json={"query": "What caused the memory leak in the payment service?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["dimension"] == 384
    assert len(body["embedding_preview"]) == 5
    assert isinstance(body["vector_norm"], float)
    assert abs(body["vector_norm"] - 1.0) < 1e-3  # normalised


def test_api_embed_query_empty_string(client_with_doc):
    """POST /embed-query with empty query must return 400."""
    client, _ = client_with_doc
    response = client.post(
        "/api/v1/knowledge/embed-query",
        json={"query": ""},
    )
    assert response.status_code == 400


def test_api_embed_query_whitespace_only(client_with_doc):
    """POST /embed-query with whitespace-only query must return 400."""
    client, _ = client_with_doc
    response = client.post(
        "/api/v1/knowledge/embed-query",
        json={"query": "   "},
    )
    assert response.status_code == 400


# ===========================================================================
# Stage 4 — FAISS Vector Database Tests
# ===========================================================================

from app.knowledge.vector_store import FaissVectorStore
from app.knowledge.exceptions import (
    CorruptedIndexException,
    DimensionMismatchException,
    DuplicateDocumentException,
    VectorStoreException,
)
from app.schemas.knowledge import (
    DocumentIndexInfo,
    IndexDocumentResponse,
    VectorMetadata,
    VectorStoreActionResponse,
    VectorStoreStatsResponse,
    VectorStoreStatusResponse,
)


def _make_mock_chunks_and_embeddings(doc_id: str, count: int = 2, dimension: int = 384):
    """Helper to create matching DocumentChunks and ChunkEmbeddings."""
    chunks = [
        DocumentChunk(
            chunk_id=f"{doc_id}_c{i:04d}",
            document_id=doc_id,
            filename=f"{doc_id}.txt",
            chunk_index=i,
            text=f"Incident runbook section {i} for {doc_id}. Check log traces and system metrics.",
            character_count=70,
            word_count=12,
            metadata={"source": f"{doc_id}.txt", "priority": "high"},
        )
        for i in range(count)
    ]
    # Unit vectors
    embeddings = []
    for i in range(count):
        raw = [0.0] * dimension
        raw[i % dimension] = 1.0  # simple orthogonal unit vector
        embeddings.append(
            ChunkEmbedding(
                chunk_id=f"{doc_id}_c{i:04d}",
                document_id=doc_id,
                chunk_index=i,
                embedding=raw,
                dimension=dimension,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            )
        )
    return chunks, embeddings


def test_faiss_vector_store_initialization(tmp_path):
    """FaissVectorStore must initialize an empty IndexIDMap2 index."""
    store = FaissVectorStore(storage_dir=tmp_path / "faiss", auto_load=False)
    assert store.total_vectors == 0
    assert store.total_documents == 0
    assert store.dimension == 384
    assert not store.is_persisted()


def test_faiss_vector_store_add_vectors_and_metadata(tmp_path):
    """add_document_vectors() must insert vectors and preserve chunk metadata."""
    store = FaissVectorStore(storage_dir=tmp_path / "faiss", auto_load=False)
    chunks, embeddings = _make_mock_chunks_and_embeddings("doc_test_1", count=3)

    resp = store.add_document_vectors("doc_test_1", chunks, embeddings)
    assert resp.status == "success"
    assert resp.vectors_added == 3
    assert resp.total_vectors_in_index == 3
    assert store.total_vectors == 3
    assert store.total_documents == 1
    assert store.is_document_indexed("doc_test_1")

    # Verify metadata preservation
    info = store.get_document_info("doc_test_1")
    assert info is not None
    assert info.filename == "doc_test_1.txt"
    assert info.total_chunks == 3
    assert len(info.vector_ids) == 3


def test_faiss_vector_store_persistence_and_reload(tmp_path):
    """Index and metadata must persist to disk and reload accurately."""
    storage_dir = tmp_path / "faiss"
    store = FaissVectorStore(storage_dir=storage_dir, auto_load=False)
    chunks, embeddings = _make_mock_chunks_and_embeddings("doc_persist_1", count=2)
    store.add_document_vectors("doc_persist_1", chunks, embeddings, auto_save=True)

    assert store.is_persisted()
    assert (storage_dir / "index.faiss").exists()
    assert (storage_dir / "metadata.json").exists()

    # Simulate fresh process startup / restart
    reloaded_store = FaissVectorStore(storage_dir=storage_dir, auto_load=True)
    assert reloaded_store.total_vectors == 2
    assert reloaded_store.total_documents == 1
    assert reloaded_store.is_document_indexed("doc_persist_1")
    doc_info = reloaded_store.get_document_info("doc_persist_1")
    assert doc_info.total_chunks == 2


def test_faiss_vector_store_duplicate_prevention(tmp_path):
    """Adding a duplicate document without overwrite=True must raise DuplicateDocumentException."""
    store = FaissVectorStore(storage_dir=tmp_path / "faiss", auto_load=False)
    chunks, embeddings = _make_mock_chunks_and_embeddings("doc_dup_1", count=2)
    store.add_document_vectors("doc_dup_1", chunks, embeddings)

    with pytest.raises(DuplicateDocumentException, match="already indexed"):
        store.add_document_vectors("doc_dup_1", chunks, embeddings, overwrite=False)


def test_faiss_vector_store_overwrite(tmp_path):
    """Adding an existing document with overwrite=True must replace old vectors."""
    store = FaissVectorStore(storage_dir=tmp_path / "faiss", auto_load=False)
    chunks1, embeddings1 = _make_mock_chunks_and_embeddings("doc_replace_1", count=2)
    store.add_document_vectors("doc_replace_1", chunks1, embeddings1)
    assert store.total_vectors == 2

    chunks2, embeddings2 = _make_mock_chunks_and_embeddings("doc_replace_1", count=4)
    resp = store.add_document_vectors("doc_replace_1", chunks2, embeddings2, overwrite=True)
    assert resp.vectors_added == 4
    assert store.total_vectors == 4
    assert store.total_documents == 1


def test_faiss_vector_store_remove_document(tmp_path):
    """remove_document_vectors() must delete vector IDs from index and metadata."""
    store = FaissVectorStore(storage_dir=tmp_path / "faiss", auto_load=False)
    c1, e1 = _make_mock_chunks_and_embeddings("doc_del_1", count=3)
    c2, e2 = _make_mock_chunks_and_embeddings("doc_del_2", count=2)
    store.add_document_vectors("doc_del_1", c1, e1)
    store.add_document_vectors("doc_del_2", c2, e2)
    assert store.total_vectors == 5
    assert store.total_documents == 2

    removed = store.remove_document_vectors("doc_del_1")
    assert removed == 3
    assert store.total_vectors == 2
    assert store.total_documents == 1
    assert not store.is_document_indexed("doc_del_1")
    assert store.is_document_indexed("doc_del_2")


def test_faiss_vector_store_remove_nonexistent_raises(tmp_path):
    """remove_document_vectors() with unknown document must raise DocumentNotFoundException."""
    store = FaissVectorStore(storage_dir=tmp_path / "faiss", auto_load=False)
    with pytest.raises(DocumentNotFoundException):
        store.remove_document_vectors("doc_unknown")


def test_faiss_vector_store_dimension_mismatch(tmp_path):
    """Vectors with mismatched dimension must raise DimensionMismatchException."""
    store = FaissVectorStore(storage_dir=tmp_path / "faiss", auto_load=False)
    c, e = _make_mock_chunks_and_embeddings("doc_dim_err", count=2, dimension=128)
    with pytest.raises(DimensionMismatchException):
        store.add_document_vectors("doc_dim_err", c, e)


def test_faiss_vector_store_empty_inputs(tmp_path):
    """Empty chunks or empty embeddings must raise VectorStoreException."""
    store = FaissVectorStore(storage_dir=tmp_path / "faiss", auto_load=False)
    with pytest.raises(VectorStoreException):
        store.add_document_vectors("doc_empty", [], [])


def test_faiss_vector_store_corrupted_index_handling(tmp_path):
    """Corrupted metadata or index must raise CorruptedIndexException on load."""
    storage_dir = tmp_path / "faiss"
    storage_dir.mkdir(parents=True, exist_ok=True)
    # Write garbage into index files
    (storage_dir / "index.faiss").write_bytes(b"garbage not a faiss index")
    (storage_dir / "metadata.json").write_text("invalid json {", encoding="utf-8")

    store = FaissVectorStore(storage_dir=storage_dir, auto_load=False)
    with pytest.raises(CorruptedIndexException):
        store.load_index()


def test_faiss_vector_store_clear(tmp_path):
    """clear() must empty index and delete files."""
    storage_dir = tmp_path / "faiss"
    store = FaissVectorStore(storage_dir=storage_dir, auto_load=False)
    c, e = _make_mock_chunks_and_embeddings("doc_clear", count=2)
    store.add_document_vectors("doc_clear", c, e, auto_save=True)
    assert store.is_persisted()

    store.clear()
    assert store.total_vectors == 0
    assert store.total_documents == 0
    assert not (storage_dir / "index.faiss").exists()


# ---------------------------------------------------------------------------
# API Tests for Stage 4
# ---------------------------------------------------------------------------

def test_api_vectorstore_status(client_with_doc):
    """GET /api/v1/knowledge/vectorstore/status must return 200 with vector store health."""
    client, _ = client_with_doc
    response = client.get("/api/v1/knowledge/vectorstore/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["initialized"] is True
    assert body["dimension"] == 384
    assert "MiniLM" in body["embedding_model"] or "all-" in body["embedding_model"]


def test_api_vectorstore_stats(client_with_doc):
    """GET /api/v1/knowledge/vectorstore/stats must return 200 with inventory."""
    client, _ = client_with_doc
    response = client.get("/api/v1/knowledge/vectorstore/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "total_vectors" in body
    assert "indexed_documents" in body


def test_api_index_document_success(client_with_doc):
    """POST /documents/{id}/index must add vectors and return 200."""
    client, doc_id = client_with_doc
    response = client.post(f"/api/v1/knowledge/documents/{doc_id}/index")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["document_id"] == doc_id
    assert body["vectors_added"] > 0
    assert body["total_vectors_in_index"] > 0


def test_api_index_document_duplicate_returns_409(client_with_doc):
    """Indexing the same document twice with overwrite=False must return 409 Conflict."""
    client, doc_id = client_with_doc
    # First index
    resp1 = client.post(f"/api/v1/knowledge/documents/{doc_id}/index")
    assert resp1.status_code == 200

    # Second index without overwrite
    resp2 = client.post(f"/api/v1/knowledge/documents/{doc_id}/index?overwrite=false")
    assert resp2.status_code == 409
    assert "already indexed" in resp2.json()["detail"]


def test_api_index_document_overwrite_success(client_with_doc):
    """Indexing the same document with overwrite=True must return 200."""
    client, doc_id = client_with_doc
    resp1 = client.post(f"/api/v1/knowledge/documents/{doc_id}/index")
    assert resp1.status_code == 200

    resp2 = client.post(f"/api/v1/knowledge/documents/{doc_id}/index?overwrite=true")
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "success"


def test_api_index_document_not_found(client_with_doc):
    """Indexing a non-existent document must return 404."""
    client, _ = client_with_doc
    response = client.post("/api/v1/knowledge/documents/doc_ghost_xyz/index")
    assert response.status_code == 404


def test_api_vectorstore_save_and_load(client_with_doc):
    """POST /vectorstore/save and POST /vectorstore/load must return 200."""
    client, doc_id = client_with_doc
    # Index document first
    client.post(f"/api/v1/knowledge/documents/{doc_id}/index")

    save_resp = client.post("/api/v1/knowledge/vectorstore/save")
    assert save_resp.status_code == 200
    assert save_resp.json()["status"] == "success"

    load_resp = client.post("/api/v1/knowledge/vectorstore/load")
    assert load_resp.status_code == 200
    assert load_resp.json()["status"] == "success"


def test_api_remove_document_from_vector_store(client_with_doc):
    """DELETE /vectorstore/documents/{id} removes vectors; subsequent call returns 404."""
    client, doc_id = client_with_doc
    # Index document
    client.post(f"/api/v1/knowledge/documents/{doc_id}/index")

    # Remove from vector store
    del_resp = client.delete(f"/api/v1/knowledge/vectorstore/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "success"

    # Remove again -> 404
    del_resp2 = client.delete(f"/api/v1/knowledge/vectorstore/documents/{doc_id}")
    assert del_resp2.status_code == 404


def test_api_delete_document_syncs_with_vector_store(client_with_doc):
    """DELETE /documents/{id} must also remove vectors from vector store."""
    client, doc_id = client_with_doc
    # Index document
    client.post(f"/api/v1/knowledge/documents/{doc_id}/index")

    # Verify indexed in stats
    stats1 = client.get("/api/v1/knowledge/vectorstore/stats").json()
    assert stats1["total_documents"] >= 1

    # Delete full document
    del_resp = client.delete(f"/api/v1/knowledge/documents/{doc_id}")
    assert del_resp.status_code == 200

    # Verify vector store is updated
    stats2 = client.get("/api/v1/knowledge/vectorstore/stats").json()
    assert not any(d["document_id"] == doc_id for d in stats2["indexed_documents"])
