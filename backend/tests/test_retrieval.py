"""
Unit and integration tests for Semantic Retrieval Engine (Stage 5).
"""

import pytest
from fastapi.testclient import TestClient

from app.api.retrieval import service as api_retrieval_service
import app.api.retrieval as retrieval_router_module
import app.api.knowledge as knowledge_router_module
from app.knowledge.embeddings import get_embedding_service
from app.knowledge.retrieval import SemanticRetrievalEngine
from app.knowledge.service import KnowledgeDocumentService
from app.knowledge.storage import DocumentStorageManager
from app.knowledge.vector_store import FaissVectorStore
from app.main import app
from app.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    SearchResultItem,
    SourceReference,
)


@pytest.fixture(scope="module")
def embedding_svc():
    """Module-level cached embedding service."""
    return get_embedding_service()


@pytest.fixture()
def seeded_retrieval_engine(tmp_path, embedding_svc):
    """Create an isolated SemanticRetrievalEngine populated with known incident runbooks."""
    storage_dir = tmp_path / "retrieval_kb"
    storage = DocumentStorageManager(base_dir=str(storage_dir))
    vector_store = FaissVectorStore(
        storage_dir=storage.faiss_dir,
        embedding_service=embedding_svc,
        auto_load=False,
    )
    doc_service = KnowledgeDocumentService(
        storage_manager=storage,
        vector_store=vector_store,
        embedding_service=embedding_svc,
    )

    # 1. Document: Database CPU & Memory Runbook
    db_text = (
        "Runbook: Database Performance Degradation.\n\n"
        "When PostgreSQL CPU utilization exceeds 90%, check pg_stat_activity for long-running transactions.\n\n"
        "Terminate orphaned client connections using pg_terminate_backend. Verify shared_buffers allocation."
    )
    db_resp = doc_service.process_and_store_document("db_runbook.txt", db_text.encode("utf-8"))
    doc_service.index_document(db_resp.document.document_id)

    # 2. Document: Payment Gateway 502 Bad Gateway Outage
    payment_text = (
        "Incident Postmortem: Payment Gateway 502 Bad Gateway.\n\n"
        "Upstream Stripe webhook endpoints timed out due to DNS resolution latency on AWS Route53.\n\n"
        "Resolution: Switched to static fallback IP endpoints and increased HTTP client timeout from 5s to 15s."
    )
    payment_resp = doc_service.process_and_store_document("payment_gateway.txt", payment_text.encode("utf-8"))
    doc_service.index_document(payment_resp.document.document_id)

    # 3. Document: Kubernetes OOMKilled Container Triage
    k8s_text = (
        "Kubernetes Triage: Container OOMKilled Exit Code 137.\n\n"
        "Pod payment-service exceeded memory limit of 512MiB. Java JVM heap was unconstrained.\n\n"
        "Remediation: Set -XX:MaxRAMPercentage=75.0 and bumped pod memory limit to 1Gi in deployment manifest."
    )
    k8s_resp = doc_service.process_and_store_document("k8s_oom_triage.txt", k8s_text.encode("utf-8"))
    doc_service.index_document(k8s_resp.document.document_id)

    return doc_service, {
        "db_doc_id": db_resp.document.document_id,
        "payment_doc_id": payment_resp.document.document_id,
        "k8s_doc_id": k8s_resp.document.document_id,
    }


# ===========================================================================
# 1. Query Normalization & Validation Tests
# ===========================================================================

def test_query_normalization_strips_whitespace():
    """normalize_query must trim whitespace and consolidate internal spaces."""
    cleaned = SemanticRetrievalEngine.normalize_query("   database   cpu   spike   ")
    assert cleaned == "database cpu spike"


def test_query_normalization_empty_raises():
    """Empty query string must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        SemanticRetrievalEngine.normalize_query("")


def test_query_normalization_whitespace_raises():
    """Whitespace-only query string must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        SemanticRetrievalEngine.normalize_query("     \n\t  ")


def test_query_normalization_overly_long_raises():
    """Query exceeding 4,000 characters must raise ValueError."""
    with pytest.raises(ValueError, match="4,000"):
        SemanticRetrievalEngine.normalize_query("word " * 1000)


# ===========================================================================
# 2. Semantic Search & Ranking Tests
# ===========================================================================

def test_semantic_search_relevant_query(seeded_retrieval_engine):
    """Query about database performance must rank database runbook as top match."""
    doc_service, doc_ids = seeded_retrieval_engine
    req = RetrievalRequest(query="How to fix high CPU usage in PostgreSQL?", top_k=3)
    resp = doc_service.search(req)

    assert resp.status == "success"
    assert resp.total_results > 0
    top_match = resp.results[0]
    assert top_match.document_id == doc_ids["db_doc_id"]
    assert "pg_stat_activity" in top_match.text or "PostgreSQL" in top_match.text
    assert top_match.similarity_score > 0.4


def test_semantic_search_payment_gateway_query(seeded_retrieval_engine):
    """Query about payment gateway 502 must match the payment incident document."""
    doc_service, doc_ids = seeded_retrieval_engine
    req = RetrievalRequest(query="Stripe webhook 502 timeout errors", top_k=3)
    resp = doc_service.search(req)

    assert resp.total_results > 0
    top_match = resp.results[0]
    assert top_match.document_id == doc_ids["payment_doc_id"]
    assert "Stripe" in top_match.text


def test_semantic_search_results_ranked_descending(seeded_retrieval_engine):
    """Results must be ordered monotonically descending by similarity score."""
    doc_service, _ = seeded_retrieval_engine
    req = RetrievalRequest(query="container crash memory allocation failure", top_k=5)
    resp = doc_service.search(req)

    scores = [r.similarity_score for r in resp.results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"Ranking violated: {scores[i]} < {scores[i+1]}"


def test_semantic_search_top_k_limit(seeded_retrieval_engine):
    """Search must respect the top_k parameter boundary."""
    doc_service, _ = seeded_retrieval_engine
    req = RetrievalRequest(query="service error", top_k=1)
    resp = doc_service.search(req)
    assert len(resp.results) <= 1


def test_semantic_search_score_threshold_filtering(seeded_retrieval_engine):
    """Chunks below score_threshold must be filtered out."""
    doc_service, _ = seeded_retrieval_engine
    # Very high threshold that no result should satisfy
    req = RetrievalRequest(query="high database cpu", score_threshold=0.999)
    resp = doc_service.search(req)
    assert resp.total_results == 0
    assert len(resp.results) == 0
    assert resp.context_text == ""


def test_semantic_search_document_id_filter(seeded_retrieval_engine):
    """document_id filter must restrict retrieval to only the specified document."""
    doc_service, doc_ids = seeded_retrieval_engine
    req = RetrievalRequest(
        query="memory timeout",
        document_id=doc_ids["k8s_doc_id"],
        top_k=5,
    )
    resp = doc_service.search(req)
    for item in resp.results:
        assert item.document_id == doc_ids["k8s_doc_id"]


def test_semantic_search_empty_vector_store(tmp_path, embedding_svc):
    """Searching an empty vector store must return 0 results gracefully without crashing."""
    storage_dir = tmp_path / "empty_kb"
    storage = DocumentStorageManager(base_dir=str(storage_dir))
    vector_store = FaissVectorStore(storage_dir=storage.faiss_dir, embedding_service=embedding_svc, auto_load=False)
    engine = SemanticRetrievalEngine(vector_store=vector_store, embedding_service=embedding_svc)

    req = RetrievalRequest(query="Why did the server crash?")
    resp = engine.search(req)

    assert resp.status == "success"
    assert resp.total_results == 0
    assert resp.results == []
    assert resp.context_text == ""


# ===========================================================================
# 3. Context Preparation Tests
# ===========================================================================

def test_context_preparation_includes_provenance_headers(seeded_retrieval_engine):
    """Prepared context must include provenance headers and source references."""
    doc_service, _ = seeded_retrieval_engine
    req = RetrievalRequest(query="PostgreSQL high load", top_k=2)
    resp = doc_service.search(req)

    assert "--- SOURCE:" in resp.context_text
    assert resp.total_context_characters == len(resp.context_text)
    assert len(resp.sources) > 0
    for s in resp.sources:
        assert isinstance(s, SourceReference)
        assert s.filename
        assert s.chunk_id


def test_context_preparation_character_limit_budget():
    """Context preparation must not exceed max_characters budget."""
    engine = SemanticRetrievalEngine(vector_store=None, embedding_service=None)
    items = [
        SearchResultItem(
            chunk_id=f"c{i}",
            document_id=f"doc_{i}",
            filename=f"doc_{i}.txt",
            chunk_index=i,
            text="A" * 500,
            similarity_score=0.9 - (i * 0.1),
            character_count=500,
            word_count=1,
            metadata={},
        )
        for i in range(5)
    ]
    # Restrict budget to 1200 characters
    context, sources = engine.prepare_context(items, max_characters=1200)
    assert len(context) <= 1400  # including header overhead
    assert len(sources) < 5  # stopped before adding all 5 items


def test_context_preparation_deduplication():
    """Duplicate chunk_ids must be excluded from context preparation."""
    engine = SemanticRetrievalEngine(vector_store=None, embedding_service=None)
    items = [
        SearchResultItem(
            chunk_id="c_duplicate",
            document_id="doc_1",
            filename="doc_1.txt",
            chunk_index=0,
            text="Duplicate chunk text content.",
            similarity_score=0.88,
            character_count=30,
            word_count=4,
            metadata={},
        ),
        SearchResultItem(
            chunk_id="c_duplicate",
            document_id="doc_1",
            filename="doc_1.txt",
            chunk_index=0,
            text="Duplicate chunk text content.",
            similarity_score=0.88,
            character_count=30,
            word_count=4,
            metadata={},
        ),
    ]
    context, sources = engine.prepare_context(items, max_characters=5000)
    assert len(sources) == 1
    assert context.count("--- SOURCE:") == 1


# ===========================================================================
# 4. API Endpoint Integration Tests (POST /api/v1/retrieval/search)
# ===========================================================================

@pytest.fixture()
def client_with_seeded_data(seeded_retrieval_engine, monkeypatch):
    """Configure TestClient with seeded knowledge service."""
    doc_service, _ = seeded_retrieval_engine
    monkeypatch.setattr(retrieval_router_module, "service", doc_service)
    monkeypatch.setattr(knowledge_router_module, "service", doc_service)
    return TestClient(app)


def test_api_retrieval_search_success(client_with_seeded_data):
    """POST /api/v1/retrieval/search must return 200 with structured response."""
    client = client_with_seeded_data
    payload = {
        "query": "How to resolve container exit code 137 OOMKilled?",
        "top_k": 3,
        "score_threshold": 0.1,
    }
    response = client.post("/api/v1/retrieval/search", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["status"] == "success"
    assert data["total_results"] > 0
    assert len(data["results"]) <= 3
    assert data["top_k"] == 3
    assert "context_text" in data
    assert "sources" in data
    assert len(data["sources"]) > 0


def test_api_retrieval_search_empty_query_rejected(client_with_seeded_data):
    """POST /api/v1/retrieval/search with blank query must return 422 or 400."""
    client = client_with_seeded_data
    payload = {"query": "    "}
    response = client.post("/api/v1/retrieval/search", json=payload)
    assert response.status_code in (400, 422)


def test_api_retrieval_search_alias_endpoint(client_with_seeded_data):
    """POST /api/v1/knowledge/search alias endpoint must also return 200."""
    client = client_with_seeded_data
    payload = {
        "query": "PostgreSQL cpu spike triage",
        "top_k": 2,
    }
    response = client.post("/api/v1/knowledge/search", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["total_results"] > 0

