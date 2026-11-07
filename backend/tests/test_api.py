"""Unit and integration tests for FastAPI application endpoints."""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure backend directory is in sys.path for test execution
backend_path = Path(__file__).resolve().parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.main import app
from app.schemas.health import HealthResponse, RootResponse


@pytest.fixture
def client() -> TestClient:
    """Fixture providing a FastAPI TestClient."""
    return TestClient(app)


def test_root_endpoint_success(client: TestClient):
    """Test GET / returns 200 and matches RootResponse schema."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    model = RootResponse(**data)
    assert model.name == "IncidentX API"
    assert "IncidentX" in model.message
    assert model.docs_url == "/docs"


def test_health_endpoint_success(client: TestClient):
    """Test GET /health returns 200 and matches HealthResponse schema."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    model = HealthResponse(**data)
    assert model.status == "healthy"
    assert model.app == "IncidentX API"
    assert model.environment == "development"
    assert model.timestamp is not None


def test_cors_headers_for_react_origin(client: TestClient):
    """Test CORS headers when called from React frontend (http://localhost:5173)."""
    # Simple GET request with Origin header
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert response.headers.get("access-control-allow-credentials") == "true"

    # Preflight OPTIONS request
    preflight = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "GET" in preflight.headers.get("access-control-allow-methods", "")


def test_swagger_documentation_accessible(client: TestClient):
    """Test GET /docs returns 200 and serves Swagger UI HTML."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "swagger-ui" in response.text.lower()


def test_openapi_json_schema(client: TestClient):
    """Test GET /openapi.json returns valid OpenAPI specification."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "paths" in data
    assert "/" in data["paths"]
    assert "/health" in data["paths"]

