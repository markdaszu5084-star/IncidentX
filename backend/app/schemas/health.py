"""Pydantic models for root and health check endpoints."""

from datetime import datetime, timezone
from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    """Response model for root welcome endpoint (GET /)."""

    name: str = Field(description="Application name")
    version: str = Field(description="Application version")
    message: str = Field(description="Welcome message and overview")
    docs_url: str = Field(description="URL path for Swagger OpenAPI documentation")


class HealthResponse(BaseModel):
    """Response model for health check endpoint (GET /health)."""

    status: str = Field(default="healthy", description="Current health status")
    app: str = Field(description="Application name")
    version: str = Field(description="Application version")
    environment: str = Field(description="Deployment environment")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the health check inspection",
    )

