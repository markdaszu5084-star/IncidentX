"""Health and welcome endpoints."""

from fastapi import APIRouter, status
from app.config import get_settings
from app.schemas.health import HealthResponse, RootResponse

router = APIRouter(tags=["System"])


@router.get(
    "/",
    response_model=RootResponse,
    status_code=status.HTTP_200_OK,
    summary="Root Welcome Endpoint",
    description="Provides basic service identification, status, and API documentation reference.",
)
async def get_root() -> RootResponse:
    """Return welcome message and API metadata."""
    settings = get_settings()
    return RootResponse(
        name=settings.app_name,
        version=settings.app_version,
        message="Welcome to IncidentX - AI-Powered Autonomous IT Incident Response Agent API",
        docs_url="/docs",
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Backend Health Check",
    description="Reports the backend operational status, active environment, and current UTC timestamp.",
)
async def get_health() -> HealthResponse:
    """Return current system health status."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )

