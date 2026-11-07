"""API router package."""
from app.api.health import router as health_router
from app.api.knowledge import router as knowledge_router

__all__ = ["health_router", "knowledge_router"]

