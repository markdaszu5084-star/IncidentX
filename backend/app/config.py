"""Application Configuration Settings."""

from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """IncidentX Application Settings."""

    app_name: str = "IncidentX API"
    app_version: str = "0.1.0"
    app_description: str = (
        "AI-powered Autonomous IT Incident Response Agent backend service."
    )
    environment: str = "development"
    debug: bool = False

    # Server binding
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS configuration
    # Permitting the React frontend at localhost:5173
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # Security & upload constraints
    max_log_file_size_bytes: int = 10 * 1024 * 1024  # 10MB limit
    max_doc_file_size_bytes: int = 15 * 1024 * 1024  # 15MB limit for PDF/TXT docs

    # Knowledge Base & Vector Storage
    knowledge_base_dir: str = "knowledge_base"
    faiss_index_dir: str = "knowledge_base/faiss"
    openai_api_key: Optional[str] = None

    # Embedding configuration
    # Override via EMBEDDING_MODEL env var in .env to switch models.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # all-MiniLM-L6-v2 produces 384-dimensional unit vectors.
    embedding_dimension: int = 384
    embedding_batch_size: int = 32
    embeddings_dir: str = "knowledge_base/embeddings"

    # Chunking defaults
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 150

    # Semantic Retrieval defaults
    default_top_k: int = 5
    default_score_threshold: float = 0.0
    max_context_characters: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached instance of application settings."""
    return Settings()

