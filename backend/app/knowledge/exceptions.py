"""Typed exception hierarchy for the Knowledge Base ingestion pipeline."""


class KnowledgeBaseException(Exception):
    """Base exception for all Knowledge Base errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidFileTypeException(KnowledgeBaseException):
    """Raised when an unsupported or spoofed file format is provided."""
    pass


class FileSizeExceededException(KnowledgeBaseException):
    """Raised when uploaded file exceeds configured size boundary."""
    pass


class CorruptedDocumentException(KnowledgeBaseException):
    """Raised when document content is corrupted or unparseable."""
    pass


class DocumentNotFoundException(KnowledgeBaseException):
    """Raised when a specified document cannot be found."""
    pass


class VectorStoreException(KnowledgeBaseException):
    """Base exception for all FAISS vector store operations."""
    pass


class IndexNotInitializedException(VectorStoreException):
    """Raised when index operations are attempted on an uninitialized index."""
    pass


class DimensionMismatchException(VectorStoreException):
    """Raised when vector dimension does not match index dimension."""
    pass


class DuplicateDocumentException(VectorStoreException):
    """Raised when attempting to add a document that is already indexed."""
    pass


class CorruptedIndexException(VectorStoreException):
    """Raised when a persisted index or metadata file is corrupted or incompatible."""
    pass


