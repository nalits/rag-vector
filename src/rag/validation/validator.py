"""Validation rules for documents uploaded to S3."""

import logging

from rag.config import DocumentExtension, Settings, document_suffix

logger = logging.getLogger(__name__)


class UnsupportedDocumentTypeError(RuntimeError):
    """Error generated when the document type is not supported."""


class DocumentTooLargeError(RuntimeError):
    """Error generated when the document exceeds the maximum size."""


def document_extension(key: str) -> DocumentExtension:
    """Return the supported extension for an object key.

    Args:
        key: S3 object key of the document.

    Raises:
        UnsupportedDocumentTypeError: If the extension is not allowed.

    Returns:
        The matching document extension.
    """
    try:
        return DocumentExtension(document_suffix(key))
    except ValueError:
        raise UnsupportedDocumentTypeError(f"unsupported document type: {key}") from None


def validate_document(key: str, size: int, settings: Settings) -> None:
    """Validate that a document is supported and within the size limit.

    Args:
        key: S3 object key of the document.
        size: Size of the document in bytes.
        settings: Pipeline settings that define the size limit.

    Raises:
        UnsupportedDocumentTypeError: If the extension is not allowed.
        DocumentTooLargeError: If the document exceeds the maximum size.
    """
    extension = document_extension(key)
    logger.info(
        "validate document key=%s extension=%s size_bytes=%s max_bytes=%s",
        key,
        extension,
        size,
        settings.max_file_size_bytes,
    )
    if size > settings.max_file_size_bytes:
        raise DocumentTooLargeError(f"document exceeds size limit: {key} ({size} bytes)")
