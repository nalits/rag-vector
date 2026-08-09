"""Loading documents into langchain Documents based on file type."""

import importlib
import logging
from pathlib import Path
import tempfile

import docx2txt
from langchain_core.documents import Document

from rag.config import DocumentExtension, Settings
from rag.validation.validator import UnsupportedDocumentTypeError, document_extension

logger = logging.getLogger(__name__)


def load_document(key: str, content: bytes, settings: Settings) -> list[Document]:
    """Load the raw content of a document into langchain Documents.

    Content is written to a temporary file so binary formats can be parsed.

    Args:
        key: S3 object key of the document, used to derive the file type.
        content: Raw document content.
        settings: Pipeline settings, including text encoding.

    Raises:
        UnsupportedDocumentTypeError: If the document type is not supported.

    Returns:
        The loaded documents.
    """
    extension = document_extension(key)
    logger.info("load document key=%s extension=%s bytes=%s", key, extension, len(content))
    with tempfile.NamedTemporaryFile(suffix=extension.value) as file:
        file.write(content)
        file.flush()
        documents = _load(file.name, extension, settings)
    logger.info(
        "load document done key=%s documents=%s chars=%s",
        key,
        len(documents),
        sum(len(document.page_content) for document in documents),
    )
    return documents


def _load(path: str, extension: DocumentExtension, settings: Settings) -> list[Document]:
    match extension:
        case DocumentExtension.DOCX:
            return _load_docx(path)
        case DocumentExtension.DOC:
            return _load_doc(path)
        case DocumentExtension.TXT | DocumentExtension.MARKDOWN:
            return _load_text(path, settings.text_encoding)
        case _ as unreachable:
            raise NotImplementedError(unreachable)


def _load_text(path: str, encoding: str) -> list[Document]:
    return [Document(page_content=Path(path).read_text(encoding=encoding))]


def _load_docx(path: str) -> list[Document]:
    return [Document(page_content=docx2txt.process(path) or "")]


def _load_doc(path: str) -> list[Document]:
    elements = _partition_doc(path)
    return [Document(page_content="\n".join(str(element) for element in elements))]


def _partition_doc(path: str) -> list[object]:
    """Parse a legacy ``.doc`` file. Optional; omitted from the Lambda layer."""
    try:
        module = importlib.import_module("unstructured.partition.doc")
    except ImportError as exc:
        raise UnsupportedDocumentTypeError(
            "legacy .doc files require the unstructured package"
        ) from exc
    return list(module.partition_doc(filename=path))
