"""Orchestrates the RAG ingestion pipeline."""

import logging
from typing import Protocol

from langchain_core.documents import Document

from rag.chunker.splitter import ChunkSplitter
from rag.config import Settings, document_suffix
from rag.embedder.embeddings import EmbeddingModel
from rag.loader.document_loader import load_document
from rag.s3.document_store import DocumentStore, s3_uri
from rag.validation.validator import validate_document
from rag.vector_store.s3_vector_store import S3VectorStore

logger = logging.getLogger(__name__)


class SupportsDocumentStore(Protocol):
    """Reads object size and content from object storage."""

    def size(self, bucket: str, key: str) -> int:
        """Return the object size in bytes."""

    def download(self, bucket: str, key: str) -> bytes:
        """Return the object content."""


class SupportsChunkSplitter(Protocol):
    """Splits loaded documents into chunks."""

    def split(self, documents: list[Document]) -> list[Document]:
        """Return overlapping chunks for the given documents."""


class SupportsEmbeddingModel(Protocol):
    """Converts chunk text into embedding vectors."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input text."""


class SupportsVectorStore(Protocol):
    """Persists embeddings in a vector index."""

    def store_vectors(
        self,
        chunks: list[Document],
        embeddings: list[list[float]],
        source: str,
    ) -> None:
        """Write chunk embeddings and metadata to the index."""


class RagPipeline:
    """Runs the document ingestion pipeline end to end.

    The pipeline steps are:

    1. Read the document from S3.
    2. Validate the document type and size.
    3. Load the document into text.
    4. Split the text into chunks.
    5. Embed the chunks into vectors.
    6. Store the vectors in the S3 vector index.
    """

    def __init__(
        self,
        document_store: SupportsDocumentStore,
        splitter: SupportsChunkSplitter,
        embeddings: SupportsEmbeddingModel,
        vector_store: SupportsVectorStore,
        settings: Settings,
    ) -> None:
        self._document_store = document_store
        self._splitter = splitter
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._settings = settings

    def process(self, bucket: str, key: str) -> int:
        """Ingest a document from S3 and store its embeddings.

        Args:
            bucket: S3 bucket containing the document.
            key: S3 object key of the document.

        Returns:
            The number of chunks stored.
        """
        location = s3_uri(bucket, key)
        logger.info(
            "pipeline start %s suffix=%s vector_bucket=%s vector_index=%s "
            "chunk_size=%s chunk_overlap=%s",
            location,
            document_suffix(key),
            self._settings.s3_vector_bucket,
            self._settings.s3_vector_index,
            self._settings.chunk_size,
            self._settings.chunk_overlap,
        )
        size = self._document_store.size(bucket, key)
        logger.info("object size %s bytes=%s", location, size)
        validate_document(key, size, self._settings)
        logger.info("validation ok %s max_bytes=%s", location, self._settings.max_file_size_bytes)
        content = self._document_store.download(bucket, key)
        logger.info("download complete %s bytes=%s", location, len(content))
        documents = load_document(key, content, self._settings)
        logger.info("loaded documents=%s %s", len(documents), location)
        raw_chunks = self._splitter.split(documents)
        chunks = [chunk for chunk in raw_chunks if chunk.page_content.strip()]
        blank_skipped = len(raw_chunks) - len(chunks)
        logger.info(
            "chunked %s total=%s non_empty=%s blank_skipped=%s",
            location,
            len(raw_chunks),
            len(chunks),
            blank_skipped,
        )
        if not chunks:
            logger.warning(
                "no non-empty chunks; skipping embeddings and vector store %s",
                location,
            )
            return 0
        embeddings = self._embeddings.embed_documents([chunk.page_content for chunk in chunks])
        logger.info(
            "embeddings ready %s count=%s dimensions=%s",
            location,
            len(embeddings),
            self._settings.embedding_dimensions,
        )
        self._vector_store.store_vectors(chunks, embeddings, location)
        logger.info("pipeline stored %s chunks=%s", location, len(chunks))
        return len(chunks)


def build_pipeline(settings: Settings) -> RagPipeline:
    """Build the RAG pipeline with production components.

    Args:
        settings: Pipeline settings loaded from the environment.

    Returns:
        A fully wired ingestion pipeline.
    """
    logger.info(
        "build pipeline region=%s vector_bucket=%s vector_index=%s model=%s dimensions=%s",
        settings.aws_region,
        settings.s3_vector_bucket,
        settings.s3_vector_index,
        settings.embedding_model_id,
        settings.embedding_dimensions,
    )
    return RagPipeline(
        document_store=DocumentStore(settings),
        splitter=ChunkSplitter(settings),
        embeddings=EmbeddingModel(settings),
        vector_store=S3VectorStore(settings),
        settings=settings,
    )
