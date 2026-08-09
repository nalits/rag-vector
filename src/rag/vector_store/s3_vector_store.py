"""Storing embeddings in an Amazon S3 vector index."""

from collections.abc import Iterator, Sequence
import json
import logging
from typing import TypedDict
import uuid

import boto3
from langchain_core.documents import Document

from rag.config import AwsService, LogSample, Settings, VectorDataType, VectorMetadataField

logger = logging.getLogger(__name__)


class _VectorPayload(TypedDict):
    key: str
    data: dict[str, list[float]]
    metadata: dict[str, str]


class S3VectorStore:
    """Writes chunk embeddings into an S3 vector index through the s3vectors API."""

    def __init__(self, settings: Settings) -> None:
        self._client = boto3.client(AwsService.S3_VECTORS, region_name=settings.aws_region)
        self._bucket = settings.s3_vector_bucket
        self._index = settings.s3_vector_index
        self._max_batch = settings.put_vectors_max_batch

    def store_vectors(
        self,
        chunks: list[Document],
        embeddings: list[list[float]],
        source: str,
    ) -> None:
        """Persist the chunk embeddings to the S3 vector index.

        Args:
            chunks: The document chunks.
            embeddings: One embedding vector per chunk.
            source: Source location of the document (e.g. ``s3://bucket/key``).

        Raises:
            ValueError: If the number of chunks and embeddings differ.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch"
            )
        if not chunks:
            logger.warning(
                "put_vectors skipped empty payload bucket=%s index=%s source=%s",
                self._bucket,
                self._index,
                source,
            )
            return
        vectors: list[_VectorPayload] = [
            {
                "key": str(uuid.uuid4()),
                "data": {VectorDataType.FLOAT32: embedding},
                "metadata": {
                    VectorMetadataField.TEXT_CHUNK: chunk.page_content,
                    VectorMetadataField.METADATA: json.dumps(
                        chunk.metadata | {VectorMetadataField.SOURCE: source}
                    ),
                    VectorMetadataField.SOURCE: source,
                },
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        batch_count = 0
        logger.info(
            "put_vectors start bucket=%s index=%s count=%s max_batch=%s source=%s",
            self._bucket,
            self._index,
            len(vectors),
            self._max_batch,
            source,
        )
        for batch in _batched(vectors, self._max_batch):
            batch_count += 1
            logger.info(
                "put_vectors batch bucket=%s index=%s batch=%s size=%s",
                self._bucket,
                self._index,
                batch_count,
                len(batch),
            )
            self._client.put_vectors(
                vectorBucketName=self._bucket,
                indexName=self._index,
                vectors=list(batch),
            )
        logger.info(
            "put_vectors done bucket=%s index=%s count=%s batches=%s source=%s",
            self._bucket,
            self._index,
            len(vectors),
            batch_count,
            source,
        )
        _log_inserted_sample(vectors[0])


def _log_inserted_sample(vector: _VectorPayload) -> None:
    """Log one newly inserted vector from this upload only."""
    embedding = vector["data"][VectorDataType.FLOAT32]
    metadata = vector["metadata"]
    logger.info(
        "put_vectors sample inserted key=%s source=%s dimensions=%s "
        "values_preview=%s text_chunk=%r",
        vector["key"],
        metadata[VectorMetadataField.SOURCE],
        len(embedding),
        embedding[: LogSample.EMBEDDING_VALUES],
        metadata[VectorMetadataField.TEXT_CHUNK],
    )


def _batched[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
