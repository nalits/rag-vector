"""Storing embeddings in an Amazon S3 vector index."""

from collections.abc import Iterator, Sequence
import json
import uuid

import boto3
from langchain_core.documents import Document

from rag.config import AwsService, Settings, VectorDataType, VectorMetadataField


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
            return
        vectors = [
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
        for batch in _batched(vectors, self._max_batch):
            self._client.put_vectors(
                vectorBucketName=self._bucket,
                indexName=self._index,
                vectors=list(batch),
            )


def _batched[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
