"""Embeddings for the RAG pipeline using Amazon Bedrock."""

import logging

from langchain_aws import BedrockEmbeddings

from rag.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wraps langchain BedrockEmbeddings using Amazon Titan Text Embeddings V2."""

    def __init__(self, settings: Settings) -> None:
        self._model_id = settings.embedding_model_id
        self._dimensions = settings.embedding_dimensions
        self._normalize = settings.embedding_normalize
        self._embeddings = BedrockEmbeddings(
            model_id=settings.embedding_model_id,
            region_name=settings.aws_region,
            dimensions=settings.embedding_dimensions,
            normalize=settings.embedding_normalize,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors.

        Args:
            texts: Texts to embed.

        Returns:
            The embedding vectors, one per text.
        """
        logger.info(
            "embedding start model=%s texts=%s dimensions=%s normalize=%s",
            self._model_id,
            len(texts),
            self._dimensions,
            self._normalize,
        )
        vectors = self._embeddings.embed_documents(texts)
        logger.info(
            "embedding done model=%s count=%s dimensions=%s",
            self._model_id,
            len(vectors),
            self._dimensions,
        )
        return vectors
