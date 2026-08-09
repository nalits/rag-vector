"""Embeddings for the RAG pipeline using Amazon Bedrock."""

from langchain_aws import BedrockEmbeddings

from rag.config import Settings


class EmbeddingModel:
    """Wraps langchain BedrockEmbeddings using Amazon Titan Text Embeddings V2."""

    def __init__(self, settings: Settings) -> None:
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
        return self._embeddings.embed_documents(texts)
