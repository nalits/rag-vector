"""Chunking documents using langchain."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import Settings


class ChunkSplitter:
    """Splits documents into chunks using a recursive character strategy."""

    def __init__(self, settings: Settings) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Split the given documents into smaller overlapping chunks.

        Args:
            documents: Documents to split.

        Returns:
            The split documents.
        """
        return self._splitter.split_documents(documents)
