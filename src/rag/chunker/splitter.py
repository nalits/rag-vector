"""Chunking documents using langchain."""

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import Settings

logger = logging.getLogger(__name__)


class ChunkSplitter:
    """Splits documents into chunks using a recursive character strategy."""

    def __init__(self, settings: Settings) -> None:
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap
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
        logger.info(
            "split start documents=%s chunk_size=%s chunk_overlap=%s",
            len(documents),
            self._chunk_size,
            self._chunk_overlap,
        )
        chunks = self._splitter.split_documents(documents)
        logger.info("split done chunks=%s", len(chunks))
        return chunks
