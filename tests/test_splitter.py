from langchain_core.documents import Document
import pytest

from rag.chunker.splitter import ChunkSplitter
from rag.config import Settings


def test_split_long_text() -> None:
    splitter = ChunkSplitter(Settings.from_env())
    text = "word " * 500
    chunks = splitter.split([Document(page_content=text)])
    assert len(chunks) > 1
    assert sum(len(chunk.page_content.split()) for chunk in chunks) > len(text.split())


def test_split_short_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "20")
    splitter = ChunkSplitter(Settings.from_env())
    chunks = splitter.split([Document(page_content="short text")])
    assert len(chunks) == 1
    assert chunks[0].page_content == "short text"
