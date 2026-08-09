from unittest.mock import Mock

from langchain_core.documents import Document
import pytest

from rag.config import Settings
from rag.pipeline import RagPipeline, build_pipeline
from rag.validation.validator import DocumentTooLargeError, UnsupportedDocumentTypeError


def _pipeline(
    document_store: Mock,
    splitter: Mock,
    embeddings: Mock,
    vector_store: Mock,
) -> RagPipeline:
    return RagPipeline(
        document_store=document_store,
        splitter=splitter,
        embeddings=embeddings,
        vector_store=vector_store,
        settings=Settings.from_env(),
    )


def test_process() -> None:
    document_store, splitter, embeddings, vector_store = (
        Mock(),
        Mock(),
        Mock(),
        Mock(),
    )
    document_store.size.return_value = 100
    document_store.download.return_value = b"content"
    splitter.split.return_value = [Document(page_content="chunk")]
    embeddings.embed_documents.return_value = [[0.1]]
    pipeline = _pipeline(document_store, splitter, embeddings, vector_store)
    assert pipeline.process("bucket", "guide.md") == 1
    document_store.size.assert_called_once_with("bucket", "guide.md")
    document_store.download.assert_called_once_with("bucket", "guide.md")
    splitter.split.assert_called_once()
    embeddings.embed_documents.assert_called_once_with(["chunk"])
    vector_store.store_vectors.assert_called_once_with(
        [Document(page_content="chunk")],
        [[0.1]],
        "s3://bucket/guide.md",
    )


def test_process_skips_blank_chunks() -> None:
    document_store, splitter, embeddings, vector_store = (
        Mock(),
        Mock(),
        Mock(),
        Mock(),
    )
    document_store.size.return_value = 100
    document_store.download.return_value = b"content"
    splitter.split.return_value = [Document(page_content="  "), Document(page_content="")]
    pipeline = _pipeline(document_store, splitter, embeddings, vector_store)
    assert pipeline.process("bucket", "guide.md") == 0
    embeddings.embed_documents.assert_not_called()
    vector_store.store_vectors.assert_not_called()


def test_process_unsupported_type() -> None:
    document_store, splitter, embeddings, vector_store = (
        Mock(),
        Mock(),
        Mock(),
        Mock(),
    )
    document_store.size.return_value = 100
    pipeline = _pipeline(document_store, splitter, embeddings, vector_store)
    with pytest.raises(UnsupportedDocumentTypeError):
        pipeline.process("bucket", "image.png")
    document_store.download.assert_not_called()


def test_process_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "1")
    document_store, splitter, embeddings, vector_store = (
        Mock(),
        Mock(),
        Mock(),
        Mock(),
    )
    document_store.size.return_value = 2 * 1024 * 1024
    pipeline = _pipeline(document_store, splitter, embeddings, vector_store)
    with pytest.raises(DocumentTooLargeError):
        pipeline.process("bucket", "guide.md")
    document_store.download.assert_not_called()


def test_build_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("boto3.client", Mock())
    monkeypatch.setattr("rag.embedder.embeddings.BedrockEmbeddings", Mock())
    pipeline = build_pipeline(Settings.from_env())
    assert isinstance(pipeline, RagPipeline)
