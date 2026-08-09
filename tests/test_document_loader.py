from langchain_core.documents import Document
import pytest

from rag.config import Settings
from rag.loader.document_loader import load_document
from rag.validation.validator import UnsupportedDocumentTypeError


def test_load_text() -> None:
    documents = load_document("notes.txt", b"hello world", Settings.from_env())
    assert documents == [Document(page_content="hello world")]


def test_load_markdown() -> None:
    documents = load_document("guide.md", b"# Title", Settings.from_env())
    assert documents == [Document(page_content="# Title")]


def test_load_docx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rag.loader.document_loader.docx2txt.process", lambda _path: "docx text")
    documents = load_document("letter.docx", b"content", Settings.from_env())
    assert documents == [Document(page_content="docx text")]


def test_load_docx_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rag.loader.document_loader.docx2txt.process", lambda _path: None)
    documents = load_document("letter.docx", b"content", Settings.from_env())
    assert documents == [Document(page_content="")]


def test_load_doc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rag.loader.document_loader.partition_doc",
        lambda filename: ["paragraph"],
    )
    documents = load_document("report.doc", b"content", Settings.from_env())
    assert documents == [Document(page_content="paragraph")]


def test_load_unsupported_type() -> None:
    with pytest.raises(UnsupportedDocumentTypeError, match="unsupported document type"):
        load_document("image.png", b"content", Settings.from_env())


def test_text_loader_encoding() -> None:
    documents = load_document("notes.txt", "café".encode(), Settings.from_env())
    assert documents == [Document(page_content="café")]
