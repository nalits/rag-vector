import pytest

from rag.config import Settings
from rag.validation.validator import (
    DocumentTooLargeError,
    UnsupportedDocumentTypeError,
    document_extension,
    validate_document,
)


@pytest.mark.parametrize(
    "key",
    [
        "guide.md",
        "report.doc",
        "letter.docx",
        "notes.txt",
        "README.MD",
    ],
)
def test_validate_document_supported(key: str) -> None:
    validate_document(key, size=0, settings=Settings.from_env())


def test_validate_document_unsupported_type() -> None:
    with pytest.raises(UnsupportedDocumentTypeError, match="unsupported document type"):
        validate_document("image.png", size=0, settings=Settings.from_env())


def test_validate_document_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "1")
    with pytest.raises(DocumentTooLargeError, match="exceeds size limit"):
        validate_document("guide.md", size=2 * 1024 * 1024, settings=Settings.from_env())


def test_validate_document_exact_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "1")
    validate_document("guide.md", size=1024 * 1024, settings=Settings.from_env())


def test_document_extension() -> None:
    assert document_extension("folder/notes.TXT").value == ".txt"
