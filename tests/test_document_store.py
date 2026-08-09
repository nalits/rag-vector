from unittest.mock import Mock

from botocore.exceptions import ClientError
import pytest

from rag.config import Settings
from rag.s3.document_store import DocumentStore, S3Error


def test_size(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    client.head_object.return_value = {"ContentLength": 42}
    monkeypatch.setattr("boto3.client", Mock(return_value=client))
    store = DocumentStore(Settings.from_env())
    assert store.size("bucket", "key") == 42
    client.head_object.assert_called_once_with(Bucket="bucket", Key="key")


def test_size_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    client.head_object.side_effect = ClientError({"Error": {}}, "HeadObject")
    monkeypatch.setattr("boto3.client", Mock(return_value=client))
    store = DocumentStore(Settings.from_env())
    with pytest.raises(S3Error, match="s3://bucket/key"):
        store.size("bucket", "key")


def test_download(monkeypatch: pytest.MonkeyPatch) -> None:
    body = Mock()
    body.read.return_value = b"content"
    client = Mock()
    client.get_object.return_value = {"Body": body}
    monkeypatch.setattr("boto3.client", Mock(return_value=client))
    store = DocumentStore(Settings.from_env())
    assert store.download("bucket", "key") == b"content"
    client.get_object.assert_called_once_with(Bucket="bucket", Key="key")


def test_download_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    client.get_object.side_effect = ClientError({"Error": {}}, "GetObject")
    monkeypatch.setattr("boto3.client", Mock(return_value=client))
    store = DocumentStore(Settings.from_env())
    with pytest.raises(S3Error, match="s3://bucket/key"):
        store.download("bucket", "key")
