import json
import logging
from unittest.mock import Mock

from langchain_core.documents import Document
import pytest

from rag.config import LogSample, Settings, VectorMetadataField
from rag.vector_store.s3_vector_store import S3VectorStore


def _make_store(client: Mock, monkeypatch: pytest.MonkeyPatch) -> S3VectorStore:
    monkeypatch.setattr("boto3.client", Mock(return_value=client))
    return S3VectorStore(Settings.from_env())


def test_store_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    store = _make_store(client, monkeypatch)
    chunks = [Document(page_content="chunk text", metadata={"page": 1})]
    store.store_vectors(chunks, [[0.5, 0.5]], "s3://bucket/key")
    assert client.put_vectors.call_count == 1
    call = client.put_vectors.call_args.kwargs
    assert call["vectorBucketName"] == "vector-bucket"
    assert call["indexName"] == "vector-index"
    assert len(call["vectors"]) == 1
    vector = call["vectors"][0]
    assert vector["data"] == {"float32": [0.5, 0.5]}
    assert vector["metadata"][VectorMetadataField.TEXT_CHUNK] == "chunk text"
    assert vector["metadata"][VectorMetadataField.SOURCE] == "s3://bucket/key"
    assert json.loads(vector["metadata"][VectorMetadataField.METADATA]) == {
        "page": 1,
        "source": "s3://bucket/key",
    }


def test_store_vectors_multiple(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    store = _make_store(client, monkeypatch)
    chunks = [Document(page_content="a"), Document(page_content="b")]
    store.store_vectors(chunks, [[1.0], [2.0]], "s3://bucket/key")
    vectors = client.put_vectors.call_args.kwargs["vectors"]
    assert [vector["data"] for vector in vectors] == [{"float32": [1.0]}, {"float32": [2.0]}]


def test_store_vectors_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUT_VECTORS_MAX_BATCH", "2")
    client = Mock()
    store = _make_store(client, monkeypatch)
    chunks = [Document(page_content="a"), Document(page_content="b"), Document(page_content="c")]
    store.store_vectors(chunks, [[1.0], [2.0], [3.0]], "s3://bucket/key")
    assert client.put_vectors.call_count == 2
    first = client.put_vectors.call_args_list[0].kwargs["vectors"]
    second = client.put_vectors.call_args_list[1].kwargs["vectors"]
    assert len(first) == 2
    assert len(second) == 1


def test_store_vectors_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    store = _make_store(client, monkeypatch)
    store.store_vectors([], [], "s3://bucket/key")
    client.put_vectors.assert_not_called()


def test_store_vectors_logs_one_new_sample(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = Mock()
    store = _make_store(client, monkeypatch)
    chunks = [Document(page_content="first chunk"), Document(page_content="second chunk")]
    with caplog.at_level(logging.INFO):
        store.store_vectors(chunks, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "s3://bucket/notes.txt")
    samples = [
        record.message
        for record in caplog.records
        if "put_vectors sample inserted" in record.message
    ]
    assert len(samples) == 1
    sample = samples[0]
    assert "source=s3://bucket/notes.txt" in sample
    assert "dimensions=3" in sample
    assert str([0.1, 0.2, 0.3][: LogSample.EMBEDDING_VALUES]) in sample
    assert "first chunk" in sample
    assert "second chunk" not in sample


def test_store_vectors_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    store = _make_store(client, monkeypatch)
    chunks = [Document(page_content="a")]
    with pytest.raises(ValueError, match="count mismatch"):
        store.store_vectors(chunks, [], "s3://bucket/key")
    client.put_vectors.assert_not_called()
