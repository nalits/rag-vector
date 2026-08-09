from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from rag.cli import app


def test_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    client.head_object.return_value = {"ContentLength": 23}
    body = Mock()
    body.read.return_value = b"hello world"
    client.get_object.return_value = {"Body": body}
    embeddings = Mock()
    embeddings.embed_documents.return_value = [[0.1]]
    monkeypatch.setattr("boto3.client", Mock(return_value=client))
    monkeypatch.setattr(
        "rag.embedder.embeddings.BedrockEmbeddings",
        Mock(return_value=embeddings),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "--bucket", "bucket", "--key", "notes.txt"])
    assert result.exit_code == 0
    assert "Ingested 1 chunks from s3://bucket/notes.txt" in result.output
    client.put_vectors.assert_called_once()
