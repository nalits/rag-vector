from unittest.mock import Mock

import pytest

from rag.config import Settings
from rag.embedder.embeddings import EmbeddingModel


def test_embed_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    embeddings = Mock()
    embeddings.embed_documents.return_value = [[0.1, 0.2]]
    factory = Mock(return_value=embeddings)
    monkeypatch.setattr("rag.embedder.embeddings.BedrockEmbeddings", factory)
    model = EmbeddingModel(Settings.from_env())
    assert model.embed_documents(["text"]) == [[0.1, 0.2]]
    embeddings.embed_documents.assert_called_once_with(["text"])
    assert factory.call_args.kwargs["model_id"] == "amazon.titan-embed-text-v2:0"
    assert factory.call_args.kwargs["region_name"] == "us-east-1"
    assert factory.call_args.kwargs["dimensions"] == 1024
    assert factory.call_args.kwargs["normalize"] is True
