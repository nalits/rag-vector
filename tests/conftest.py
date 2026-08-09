import pytest


@pytest.fixture(autouse=True)
def pipeline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "25")
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    monkeypatch.setenv("S3_VECTOR_BUCKET", "vector-bucket")
    monkeypatch.setenv("S3_VECTOR_INDEX", "vector-index")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setenv("EMBEDDING_NORMALIZE", "true")
    monkeypatch.setenv("TEXT_ENCODING", "utf-8")
    monkeypatch.setenv("PUT_VECTORS_MAX_BATCH", "500")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
