import pytest

from rag.config import DocumentExtension, EnvVar, Settings, document_suffix


def test_settings_from_env() -> None:
    settings = Settings.from_env()
    assert settings.max_file_size_mb == 25
    assert settings.max_file_size_bytes == 25 * 1024 * 1024
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.s3_vector_bucket == "vector-bucket"
    assert settings.s3_vector_index == "vector-index"
    assert settings.aws_region == "us-east-1"
    assert settings.embedding_model_id == "amazon.titan-embed-text-v2:0"
    assert settings.embedding_dimensions == 1024
    assert settings.embedding_normalize is True
    assert settings.text_encoding == "utf-8"
    assert settings.put_vectors_max_batch == 500


def test_settings_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "10")
    monkeypatch.setenv("CHUNK_SIZE", "1500")
    monkeypatch.setenv("CHUNK_OVERLAP", "250")
    monkeypatch.setenv("EMBEDDING_NORMALIZE", "false")
    settings = Settings.from_env()
    assert settings.max_file_size_bytes == 10 * 1024 * 1024
    assert settings.chunk_size == 1500
    assert settings.chunk_overlap == 250
    assert settings.embedding_normalize is False


@pytest.mark.parametrize(
    "name",
    [
        EnvVar.MAX_FILE_SIZE_MB,
        EnvVar.CHUNK_SIZE,
        EnvVar.S3_VECTOR_BUCKET,
        EnvVar.S3_VECTOR_INDEX,
        EnvVar.EMBEDDING_MODEL_ID,
        EnvVar.TEXT_ENCODING,
    ],
)
def test_settings_missing_required(name: EnvVar, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match=str(name)):
        Settings.from_env()


def test_settings_missing_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    with pytest.raises(ValueError, match="AWS_REGION"):
        Settings.from_env()


def test_settings_aws_default_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    assert Settings.from_env().aws_region == "eu-west-1"


def test_settings_invalid_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "large")
    with pytest.raises(ValueError, match="must be an integer"):
        Settings.from_env()


def test_settings_invalid_bool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_NORMALIZE", "yes")
    with pytest.raises(ValueError, match="must be true or false"):
        Settings.from_env()


def test_settings_invalid_chunk_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings.from_env()


def test_settings_non_positive_max_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "0")
    with pytest.raises(ValueError, match="MAX_FILE_SIZE_MB"):
        Settings.from_env()


def test_settings_non_positive_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "-1")
    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        Settings.from_env()


def test_settings_negative_chunk_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_OVERLAP", "-1")
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings.from_env()


def test_settings_non_positive_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "0")
    with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS"):
        Settings.from_env()


def test_settings_non_positive_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUT_VECTORS_MAX_BATCH", "0")
    with pytest.raises(ValueError, match="PUT_VECTORS_MAX_BATCH"):
        Settings.from_env()


def test_settings_blank_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S3_VECTOR_BUCKET", "  ")
    with pytest.raises(ValueError, match="S3_VECTOR_BUCKET"):
        Settings.from_env()


def test_settings_blank_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "  ")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    with pytest.raises(ValueError, match="AWS_REGION"):
        Settings.from_env()


def test_document_suffix() -> None:
    assert document_suffix("path/README.MD") == ".md"


def test_document_extension_values() -> None:
    assert {item.value for item in DocumentExtension} == {".md", ".doc", ".docx", ".txt"}
