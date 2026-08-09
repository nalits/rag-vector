"""Environment-driven configuration for the RAG pipeline.

All runtime values are read from environment variables defined on the Lambda
function in ``template.yaml``. There are no module-level defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
import logging
import os
from pathlib import Path


class EnvVar(StrEnum):
    """Names of environment variables consumed by the pipeline."""

    MAX_FILE_SIZE_MB = "MAX_FILE_SIZE_MB"
    CHUNK_SIZE = "CHUNK_SIZE"
    CHUNK_OVERLAP = "CHUNK_OVERLAP"
    S3_VECTOR_BUCKET = "S3_VECTOR_BUCKET"
    S3_VECTOR_INDEX = "S3_VECTOR_INDEX"
    AWS_REGION = "AWS_REGION"
    AWS_DEFAULT_REGION = "AWS_DEFAULT_REGION"
    EMBEDDING_MODEL_ID = "EMBEDDING_MODEL_ID"
    EMBEDDING_DIMENSIONS = "EMBEDDING_DIMENSIONS"
    EMBEDDING_NORMALIZE = "EMBEDDING_NORMALIZE"
    TEXT_ENCODING = "TEXT_ENCODING"
    PUT_VECTORS_MAX_BATCH = "PUT_VECTORS_MAX_BATCH"
    LOG_LEVEL = "LOG_LEVEL"


class LogLevel(StrEnum):
    """Supported application log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LambdaEventSource(StrEnum):
    """Lambda invocation event shapes handled by the ingestion function."""

    S3 = "s3"
    EVENTBRIDGE = "eventbridge"


class DocumentExtension(StrEnum):
    """Document types accepted by the ingestion pipeline."""

    MARKDOWN = ".md"
    DOC = ".doc"
    DOCX = ".docx"
    TXT = ".txt"


class AwsService(StrEnum):
    """Boto3 service identifiers used by the pipeline."""

    S3 = "s3"
    S3_VECTORS = "s3vectors"


class VectorMetadataField(StrEnum):
    """Metadata keys stored alongside each vector."""

    TEXT_CHUNK = "AMAZON_BEDROCK_TEXT_CHUNK"
    METADATA = "AMAZON_BEDROCK_METADATA"
    SOURCE = "source"


class VectorDataType(StrEnum):
    """S3 Vectors payload data type."""

    FLOAT32 = "float32"


class IngestionStatus(StrEnum):
    """Per-record outcome returned by the Lambda handler."""

    OK = "ok"
    ERROR = "error"


class BooleanString(StrEnum):
    """Canonical string forms for boolean environment variables."""

    TRUE = "true"
    FALSE = "false"


class SizeUnit(IntEnum):
    """Byte conversion factors."""

    BYTES_PER_MEBIBYTE = 1024 * 1024


@dataclass(frozen=True)
class Settings:
    """Immutable pipeline settings loaded from the process environment."""

    max_file_size_mb: int
    chunk_size: int
    chunk_overlap: int
    s3_vector_bucket: str
    s3_vector_index: str
    aws_region: str
    embedding_model_id: str
    embedding_dimensions: int
    embedding_normalize: bool
    text_encoding: str
    put_vectors_max_batch: int
    log_level: LogLevel

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables.

        Raises:
            ValueError: If a required variable is missing or invalid.
        """
        settings = cls(
            max_file_size_mb=_required_int(EnvVar.MAX_FILE_SIZE_MB),
            chunk_size=_required_int(EnvVar.CHUNK_SIZE),
            chunk_overlap=_required_int(EnvVar.CHUNK_OVERLAP),
            s3_vector_bucket=_required_str(EnvVar.S3_VECTOR_BUCKET),
            s3_vector_index=_required_str(EnvVar.S3_VECTOR_INDEX),
            aws_region=_aws_region(),
            embedding_model_id=_required_str(EnvVar.EMBEDDING_MODEL_ID),
            embedding_dimensions=_required_int(EnvVar.EMBEDDING_DIMENSIONS),
            embedding_normalize=_required_bool(EnvVar.EMBEDDING_NORMALIZE),
            text_encoding=_required_str(EnvVar.TEXT_ENCODING),
            put_vectors_max_batch=_required_int(EnvVar.PUT_VECTORS_MAX_BATCH),
            log_level=_required_log_level(EnvVar.LOG_LEVEL),
        )
        settings._validate()
        return settings

    @property
    def max_file_size_bytes(self) -> int:
        """Return the maximum accepted document size in bytes."""
        return self.max_file_size_mb * SizeUnit.BYTES_PER_MEBIBYTE

    def _validate(self) -> None:
        if self.max_file_size_mb <= 0:
            raise ValueError(f"{EnvVar.MAX_FILE_SIZE_MB} must be greater than 0")
        if self.chunk_size <= 0:
            raise ValueError(f"{EnvVar.CHUNK_SIZE} must be greater than 0")
        if self.chunk_overlap < 0:
            raise ValueError(f"{EnvVar.CHUNK_OVERLAP} must be at least 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"{EnvVar.CHUNK_OVERLAP} must be smaller than {EnvVar.CHUNK_SIZE}")
        if self.embedding_dimensions <= 0:
            raise ValueError(f"{EnvVar.EMBEDDING_DIMENSIONS} must be greater than 0")
        if self.put_vectors_max_batch <= 0:
            raise ValueError(f"{EnvVar.PUT_VECTORS_MAX_BATCH} must be greater than 0")


def document_suffix(key: str) -> str:
    """Return the lower-cased file suffix for an object key."""
    return Path(key).suffix.lower()


def _required_str(name: EnvVar) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(f"required environment variable {name} is not set")
    return value


def _required_int(name: EnvVar) -> int:
    value = _required_str(name)
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"environment variable {name} must be an integer") from exc


def _required_bool(name: EnvVar) -> bool:
    value = _required_str(name).strip().lower()
    if value == BooleanString.TRUE:
        return True
    if value == BooleanString.FALSE:
        return False
    raise ValueError(f"environment variable {name} must be true or false")


def _required_log_level(name: EnvVar) -> LogLevel:
    value = _required_str(name).strip().upper()
    try:
        return LogLevel(value)
    except ValueError:
        allowed = ", ".join(level.value for level in LogLevel)
        raise ValueError(f"environment variable {name} must be one of: {allowed}") from None


def configure_logging(level: LogLevel) -> None:
    """Configure root and application loggers for Lambda and the local CLI.

    Args:
        level: Minimum severity that should be emitted.
    """
    numeric_level = logging.getLevelNamesMapping()[level]
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.addHandler(handler)
    root.setLevel(numeric_level)
    logging.getLogger("rag").setLevel(numeric_level)


def _aws_region() -> str:
    value = os.getenv(EnvVar.AWS_REGION) or os.getenv(EnvVar.AWS_DEFAULT_REGION)
    if value is None or value.strip() == "":
        raise ValueError(
            f"required environment variable {EnvVar.AWS_REGION} or "
            f"{EnvVar.AWS_DEFAULT_REGION} is not set"
        )
    return value
