#!/usr/bin/env python3

"""Command line interface for running the RAG pipeline locally."""

from typing import Annotated

from rich.console import Console
from typer import Option, Typer

from rag.config import Settings, configure_logging
from rag.pipeline import build_pipeline
from rag.s3.document_store import s3_uri

app = Typer(add_completion=False)


@app.callback()
def main() -> None:
    """RAG document ingestion pipeline."""


@app.command()
def ingest(
    bucket: Annotated[str, Option(help="S3 bucket containing the document")],
    key: Annotated[str, Option(help="S3 object key of the document")],
) -> None:
    """Ingest a document from S3 into the S3 vector index."""
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    chunks = build_pipeline(settings).process(bucket, key)
    Console().print(f"Ingested {chunks} chunks from {s3_uri(bucket, key)}")


if __name__ == "__main__":  # pragma: no cover
    app()
