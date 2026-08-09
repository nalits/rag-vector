---
icon: material/database
status: new
---

# `rag` User Guide

## Overview

The `rag` package implements a retrieval-augmented generation ingestion pipeline:

1. A document is uploaded to an S3 bucket.
2. An AWS Lambda function is triggered by the S3 `ObjectCreated` event.
3. The document is validated (`.md`, `.doc`, `.docx`, or `.txt`, up to 25 MB).
4. The document is loaded, chunked with a recursive character splitter, and embedded with
   Amazon Titan Text Embeddings V2.
5. The embeddings are stored in an Amazon S3 vector index.

## Installation

First, [install `uv`](https://docs.astral.sh/uv/getting-started/installation):

=== "macOS and Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

Then install the `rag` package and its dependencies:

```bash
uv sync
```

## Configuration

Runtime settings are environment variables. Lambda values are defined in `template.yaml`
and supplied for the `default` environment in `samconfig.yml`.

| Variable | Description |
|:---------|:------------|
| `MAX_FILE_SIZE_MB` | Maximum accepted document size in MB. |
| `CHUNK_SIZE` | Recursive splitter chunk size. |
| `CHUNK_OVERLAP` | Recursive splitter chunk overlap. |
| `S3_VECTOR_BUCKET` | S3 vector bucket name. |
| `S3_VECTOR_INDEX` | S3 vector index name. |
| `EMBEDDING_MODEL_ID` | Bedrock embedding model ID. |
| `EMBEDDING_DIMENSIONS` | Titan V2 output dimensions. |
| `EMBEDDING_NORMALIZE` | Whether embeddings are L2-normalized. |
| `TEXT_ENCODING` | Encoding for `.txt` and `.md` files. |
| `PUT_VECTORS_MAX_BATCH` | Max vectors per `PutVectors` call. |
| `AWS_REGION` | AWS region (set automatically on Lambda). |

## Quick start

Run the pipeline locally for a document already uploaded to S3:

```bash
uv run rag ingest --help
```

Use `rag` as a library:

*[API]: Application Programming Interface

```python
from rag.config import Settings
from rag.pipeline import build_pipeline

pipeline = build_pipeline(Settings.from_env())
chunks = pipeline.process(bucket="my-documents", key="user-guide.md")
```

## Deploy

CI and production deploys use `.github/workflows/deploy.yml`. Locally:

```bash
make deploy
```

The document bucket is created by `template.yaml` on first deploy using
`DocumentBucketName` from `samconfig.yml`. Later deploys keep that bucket.
Third-party packages ship as a slim Lambda layer (without `unstructured`, to stay
under the 250 MB unzipped limit). Upload `.md`, `.txt`, or `.docx` documents to
trigger ingestion. Legacy `.doc` files are supported in the local CLI only.
