# rag

Ingestion pipeline for retrieval-augmented generation. Documents uploaded to Amazon S3
are validated, chunked with a recursive character splitter, embedded with Amazon Titan
Text Embeddings V2, and stored in an [Amazon S3 Vectors](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html)
index.

An AWS Lambda function is triggered by `s3:ObjectCreated:*` events. The same pipeline
can be run locally through the `rag` CLI.

## Pipeline

1. Read the object from the document bucket.
2. Accept only `.md`, `.doc`, `.docx`, and `.txt` files that are at most 25 MB.
3. Load text with [langchain](https://www.langchain.com/) document loaders.
4. Split text with
   [`RecursiveCharacterTextSplitter`](https://docs.langchain.com/oss/python/integrations/text_splitters/recursive_text_splitter).
5. Embed chunks with langchain
   [`BedrockEmbeddings`](https://docs.langchain.com/oss/python/integrations/text_embedding/bedrock)
   using `amazon.titan-embed-text-v2:0`.
6. Write vectors to the S3 vector index with boto3 `s3vectors.put_vectors`.

## Project structure

```
rag
├── template.yaml          # SAM / CloudFormation stack
├── samconfig.yml          # single default SAM environment
├── Makefile               # export src/requirements.txt and sam build
├── src
│   ├── requirements.txt   # Lambda layer dependencies
│   └── rag/               # Lambda function code
├── .github/workflows/deploy.yml
├── tests/
├── noxfile.py
└── pyproject.toml
```

## Configuration

Runtime settings are environment variables. Lambda values are defined in
[`template.yaml`](./template.yaml) and supplied for the `default` environment in
[`samconfig.yml`](./samconfig.yml). The Python package does not hardcode defaults.

| Variable | SAM parameter | Description |
|:---------|:--------------|:------------|
| `MAX_FILE_SIZE_MB` | `MaxFileSizeMb` | Maximum accepted document size in MB. |
| `CHUNK_SIZE` | `ChunkSize` | Recursive splitter chunk size. |
| `CHUNK_OVERLAP` | `ChunkOverlap` | Recursive splitter chunk overlap. |
| `S3_VECTOR_BUCKET` | `VectorBucketName` | S3 vector bucket name. |
| `S3_VECTOR_INDEX` | `VectorIndexName` | S3 vector index name. |
| `EMBEDDING_MODEL_ID` | `EmbeddingModelId` | Bedrock embedding model ID. |
| `EMBEDDING_DIMENSIONS` | `EmbeddingDimensions` | Titan V2 output dimensions (`256`, `512`, or `1024`). |
| `EMBEDDING_NORMALIZE` | `EmbeddingNormalize` | Whether embeddings are L2-normalized (`true` / `false`). |
| `TEXT_ENCODING` | `TextEncoding` | Encoding for `.txt` and `.md` files. |
| `PUT_VECTORS_MAX_BATCH` | `PutVectorsMaxBatch` | Max vectors per `PutVectors` call (AWS limit 500). |
| `AWS_REGION` | stack region | AWS region. Set automatically on Lambda; set locally if needed. |

`DistanceMetric` is used only when creating the vector index (`cosine` by default).

## Local development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and sync the project:

```shell
uv python install 3.13
uv sync
```

Export the same variables defined in `template.yaml`, then ingest a document that is
already in S3:

```shell
export MAX_FILE_SIZE_MB=25
export CHUNK_SIZE=1000
export CHUNK_OVERLAP=200
export S3_VECTOR_BUCKET=rag-vector-bucket
export S3_VECTOR_INDEX=rag-vector-index
export AWS_REGION=eu-west-2
export EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
export EMBEDDING_DIMENSIONS=1024
export EMBEDDING_NORMALIZE=true
export TEXT_ENCODING=utf-8
export PUT_VECTORS_MAX_BATCH=500

uv run rag ingest --bucket my-documents --key user-guide.md
```

Library usage:

```python
from rag.config import Settings
from rag.pipeline import build_pipeline

pipeline = build_pipeline(Settings.from_env())
chunks = pipeline.process(bucket="my-documents", key="user-guide.md")
```

## Deploy with AWS SAM

CI, tests, and production deploy run from [`.github/workflows/deploy.yml`](./.github/workflows/deploy.yml).
Pushes to `main` deploy after quality and tests pass.

[`template.yaml`](./template.yaml) creates:

- A private document bucket named `{DocumentBucketName}-{account-id}`.
- A Lambda layer built from [`src/requirements.txt`](./src/requirements.txt).
- The ingestion function zip (application code only), invoked on `s3:ObjectCreated:*`.
- An S3 vector bucket and index sized to the embedding dimensions.

Bucket names and other stack values come from [`samconfig.yml`](./samconfig.yml).
First `sam deploy` creates the document bucket; later deploys keep the existing one.
Bucket prefixes come from [`samconfig.yml`](./samconfig.yml). The account ID is
appended so names stay unique. Change `region` there to match `AWS_REGION`.

```shell
make deploy
```

Or step by step:

```shell
make layer-requirements
sam build
sam deploy
```

`--resolve-s3` in `samconfig.yml` creates the SAM artifact bucket if it is missing.

A stack left in `ROLLBACK_COMPLETE` cannot be updated. `make deploy` and the
GitHub Actions deploy job delete that failed stack, then create it again.

Set these GitHub Actions configuration values for deploy:

| Name | Type | Purpose |
|:-----|:-----|:--------|
| `AWS_REGION` | variable | Deploy region (defaults to `eu-west-2`). |
| `AWS_ACCESS_KEY_ID` | variable | Deploy credentials. |
| `AWS_SECRET_ACCESS_KEY` | secret | Deploy credentials. |

Upload a supported document to the `DocumentBucketName` stack output. The function
requires Amazon Bedrock access to Titan Text Embeddings V2 in the deployed region.

> [!NOTE]
> The Lambda layer excludes `unstructured` (it exceeds the 250 MB unzipped layer
> limit). `.md`, `.txt`, and `.docx` work in Lambda. Legacy `.doc` files can be
> ingested locally with `uv run rag ingest` where the full dependency set is
> installed.

## Quality checks

Automated checks run through [Nox](https://nox.thea.codes/en/stable/). After `uv sync`:

```shell
uv run nox
```

Individual sessions:

```shell
uv run nox --session test-3.13
uv run nox --session lint
uv run nox --session fmt
uv run nox --session type_check
```

Unit tests mock AWS clients and Bedrock embeddings so they run offline. Coverage is
enforced at 100%.

## Licensing

MIT. See [`LICENSE.txt`](./LICENSE.txt) and [`pyproject.toml`](./pyproject.toml).
