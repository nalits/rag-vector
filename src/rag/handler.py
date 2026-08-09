"""AWS Lambda handler for the RAG ingestion pipeline.

Accepts S3 ``ObjectCreated`` notifications delivered either as classic S3
records or as EventBridge ``Object Created`` events.
"""

from collections.abc import Mapping
import logging
from typing import Any
from urllib.parse import unquote_plus

from rag.config import IngestionStatus, LambdaEventSource, Settings, configure_logging
from rag.pipeline import build_pipeline
from rag.s3.document_store import s3_uri

logger = logging.getLogger(__name__)


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Process S3 ``ObjectCreated`` events for uploaded documents.

    Args:
        event: The Lambda event containing S3 record notifications.
        context: The Lambda runtime context.

    Returns:
        The ingestion result for every record in the event.
    """
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    objects = _s3_objects(event)
    logger.info(
        "lambda invoked request_id=%s event_source=%s objects=%s vector_bucket=%s "
        "vector_index=%s region=%s model=%s",
        getattr(context, "aws_request_id", None),
        _event_source(event),
        len(objects),
        settings.s3_vector_bucket,
        settings.s3_vector_index,
        settings.aws_region,
        settings.embedding_model_id,
    )
    if not objects:
        logger.warning("no s3 objects in event")
    pipeline = build_pipeline(settings)
    results: list[dict[str, Any]] = []
    for bucket, key in objects:
        location = s3_uri(bucket, key)
        try:
            logger.info("ingest start %s", location)
            chunks = pipeline.process(bucket, key)
            status = IngestionStatus.OK
            error = None
            logger.info("ingest ok %s chunks=%s", location, chunks)
        except Exception as exc:
            logger.exception("ingest failed %s", location)
            chunks = 0
            status = IngestionStatus.ERROR
            error = str(exc)
        results.append(
            {
                "bucket": bucket,
                "key": key,
                "chunks": chunks,
                "status": status,
                "error": error,
            }
        )
    logger.info("lambda complete results=%s", len(results))
    return {"results": results}


def _event_source(event: Mapping[str, Any]) -> LambdaEventSource:
    if "Records" in event:
        return LambdaEventSource.S3
    return LambdaEventSource.EVENTBRIDGE


def _s3_objects(event: Mapping[str, Any]) -> list[tuple[str, str]]:
    if "Records" in event:
        return [
            (
                record["s3"]["bucket"]["name"],
                unquote_plus(record["s3"]["object"]["key"]),
            )
            for record in event["Records"]
        ]
    detail = event["detail"]
    return [(detail["bucket"]["name"], unquote_plus(detail["object"]["key"]))]
