"""AWS Lambda handler for the RAG ingestion pipeline.

Accepts S3 ``ObjectCreated`` notifications delivered either as classic S3
records or as EventBridge ``Object Created`` events.
"""

from collections.abc import Mapping
import logging
from typing import Any
from urllib.parse import unquote_plus

from rag.config import IngestionStatus, Settings
from rag.pipeline import build_pipeline

logger = logging.getLogger(__name__)


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Process S3 ``ObjectCreated`` events for uploaded documents.

    Args:
        event: The Lambda event containing S3 record notifications.
        context: The Lambda runtime context.

    Returns:
        The ingestion result for every record in the event.
    """
    pipeline = build_pipeline(Settings.from_env())
    results: list[dict[str, Any]] = []
    for bucket, key in _s3_objects(event):
        try:
            chunks = pipeline.process(bucket, key)
            status = IngestionStatus.OK
            error = None
        except Exception as exc:
            logger.exception("failed to ingest s3://%s/%s", bucket, key)
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
    return {"results": results}


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
