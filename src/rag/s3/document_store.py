"""Reading documents from S3."""

import logging

import boto3
from botocore.exceptions import ClientError

from rag.config import AwsService, Settings

logger = logging.getLogger(__name__)


class S3Error(RuntimeError):
    """Error generated when an S3 operation fails."""


class DocumentStore:
    """Reads documents from an S3 bucket."""

    def __init__(self, settings: Settings) -> None:
        self._client = boto3.client(AwsService.S3, region_name=settings.aws_region)

    def size(self, bucket: str, key: str) -> int:
        """Return the size in bytes of the object at the given location.

        Args:
            bucket: S3 bucket name.
            key: S3 object key.

        Raises:
            S3Error: If the object metadata cannot be retrieved.

        Returns:
            The object size in bytes.
        """
        location = s3_uri(bucket, key)
        logger.info("s3 head_object start %s", location)
        try:
            response = self._client.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            logger.exception("s3 head_object failed %s", location)
            raise S3Error(f"failed to head object {location}") from exc
        size = int(response["ContentLength"])
        logger.info("s3 head_object ok %s bytes=%s", location, size)
        return size

    def download(self, bucket: str, key: str) -> bytes:
        """Download the object content at the given location.

        Args:
            bucket: S3 bucket name.
            key: S3 object key.

        Raises:
            S3Error: If the object cannot be downloaded.

        Returns:
            The object content as bytes.
        """
        location = s3_uri(bucket, key)
        logger.info("s3 get_object start %s", location)
        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            logger.exception("s3 get_object failed %s", location)
            raise S3Error(f"failed to get object {location}") from exc
        content = bytes(response["Body"].read())
        logger.info("s3 get_object ok %s bytes=%s", location, len(content))
        return content


def s3_uri(bucket: str, key: str) -> str:
    """Return the S3 URI for a bucket and object key."""
    return f"s3://{bucket}/{key}"
