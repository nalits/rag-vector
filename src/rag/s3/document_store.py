"""Reading documents from S3."""

import boto3
from botocore.exceptions import ClientError

from rag.config import AwsService, Settings


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
        try:
            response = self._client.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            raise S3Error(f"failed to head object {s3_uri(bucket, key)}") from exc
        return int(response["ContentLength"])

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
        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            raise S3Error(f"failed to get object {s3_uri(bucket, key)}") from exc
        return bytes(response["Body"].read())


def s3_uri(bucket: str, key: str) -> str:
    """Return the S3 URI for a bucket and object key."""
    return f"s3://{bucket}/{key}"
