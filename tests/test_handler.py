from unittest.mock import Mock

import pytest

from rag.handler import lambda_handler


def _event(*records: tuple[str, str]) -> dict[str, object]:
    return {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": bucket},
                    "object": {"key": key},
                }
            }
            for bucket, key in records
        ]
    }


def test_lambda_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = Mock()
    pipeline.process.return_value = 3
    monkeypatch.setattr("rag.handler.build_pipeline", Mock(return_value=pipeline))
    result = lambda_handler(_event(("bucket", "guide.md")), None)
    assert result == {
        "results": [
            {
                "bucket": "bucket",
                "key": "guide.md",
                "chunks": 3,
                "status": "ok",
                "error": None,
            }
        ]
    }
    pipeline.process.assert_called_once_with("bucket", "guide.md")


def test_lambda_handler_eventbridge(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = Mock()
    pipeline.process.return_value = 4
    monkeypatch.setattr("rag.handler.build_pipeline", Mock(return_value=pipeline))
    event = {
        "detail": {
            "bucket": {"name": "docs"},
            "object": {"key": "folder/my+guide.md"},
        }
    }
    result = lambda_handler(event, None)
    assert result["results"][0]["status"] == "ok"
    assert result["results"][0]["chunks"] == 4
    pipeline.process.assert_called_once_with("docs", "folder/my guide.md")


def test_lambda_handler_decodes_key(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = Mock()
    pipeline.process.return_value = 1
    monkeypatch.setattr("rag.handler.build_pipeline", Mock(return_value=pipeline))
    lambda_handler(_event(("bucket", "folder/my+guide.md")), None)
    pipeline.process.assert_called_once_with("bucket", "folder/my guide.md")


def test_lambda_handler_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = Mock()
    pipeline.process.side_effect = RuntimeError("boom")
    monkeypatch.setattr("rag.handler.build_pipeline", Mock(return_value=pipeline))
    result = lambda_handler(_event(("bucket", "guide.md")), None)
    assert result["results"][0]["status"] == "error"
    assert result["results"][0]["error"] == "boom"
    assert result["results"][0]["chunks"] == 0


def test_lambda_handler_multiple_records(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = Mock()
    pipeline.process.side_effect = [2, RuntimeError("boom")]
    monkeypatch.setattr("rag.handler.build_pipeline", Mock(return_value=pipeline))
    result = lambda_handler(
        _event(("bucket-a", "a.md"), ("bucket-b", "b.md")),
        None,
    )
    assert [record["status"] for record in result["results"]] == ["ok", "error"]


def test_lambda_handler_empty_records(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = Mock()
    monkeypatch.setattr("rag.handler.build_pipeline", Mock(return_value=pipeline))
    result = lambda_handler({"Records": []}, None)
    assert result == {"results": []}
    pipeline.process.assert_not_called()
