"""Tests for voz_bridge.errors — error mapping and formatting."""

from __future__ import annotations

import json

import pytest

from voz_bridge.errors import (
    auth_error,
    error_from_exception,
    make_error_response,
    make_stream_error_chunk,
)


class TestMakeErrorResponse:
    def test_status_code(self):
        resp = make_error_response(502, "bad_gateway", "Server down")
        assert resp.status_code == 502

    def test_body_format(self):
        resp = make_error_response(404, "not_found", "Missing")
        body = json.loads(resp.body)
        assert "error" in body
        assert body["error"]["message"] == "Missing"
        assert body["error"]["type"] == "not_found"
        assert body["error"]["code"] == 404

    def test_param_is_none(self):
        resp = make_error_response(500, "internal", "Oops")
        body = json.loads(resp.body)
        assert body["error"]["param"] is None


class TestErrorFromException:
    def test_connection_refused(self):
        exc = ConnectionRefusedError("refused")
        resp = error_from_exception(exc, url="http://localhost:8088")
        assert resp.status_code == 502
        body = json.loads(resp.body)
        assert "not reachable" in body["error"]["message"]
        assert "localhost:8088" in body["error"]["message"]

    def test_connection_error(self):
        exc = ConnectionError("DNS failed")
        resp = error_from_exception(exc)
        assert resp.status_code == 502
        body = json.loads(resp.body)
        assert body["error"]["type"] == "bad_gateway"

    def test_timeout_error(self):
        exc = TimeoutError("timed out")
        resp = error_from_exception(exc, timeout=120)
        assert resp.status_code == 504
        body = json.loads(resp.body)
        assert "120" in body["error"]["message"]
        assert body["error"]["type"] == "gateway_timeout"

    def test_json_decode_error(self):
        exc = json.JSONDecodeError("bad json", "", 0)
        resp = error_from_exception(exc)
        assert resp.status_code == 502
        body = json.loads(resp.body)
        assert "Invalid response" in body["error"]["message"]

    def test_unknown_error_fallback(self):
        exc = RuntimeError("something weird")
        resp = error_from_exception(exc)
        assert resp.status_code == 500
        body = json.loads(resp.body)
        assert body["error"]["type"] == "internal_error"
        assert "RuntimeError" in body["error"]["message"]

    def test_missing_context_key_graceful(self):
        """If template expects {url} but not provided, it shouldn't crash."""
        exc = ConnectionRefusedError("refused")
        resp = error_from_exception(exc)  # No url= provided
        assert resp.status_code == 502
        # Should still return a valid response, not raise KeyError


class TestMakeStreamErrorChunk:
    def test_contains_error_message(self):
        result = make_stream_error_chunk("chatcmpl-abc", "Connection lost")
        assert "[Error: Connection lost]" in result

    def test_ends_with_done(self):
        result = make_stream_error_chunk("chatcmpl-abc", "Oops")
        assert result.strip().endswith("data: [DONE]")

    def test_has_finish_reason_stop(self):
        result = make_stream_error_chunk("chatcmpl-abc", "Fail")
        assert '"finish_reason": "stop"' in result

    def test_valid_sse_format(self):
        result = make_stream_error_chunk("chatcmpl-abc", "Error")
        lines = [l for l in result.split("\n") if l.startswith("data: ")]
        assert len(lines) == 2  # error chunk + [DONE]

        # First line should be valid JSON
        data = json.loads(lines[0][6:])
        assert data["id"] == "chatcmpl-abc"


class TestAuthError:
    def test_status_401(self):
        resp = auth_error()
        assert resp.status_code == 401

    def test_error_format(self):
        resp = auth_error()
        body = json.loads(resp.body)
        assert body["error"]["type"] == "auth_error"
        assert "Bearer token" in body["error"]["message"]
