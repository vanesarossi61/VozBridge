"""Error handling: maps CoPaw/network errors to OpenAI error format.

All errors returned to the client follow the OpenAI error JSON schema
so that OpenClaw Assistant can parse them correctly.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from .models import OpenAIError, OpenAIErrorResponse

logger = logging.getLogger("voz_bridge")


# -- Error Mapping Table ----------------------------------------------

ERROR_MAP: dict[type, tuple[int, str, str]] = {
    ConnectionRefusedError: (502, "bad_gateway", "CoPaw server not reachable at {url}"),
    ConnectionError: (502, "bad_gateway", "Cannot connect to CoPaw: {detail}"),
    TimeoutError: (504, "gateway_timeout", "CoPaw did not respond within {timeout}s"),
    json.JSONDecodeError: (502, "bad_gateway", "Invalid response from CoPaw"),
}


def make_error_response(
    status_code: int,
    error_type: str,
    message: str,
) -> JSONResponse:
    """Build a JSONResponse in OpenAI error format."""
    body = OpenAIErrorResponse(
        error=OpenAIError(
            message=message,
            type=error_type,
            code=status_code,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
    )


def error_from_exception(
    exc: Exception,
    **context: Any,
) -> JSONResponse:
    """Map a Python exception to an OpenAI-format error response."""
    for exc_type, (status, etype, msg_template) in ERROR_MAP.items():
        if isinstance(exc, exc_type):
            try:
                message = msg_template.format(**context, detail=str(exc))
            except KeyError:
                message = msg_template.split("{")[0] + str(exc)
            logger.error("CoPaw error [%s]: %s", etype, message)
            return make_error_response(status, etype, message)

    # Fallback for unknown errors
    logger.exception("Unexpected bridge error: %s", exc)
    return make_error_response(
        500, "internal_error", f"Bridge internal error: {type(exc).__name__}"
    )


def make_stream_error_chunk(
    completion_id: str,
    message: str,
) -> str:
    """Build an SSE error chunk for mid-stream failures.

    Injects the error message as a final content chunk,
    then sends finish_reason=stop and [DONE].
    """
    import time
    from .models import ChatCompletionChunk, ChunkChoice, DeltaContent

    # Error content chunk
    error_chunk = ChatCompletionChunk(
        id=completion_id,
        created=int(time.time()),
        choices=[
            ChunkChoice(
                delta=DeltaContent(content=f"\n[Error: {message}]"),
                finish_reason="stop",
            )
        ],
    )

    lines = f"data: {error_chunk.model_dump_json()}\n\n"
    lines += "data: [DONE]\n\n"
    return lines


# -- Auth Error -------------------------------------------------------


def auth_error() -> JSONResponse:
    """Return a 401 Unauthorized in OpenAI error format."""
    return make_error_response(
        401,
        "auth_error",
        "Invalid or missing Bearer token",
    )
