"""Voz Bridge - FastAPI application.

Endpoints:
  POST /v1/chat/completions  - Main bridge (streaming + non-streaming)
  GET  /v1/models            - Model discovery for OpenClaw
  GET  /health               - Health check with CoPaw connectivity
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .adapters import accumulate_response, adapt_request, adapt_sse_stream
from .config import BridgeConfig, load_config
from .errors import auth_error, error_from_exception, make_stream_error_chunk

logger = logging.getLogger("voz_bridge")

# -- Globals (set during lifespan) ------------------------------------

config: BridgeConfig
http_client: httpx.AsyncClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage httpx client lifecycle."""
    global config, http_client
    config = load_config()

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    http_client = httpx.AsyncClient(timeout=config.copaw_timeout)
    logger.info(
        "Voz Bridge started on %s:%d -> CoPaw at %s",
        config.host,
        config.port,
        config.copaw_url,
    )
    yield
    await http_client.aclose()
    logger.info("Voz Bridge stopped")


app = FastAPI(
    title="Voz Bridge",
    version="1.0.0",
    description="CoPaw <-> OpenClaw Assistant API Bridge",
    lifespan=lifespan,
)


# -- Auth Check -------------------------------------------------------


def check_auth(request: Request) -> bool:
    """Validate Bearer token if auth is enabled."""
    if not config.auth_enabled:
        return True
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    return auth_header[7:].strip() == config.token


# -- POST /v1/chat/completions ----------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Main bridge endpoint: translates OpenAI -> CoPaw -> OpenAI."""
    # Auth
    if not check_auth(request):
        return auth_error()

    # Parse request
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Invalid JSON body", "type": "invalid_request"}},
        )

    from .models import ChatCompletionRequest

    try:
        openai_req = ChatCompletionRequest(**body)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid request: {e}", "type": "invalid_request"}},
        )

    # Translate request
    copaw_req = adapt_request(openai_req, session_prefix=config.session_prefix)
    is_streaming = openai_req.stream

    # Forward to CoPaw
    try:
        copaw_response = await http_client.post(
            config.copaw_process_endpoint,
            json=copaw_req.model_dump(),
            headers={"Accept": "text/event-stream"},
        )
        copaw_response.raise_for_status()
    except httpx.ConnectError:
        return error_from_exception(
            ConnectionRefusedError(),
            url=config.copaw_url,
        )
    except httpx.TimeoutException:
        return error_from_exception(
            TimeoutError(),
            timeout=config.copaw_timeout,
        )
    except Exception as e:
        return error_from_exception(e)

    # Stream CoPaw SSE response lines
    async def copaw_line_stream() -> AsyncIterator[str]:
        async for line in copaw_response.aiter_lines():
            yield line

    if is_streaming:
        # Streaming mode: translate chunk-by-chunk
        async def openai_stream():
            try:
                async for chunk in adapt_sse_stream(copaw_line_stream()):
                    yield chunk
            except Exception as e:
                logger.error("Mid-stream error: %s", e)
                yield make_stream_error_chunk(
                    f"chatcmpl-error-{int(time.time())}",
                    str(e),
                )

        return StreamingResponse(
            openai_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming mode: accumulate all chunks
        try:
            response = await accumulate_response(copaw_line_stream())
            return JSONResponse(content=response.model_dump())
        except Exception as e:
            return error_from_exception(e)


# -- GET /v1/models ---------------------------------------------------


@app.get("/v1/models")
async def list_models():
    """Return available models (CoPaw as single model)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "copaw",
                "object": "model",
                "created": 1700000000,
                "owned_by": "agentscope",
            }
        ],
    }


# -- GET /health ------------------------------------------------------


@app.get("/health")
async def health_check():
    """Health check with CoPaw connectivity probe."""
    copaw_status = "unknown"
    try:
        resp = await http_client.get(f"{config.copaw_url}/", timeout=5)
        copaw_status = "connected" if resp.status_code < 500 else "error"
    except Exception:
        copaw_status = "unreachable"

    return {
        "status": "ok",
        "copaw": copaw_status,
        "copaw_url": config.copaw_url,
        "bridge_version": "1.0.0",
        "auth_enabled": config.auth_enabled,
    }


# -- Entry point ------------------------------------------------------


def run():
    """CLI entry point for `voz-bridge` command."""
    cfg = load_config()
    uvicorn.run(
        "voz_bridge.main:app",
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level,
    )


if __name__ == "__main__":
    run()
