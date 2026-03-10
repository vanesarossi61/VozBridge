"""Bidirectional adapters: OpenAI <-> AgentScope.

RequestAdapter:  Translates OpenAI messages[] -> AgentScope input[]
ResponseAdapter: Translates AgentScope SSE stream -> OpenAI SSE chunks (zero buffering)
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

from .models import (
    AgentScopeContentItem,
    AgentScopeMessage,
    AgentScopeRequest,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChunkChoice,
    CompletionChoice,
    CompletionMessage,
    DeltaContent,
    UsageInfo,
)


def _gen_completion_id() -> str:
    """Generate a unique completion ID in OpenAI format."""
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


# -- Request Adapter --------------------------------------------------


def adapt_request(
    openai_req: ChatCompletionRequest,
    session_prefix: str = "voz-",
) -> AgentScopeRequest:
    """Translate an OpenAI ChatCompletion request to AgentScope format.

    - content string -> [{type: "text", text: content}]
    - messages -> input
    - user -> session_id (with prefix)
    """
    input_messages: list[AgentScopeMessage] = []

    for msg in openai_req.messages:
        if isinstance(msg.content, str):
            content_items = [AgentScopeContentItem(type="text", text=msg.content)]
        elif isinstance(msg.content, list):
            # Already structured content -- pass through
            content_items = [
                AgentScopeContentItem(type=item.get("type", "text"), text=item.get("text", ""))
                if isinstance(item, dict)
                else AgentScopeContentItem(type="text", text=str(item))
                for item in msg.content
            ]
        else:
            content_items = [AgentScopeContentItem(type="text", text=str(msg.content))]

        input_messages.append(
            AgentScopeMessage(role=msg.role, content=content_items)
        )

    session_id = (
        f"{session_prefix}{openai_req.user}"
        if openai_req.user
        else f"{session_prefix}{uuid.uuid4().hex[:8]}"
    )

    return AgentScopeRequest(input=input_messages, session_id=session_id)


# -- Response Adapter (SSE Streaming) ---------------------------------


def _build_chunk(
    completion_id: str,
    delta: DeltaContent,
    finish_reason: str | None = None,
) -> str:
    """Build a single OpenAI SSE chunk as a data line."""
    chunk = ChatCompletionChunk(
        id=completion_id,
        created=int(time.time()),
        choices=[
            ChunkChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason,
            )
        ],
    )
    return f"data: {chunk.model_dump_json()}\n\n"


async def adapt_sse_stream(
    agentscope_stream: AsyncIterator[str],
) -> AsyncIterator[str]:
    """Translate AgentScope SSE events to OpenAI SSE chunks.

    Zero buffering: each AgentScope event is translated and yielded
    immediately as an OpenAI-format SSE data line.
    """
    completion_id = _gen_completion_id()
    first_content = True

    async for line in agentscope_stream:
        line = line.strip()

        if not line.startswith("data: "):
            continue

        raw = line[6:].strip()

        # Pass through [DONE] signal
        if raw == "[DONE]":
            yield "data: [DONE]\n\n"
            break

        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue  # Skip malformed events

        obj_type = event.get("object", "")
        status = event.get("status", "")

        if obj_type == "response":
            if status == "created":
                # First chunk: send role declaration
                yield _build_chunk(
                    completion_id,
                    DeltaContent(role="assistant"),
                )
            elif status == "completed":
                # Final chunk: send finish reason
                yield _build_chunk(
                    completion_id,
                    DeltaContent(),
                    finish_reason="stop",
                )

        elif obj_type == "content":
            text = event.get("text", "")
            if text:
                yield _build_chunk(
                    completion_id,
                    DeltaContent(content=text),
                )

    # Always close with [DONE] if stream ends without it
    yield "data: [DONE]\n\n"


# -- Non-Streaming Accumulator ----------------------------------------


async def accumulate_response(
    agentscope_stream: AsyncIterator[str],
) -> ChatCompletionResponse:
    """Consume the full SSE stream and build a non-streaming response.

    Used when the client sends stream=false. CoPaw always returns SSE,
    so we accumulate all text chunks into a single response.
    """
    completion_id = _gen_completion_id()
    accumulated_text = []

    async for line in agentscope_stream:
        line = line.strip()
        if not line.startswith("data: "):
            continue
        raw = line[6:].strip()
        if raw == "[DONE]":
            break

        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if event.get("object") == "content" and event.get("text"):
            accumulated_text.append(event["text"])

    full_text = "".join(accumulated_text)

    return ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        choices=[
            CompletionChoice(
                message=CompletionMessage(role="assistant", content=full_text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(),
    )
