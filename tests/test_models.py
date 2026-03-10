"""Tests for voz_bridge.models — Pydantic model validation."""

from __future__ import annotations

import pytest

from voz_bridge.models import (
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
    OpenAIError,
    OpenAIErrorResponse,
    OpenAIMessage,
    UsageInfo,
)


# ── OpenAI Models ────────────────────────────────────────────────────


class TestOpenAIMessage:
    def test_basic_message(self):
        msg = OpenAIMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_default_content_empty_string(self):
        msg = OpenAIMessage(role="system")
        assert msg.content == ""

    def test_list_content(self):
        content = [{"type": "text", "text": "Look at this"}]
        msg = OpenAIMessage(role="user", content=content)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1


class TestChatCompletionRequest:
    def test_defaults(self):
        req = ChatCompletionRequest(
            messages=[OpenAIMessage(role="user", content="Hi")]
        )
        assert req.model == "copaw"
        assert req.stream is True
        assert req.user is None
        assert req.temperature is None
        assert req.max_tokens is None

    def test_all_fields(self):
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[OpenAIMessage(role="user", content="Hi")],
            stream=False,
            user="user-123",
            temperature=0.7,
            max_tokens=1000,
        )
        assert req.model == "gpt-4"
        assert req.stream is False
        assert req.user == "user-123"
        assert req.temperature == 0.7
        assert req.max_tokens == 1000

    def test_empty_messages_allowed(self):
        req = ChatCompletionRequest(messages=[])
        assert req.messages == []


class TestStreamingChunkModels:
    def test_delta_content_defaults(self):
        delta = DeltaContent()
        assert delta.role is None
        assert delta.content is None

    def test_delta_with_role(self):
        delta = DeltaContent(role="assistant")
        assert delta.role == "assistant"

    def test_chunk_choice_defaults(self):
        choice = ChunkChoice()
        assert choice.index == 0
        assert choice.finish_reason is None

    def test_chunk_serialization(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-abc123",
            created=1700000000,
            choices=[
                ChunkChoice(
                    delta=DeltaContent(content="Hello"),
                )
            ],
        )
        data = chunk.model_dump()
        assert data["object"] == "chat.completion.chunk"
        assert data["model"] == "copaw"
        assert data["choices"][0]["delta"]["content"] == "Hello"


class TestNonStreamingModels:
    def test_completion_message_defaults(self):
        msg = CompletionMessage()
        assert msg.role == "assistant"
        assert msg.content == ""

    def test_completion_choice_defaults(self):
        choice = CompletionChoice()
        assert choice.index == 0
        assert choice.finish_reason == "stop"

    def test_usage_info_all_zeros(self):
        usage = UsageInfo()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_full_response(self):
        resp = ChatCompletionResponse(
            id="chatcmpl-xyz",
            created=1700000000,
            choices=[
                CompletionChoice(
                    message=CompletionMessage(content="Hello!"),
                )
            ],
        )
        data = resp.model_dump()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert data["usage"]["total_tokens"] == 0


# ── AgentScope Models ────────────────────────────────────────────────


class TestAgentScopeModels:
    def test_content_item_defaults(self):
        item = AgentScopeContentItem()
        assert item.type == "text"
        assert item.text == ""

    def test_content_item_custom(self):
        item = AgentScopeContentItem(type="image_url", text="http://img.jpg")
        assert item.type == "image_url"

    def test_message_structure(self):
        msg = AgentScopeMessage(
            role="user",
            content=[AgentScopeContentItem(text="Hello")],
        )
        assert msg.role == "user"
        assert len(msg.content) == 1

    def test_request_structure(self):
        req = AgentScopeRequest(
            input=[
                AgentScopeMessage(
                    role="user",
                    content=[AgentScopeContentItem(text="Hi")],
                )
            ],
            session_id="voz-abc123",
        )
        assert req.session_id == "voz-abc123"
        assert len(req.input) == 1


# ── Error Models ─────────────────────────────────────────────────────


class TestErrorModels:
    def test_openai_error(self):
        err = OpenAIError(
            message="Not found",
            type="not_found",
            code=404,
        )
        assert err.message == "Not found"
        assert err.param is None

    def test_error_response_wrapper(self):
        resp = OpenAIErrorResponse(
            error=OpenAIError(
                message="Bad request",
                type="invalid_request",
                code=400,
            )
        )
        data = resp.model_dump()
        assert data["error"]["message"] == "Bad request"
        assert data["error"]["type"] == "invalid_request"
