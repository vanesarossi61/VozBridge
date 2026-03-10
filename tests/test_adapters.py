"""Tests for voz_bridge.adapters — request/response translation."""

from __future__ import annotations

import json

import pytest

from voz_bridge.adapters import accumulate_response, adapt_request, adapt_sse_stream
from voz_bridge.models import ChatCompletionRequest, OpenAIMessage


# ── Request Adapter ──────────────────────────────────────────────────


class TestAdaptRequest:
    def test_string_content_becomes_list(self, simple_request):
        result = adapt_request(simple_request)
        # Each message content should be a list of AgentScopeContentItem
        for msg in result.input:
            assert isinstance(msg.content, list)
            assert len(msg.content) >= 1
            assert msg.content[0].type == "text"

    def test_message_roles_preserved(self, simple_request):
        result = adapt_request(simple_request)
        assert result.input[0].role == "system"
        assert result.input[1].role == "user"

    def test_content_text_preserved(self, simple_request):
        result = adapt_request(simple_request)
        assert result.input[0].content[0].text == "You are helpful."
        assert result.input[1].content[0].text == "Hello!"

    def test_user_in_session_id(self, simple_request):
        result = adapt_request(simple_request)
        assert result.session_id == "voz-test-user-42"

    def test_no_user_generates_random_session(self, minimal_request):
        result = adapt_request(minimal_request)
        assert result.session_id.startswith("voz-")
        assert len(result.session_id) > len("voz-")

    def test_custom_prefix(self, simple_request):
        result = adapt_request(simple_request, session_prefix="custom-")
        assert result.session_id == "custom-test-user-42"

    def test_list_content_passthrough(self, multi_content_request):
        result = adapt_request(multi_content_request)
        items = result.input[0].content
        assert len(items) == 2
        assert items[0].type == "text"
        assert items[0].text == "Describe this image"
        assert items[1].type == "image_url"

    def test_output_is_agentscope_request(self, simple_request):
        result = adapt_request(simple_request)
        assert hasattr(result, "input")
        assert hasattr(result, "session_id")

    def test_empty_messages(self):
        req = ChatCompletionRequest(messages=[])
        result = adapt_request(req)
        assert result.input == []


# ── SSE Stream Adapter ───────────────────────────────────────────────


class TestAdaptSSEStream:
    @pytest.mark.asyncio
    async def test_created_event_sends_role(self, mock_agentscope_stream):
        lines = ['data: {"object": "response", "status": "created"}']
        chunks = []
        async for chunk in adapt_sse_stream(mock_agentscope_stream(lines)):
            chunks.append(chunk)

        # Should have: role chunk + done
        assert any('"role": "assistant"' in c for c in chunks)

    @pytest.mark.asyncio
    async def test_content_events_forwarded(self, mock_agentscope_stream):
        lines = [
            'data: {"object": "response", "status": "created"}',
            'data: {"object": "content", "text": "Hello"}',
            'data: {"object": "content", "text": " world"}',
            'data: {"object": "response", "status": "completed"}',
            'data: [DONE]',
        ]
        chunks = []
        async for chunk in adapt_sse_stream(mock_agentscope_stream(lines)):
            chunks.append(chunk)

        # Check content chunks exist
        content_chunks = [c for c in chunks if '"content":' in c and "Hello" in c or "world" in c]
        assert len(content_chunks) >= 1

    @pytest.mark.asyncio
    async def test_done_signal_forwarded(self, mock_agentscope_stream):
        lines = ['data: [DONE]']
        chunks = []
        async for chunk in adapt_sse_stream(mock_agentscope_stream(lines)):
            chunks.append(chunk)

        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_completed_sends_stop(self, mock_agentscope_stream):
        lines = [
            'data: {"object": "response", "status": "completed"}',
            'data: [DONE]',
        ]
        chunks = []
        async for chunk in adapt_sse_stream(mock_agentscope_stream(lines)):
            chunks.append(chunk)

        assert any('"finish_reason": "stop"' in c for c in chunks)

    @pytest.mark.asyncio
    async def test_malformed_json_skipped(self, mock_agentscope_stream):
        lines = [
            'data: {"object": "response", "status": "created"}',
            'data: {INVALID JSON}',
            'data: {"object": "content", "text": "OK"}',
            'data: [DONE]',
        ]
        chunks = []
        async for chunk in adapt_sse_stream(mock_agentscope_stream(lines)):
            chunks.append(chunk)

        # Should not crash, content "OK" should appear
        assert any("OK" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_non_data_lines_ignored(self, mock_agentscope_stream):
        lines = [
            'event: ping',
            ': keepalive',
            'data: {"object": "content", "text": "Hi"}',
            'data: [DONE]',
        ]
        chunks = []
        async for chunk in adapt_sse_stream(mock_agentscope_stream(lines)):
            chunks.append(chunk)

        assert any("Hi" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_all_chunks_have_same_id(self, mock_agentscope_stream):
        lines = [
            'data: {"object": "response", "status": "created"}',
            'data: {"object": "content", "text": "A"}',
            'data: {"object": "content", "text": "B"}',
            'data: {"object": "response", "status": "completed"}',
            'data: [DONE]',
        ]
        chunks = []
        async for chunk in adapt_sse_stream(mock_agentscope_stream(lines)):
            chunks.append(chunk)

        ids = set()
        for c in chunks:
            if c.startswith("data: {"):
                data = json.loads(c[6:].strip())
                ids.add(data.get("id"))

        # All data chunks should share the same completion ID
        ids.discard(None)
        assert len(ids) == 1

    @pytest.mark.asyncio
    async def test_stream_always_ends_with_done(self, mock_agentscope_stream):
        """Even if source doesn't send [DONE], adapter appends it."""
        lines = [
            'data: {"object": "content", "text": "partial"}',
        ]
        chunks = []
        async for chunk in adapt_sse_stream(mock_agentscope_stream(lines)):
            chunks.append(chunk)

        assert chunks[-1] == "data: [DONE]\n\n"


# ── Non-Streaming Accumulator ────────────────────────────────────────


class TestAccumulateResponse:
    @pytest.mark.asyncio
    async def test_accumulates_text(self, mock_agentscope_stream):
        lines = [
            'data: {"object": "response", "status": "created"}',
            'data: {"object": "content", "text": "Hello"}',
            'data: {"object": "content", "text": " world"}',
            'data: {"object": "response", "status": "completed"}',
            'data: [DONE]',
        ]
        resp = await accumulate_response(mock_agentscope_stream(lines))

        assert resp.choices[0].message.content == "Hello world"
        assert resp.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_empty_stream(self, mock_agentscope_stream):
        lines = ['data: [DONE]']
        resp = await accumulate_response(mock_agentscope_stream(lines))
        assert resp.choices[0].message.content == ""

    @pytest.mark.asyncio
    async def test_response_format(self, mock_agentscope_stream):
        lines = [
            'data: {"object": "content", "text": "Test"}',
            'data: [DONE]',
        ]
        resp = await accumulate_response(mock_agentscope_stream(lines))

        assert resp.object == "chat.completion"
        assert resp.model == "copaw"
        assert resp.id.startswith("chatcmpl-")
        assert resp.usage.total_tokens == 0

    @pytest.mark.asyncio
    async def test_skips_non_content_events(self, mock_agentscope_stream):
        lines = [
            'data: {"object": "response", "status": "created"}',
            'data: {"object": "content", "text": "Only this"}',
            'data: {"object": "response", "status": "completed"}',
            'data: [DONE]',
        ]
        resp = await accumulate_response(mock_agentscope_stream(lines))
        assert resp.choices[0].message.content == "Only this"
