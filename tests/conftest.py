"""Shared fixtures for voz-bridge tests."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from voz_bridge.models import ChatCompletionRequest, OpenAIMessage


@pytest.fixture
def simple_request() -> ChatCompletionRequest:
    """Minimal valid OpenAI chat completion request."""
    return ChatCompletionRequest(
        model="copaw",
        messages=[
            OpenAIMessage(role="system", content="You are helpful."),
            OpenAIMessage(role="user", content="Hello!"),
        ],
        stream=True,
        user="test-user-42",
    )


@pytest.fixture
def multi_content_request() -> ChatCompletionRequest:
    """Request with list-type content (multimodal format)."""
    return ChatCompletionRequest(
        model="copaw",
        messages=[
            OpenAIMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "text": ""},
                ],
            ),
        ],
        stream=True,
    )


@pytest.fixture
def minimal_request() -> ChatCompletionRequest:
    """Request with only required fields, no user."""
    return ChatCompletionRequest(
        messages=[OpenAIMessage(role="user", content="Hi")],
    )


async def _lines_from(lines: list[str]) -> AsyncIterator[str]:
    """Helper: convert a list of strings to an async iterator."""
    for line in lines:
        yield line


@pytest.fixture
def mock_agentscope_stream():
    """Factory fixture: returns an async iterator from a list of SSE lines."""
    def _factory(lines: list[str]) -> AsyncIterator[str]:
        return _lines_from(lines)
    return _factory
