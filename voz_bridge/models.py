"""Pydantic models for OpenAI and AgentScope request/response formats.

These models define the exact wire format for both protocols,
enabling type-safe translation in the adapters.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# -- OpenAI Chat Completions (what OpenClaw sends / expects) ----------


class OpenAIMessage(BaseModel):
    """Single message in OpenAI format."""
    role: str
    content: str | list[Any] = ""


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible /v1/chat/completions request."""
    model: str = "copaw"
    messages: list[OpenAIMessage]
    stream: bool = True
    user: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class DeltaContent(BaseModel):
    """Delta object in a streaming chunk."""
    role: str | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    """Single choice in a streaming chunk."""
    index: int = 0
    delta: DeltaContent = Field(default_factory=DeltaContent)
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """Single SSE chunk in OpenAI streaming format."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str = "copaw"
    choices: list[ChunkChoice] = Field(default_factory=list)


class CompletionMessage(BaseModel):
    """Message in a non-streaming completion response."""
    role: str = "assistant"
    content: str = ""


class CompletionChoice(BaseModel):
    """Single choice in a non-streaming response."""
    index: int = 0
    message: CompletionMessage = Field(default_factory=CompletionMessage)
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    """Token usage info (placeholder, CoPaw doesn't report tokens)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Non-streaming /v1/chat/completions response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str = "copaw"
    choices: list[CompletionChoice] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)


# -- AgentScope / CoPaw format (what CoPaw expects / returns) ---------


class AgentScopeContentItem(BaseModel):
    """Single content item in AgentScope format."""
    type: str = "text"
    text: str = ""


class AgentScopeMessage(BaseModel):
    """Single message in AgentScope input format."""
    role: str
    content: list[AgentScopeContentItem]


class AgentScopeRequest(BaseModel):
    """CoPaw /api/agent/process request body."""
    input: list[AgentScopeMessage]
    session_id: str


class OpenAIError(BaseModel):
    """OpenAI-compatible error response body."""
    message: str
    type: str
    param: str | None = None
    code: int | None = None


class OpenAIErrorResponse(BaseModel):
    """Wrapper for OpenAI error format."""
    error: OpenAIError
