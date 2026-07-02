"""
Request models for POST /chat.

The API is stateless: every call carries the FULL conversation history.
No per-conversation state is stored server-side (doc: "Your service
stores no per-conversation state").
"""

from typing import Literal
from pydantic import BaseModel, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def validate_messages_non_empty(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages must contain at least one message")
        return v
