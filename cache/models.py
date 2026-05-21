import time
import uuid
from typing import Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass


class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stop: Optional[list[str] | str] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage
    x_cache_status: Optional[str] = None


@dataclass
class CacheHit:
    id: str
    response_text: str
    response_metadata: dict
    similarity: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_openai_response(self, model: str) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            model=model,
            choices=[
                Choice(
                    message=Message(role="assistant", content=self.response_text),
                    finish_reason=self.response_metadata.get("finish_reason", "stop"),
                )
            ],
            usage=Usage(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.total_tokens,
            ),
            x_cache_status="hit",
        )
