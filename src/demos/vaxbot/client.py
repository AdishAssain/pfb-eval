"""VaxBot demo client: loads the system prompt and delegates to the generic chat wrapper."""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.eval.llm_client import chat, chat_async
from src.eval.schemas import ChatResponse

SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"


def system_prompt_text() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def system_prompt_sha256() -> str:
    return hashlib.sha256(system_prompt_text().encode("utf-8")).hexdigest()


def vaxbot_chat(user_messages: list[dict[str, str]], model: str, **kwargs) -> ChatResponse:
    """Sync VaxBot chat. The system prompt is prepended automatically."""
    messages = [{"role": "system", "content": system_prompt_text()}, *user_messages]
    return chat(messages, model=model, **kwargs)


async def vaxbot_chat_async(
    user_messages: list[dict[str, str]],
    model: str,
    **kwargs,
) -> ChatResponse:
    """Async VaxBot chat. The system prompt is prepended automatically."""
    messages = [{"role": "system", "content": system_prompt_text()}, *user_messages]
    return await chat_async(messages, model=model, **kwargs)
