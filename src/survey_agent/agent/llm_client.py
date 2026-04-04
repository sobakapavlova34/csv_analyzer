from __future__ import annotations

import os
from typing import Any

import httpx
from openai import OpenAI

from survey_agent.config import AgentConfig


def _http_client(*, trust_env: bool) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(120.0, connect=60.0),
        trust_env=trust_env,
    )


def make_openai_client(cfg: AgentConfig, *, ignore_system_proxy: bool | None = None) -> OpenAI:
    if ignore_system_proxy is None:
        ignore_system_proxy = os.environ.get("BOTHUB_IGNORE_SYSTEM_PROXY", "").lower() in (
            "1",
            "true",
            "yes",
        )
    te = not ignore_system_proxy
    return OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        http_client=_http_client(trust_env=te),
    )


def chat_with_tools(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> Any:
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.2,
    )
