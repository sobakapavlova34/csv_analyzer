from __future__ import annotations

import os
from dataclasses import dataclass

@dataclass
class AgentConfig:

    api_key: str
    base_url: str 
    model: str = "gpt-4o-mini"
    max_agent_steps: int = 16
    max_tool_result_chars: int = 12000

    @classmethod
    def from_env(cls) -> AgentConfig:
        key = os.environ.get("BOTHUB_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "нужен BOTHUB_API_KEY в окружении (или OPENAI_API_KEY для совместимости)"
            )
        base = os.environ.get("BOTHUB_BASE_URL")
        model = os.environ.get("BOTHUB_MODEL", "gpt-4o-mini")
        max_steps = int(os.environ.get("AGENT_MAX_STEPS", "16"))
        max_chars = int(os.environ.get("AGENT_MAX_TOOL_CHARS", "12000"))
        return cls(
            api_key=key,
            base_url=base,
            model=model,
            max_agent_steps=max_steps,
            max_tool_result_chars=max_chars,
        )
