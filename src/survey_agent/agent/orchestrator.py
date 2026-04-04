from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from survey_agent.agent.llm_client import chat_with_tools, make_openai_client
from survey_agent.agent.openai_tools import specs_to_openai_tools
from survey_agent.agent.prompts import SYSTEM_PROMPT, USER_TASK_TEMPLATE
from survey_agent.config import AgentConfig
from survey_agent.state import SessionState
from survey_agent.tools.runner import run_tool
from survey_agent.types import ToolCall


def _json_for_tool_message(obj: dict[str, Any], max_chars: int) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + "\n…[truncated]"


def run_llm_agent(
    state: SessionState,
    cfg: AgentConfig | None = None,
    *,
    csv_path_display: str | None = None,
) -> list[dict[str, Any]]:
    cfg = cfg or AgentConfig.from_env()
    client = make_openai_client(cfg)
    tools = specs_to_openai_tools()

    path_str = csv_path_display or str(state.csv_path)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TASK_TEMPLATE.format(csv_path=path_str)},
    ]

    transcript: list[dict[str, Any]] = [{"phase": "start", "csv": path_str}]

    for step in range(cfg.max_agent_steps):
        resp = chat_with_tools(
            client,
            model=cfg.model,
            messages=messages,
            tools=tools,
        )
        choice = resp.choices[0]
        msg = choice.message

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            transcript.append({"step": step, "assistant_text": msg.content})
            state.phase = "done"
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = run_tool(state, ToolCall(tool=name, arguments=args, call_id=tc.id))
            body = result.to_message_dict()
            content = _json_for_tool_message(body, cfg.max_tool_result_chars)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                }
            )
            transcript.append(
                {
                    "step": step,
                    "tool": name,
                    "ok": result.ok,
                    "summary": result.summary,
                }
            )

        if choice.finish_reason == "stop" and not msg.tool_calls:
            break
    else:
        state.phase = "max_steps"

    return transcript


def run_from_csv(
    csv_path: str | Path,
    cfg: AgentConfig | None = None,
) -> tuple[SessionState, list[dict[str, Any]]]:
    """создать состояние + прогнать агента"""
    path = Path(csv_path)
    state = SessionState(csv_path=path.resolve())
    transcript = run_llm_agent(state, cfg=cfg, csv_path_display=str(path))
    return state, transcript
