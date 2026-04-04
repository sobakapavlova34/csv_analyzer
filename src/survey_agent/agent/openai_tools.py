from __future__ import annotations

import copy
from typing import Any

from survey_agent.tools.definitions import TOOL_SPECS


def specs_to_openai_tools(specs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    specs = specs or TOOL_SPECS
    out: list[dict[str, Any]] = []
    for s in specs:
        params = copy.deepcopy(s["parameters"])
        # OpenAI ожидает additionalProperties: false для strict в некоторых режимах; для совместимости оставляем как есть
        out.append(
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": params,
                },
            }
        )
    return out
