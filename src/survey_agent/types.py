from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ToolName = Literal[
    "dataset_profile",
    "column_snapshot",
    "numeric_correlation",
    "ttest_groups",
    "mannwhitney_groups",
    "chi_square_independence",
    "cramers_v",
    "kmeans_cluster_summary",
    "pca_projection_2d",
    "anova_numeric_by_category",
]


@dataclass
class ToolCall:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None


@dataclass
class ToolResult:
    tool: str
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    summary: str = "" # для LLM

    def to_message_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "summary": self.summary,
            "payload": self.payload if self.ok else {},
            "error": self.error,
        }
