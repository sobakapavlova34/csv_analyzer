"""Диспетчер: ToolCall + SessionState → ToolResult."""

from __future__ import annotations

import traceback

from survey_agent.state import SessionState
from survey_agent.tools import implementations as impl
from survey_agent.types import ToolCall, ToolResult

_HANDLERS = {
    "dataset_profile": impl.tool_dataset_profile,
    "column_snapshot": impl.tool_column_snapshot,
    "numeric_correlation": impl.tool_numeric_correlation,
    "ttest_groups": impl.tool_ttest_groups,
    "mannwhitney_groups": impl.tool_mannwhitney_groups,
    "chi_square_independence": impl.tool_chi_square_independence,
    "cramers_v": impl.tool_cramers_v,
    "kmeans_cluster_summary": impl.tool_kmeans_cluster_summary,
    "pca_projection_2d": impl.tool_pca_projection_2d,
    "anova_numeric_by_category": impl.tool_anova_numeric_by_category,
}


def run_tool(state: SessionState, call: ToolCall) -> ToolResult:
    name = call.tool
    if name not in _HANDLERS:
        return ToolResult(
            tool=name,
            ok=False,
            error=f"Неизвестный инструмент: {name}",
            summary=f"unknown tool {name}",
        )
    try:
        payload = _HANDLERS[name](state, call.arguments)
        summary = _auto_summary(name, payload)
        entry = {
            "tool": name,
            "arguments": call.arguments,
            "ok": True,
            "summary": summary,
        }
        state.tool_log.append(entry)
        return ToolResult(tool=name, ok=True, payload=payload, summary=summary)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        state.tool_log.append(
            {
                "tool": name,
                "arguments": call.arguments,
                "ok": False,
                "error": err,
                "traceback": tb,
            }
        )
        return ToolResult(
            tool=name,
            ok=False,
            error=err,
            summary=err,
            payload={"traceback": tb},
        )


def _auto_summary(name: str, payload: dict) -> str:
    if name == "dataset_profile":
        return f"rows={payload['n_rows']} cols={payload['n_columns']}"
    if name == "column_snapshot":
        return f"{payload['column']}: kind={payload.get('kind')}, missing={payload.get('missing_share')}"
    if name == "numeric_correlation":
        used = payload.get("columns_used", [])
        return f"corr matrix for {len(used)} columns"
    if name == "ttest_groups":
        return f"t-test p={payload.get('pvalue', 'n/a'):.4g}" if isinstance(payload.get("pvalue"), float) else "t-test"
    if name == "mannwhitney_groups":
        return f"MW p={payload.get('pvalue', 'n/a'):.4g}" if isinstance(payload.get("pvalue"), float) else "MW"
    if name in ("chi_square_independence",):
        return f"chi2 p={payload.get('pvalue', 'n/a'):.4g}" if isinstance(payload.get("pvalue"), float) else "chi2"
    if name == "cramers_v":
        return f"Cramér V={payload.get('cramers_v', 'n/a')}"
    if name == "kmeans_cluster_summary":
        return f"k-means k={payload.get('n_clusters')}, n={payload.get('n_used')}"
    if name == "pca_projection_2d":
        ev = payload.get("explained_variance_ratio", [])
        return f"PCA 2D var={ev[:2] if ev else '?'}"
    if name == "anova_numeric_by_category":
        return f"ANOVA p={payload.get('pvalue', 'n/a'):.4g}" if isinstance(payload.get("pvalue"), float) else "ANOVA"
    return name
