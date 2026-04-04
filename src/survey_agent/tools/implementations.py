"""Реализации инструментов (pandas / scipy / sklearn)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from survey_agent.state import SessionState


def _series_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")


def _infer_col_kind(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    n = len(s)
    if n == 0:
        return "unknown"
    num = _series_numeric(s)
    non_null = num.notna().sum()
    if non_null / max(n, 1) >= 0.7 and np.isfinite(num.dropna()).all():
        return "numeric"
    un = s.dropna().astype(str).str.strip().nunique()
    if un <= min(30, max(5, n // 20)):
        return "categorical"
    return "mixed"


def tool_dataset_profile(state: SessionState, _: dict[str, Any]) -> dict[str, Any]:
    df = state.ensure_df()
    rows, cols = df.shape
    missing = df.isna().mean().round(4).to_dict()
    kinds: dict[str, str] = {}
    for c in df.columns:
        k = _infer_col_kind(df[c])
        kinds[c] = k
    state.column_kind = kinds
    return {
        "n_rows": int(rows),
        "n_columns": int(cols),
        "columns": list(df.columns),
        "missing_share_by_column": missing,
        "column_kind": kinds,
    }


def tool_column_snapshot(state: SessionState, args: dict[str, Any]) -> dict[str, Any]:
    col = args["column"]
    top_n = int(args.get("top_n", 8))
    df = state.ensure_df()
    if col not in df.columns:
        raise KeyError(f"Нет столбца: {col}")
    s = df[col]
    out: dict[str, Any] = {
        "column": col,
        "dtype": str(s.dtype),
        "missing_count": int(s.isna().sum()),
        "missing_share": round(float(s.isna().mean()), 4),
        "kind": _infer_col_kind(s),
    }
    num = _series_numeric(s)
    if num.notna().sum() > 0:
        out["numeric_coerced"] = {
            "min": float(np.nanmin(num)),
            "max": float(np.nanmax(num)),
            "mean": float(np.nanmean(num)),
            "median": float(np.nanmedian(num)),
        }
    vc = s.astype(str).str.strip().value_counts().head(top_n)
    out["top_values"] = {str(k): int(v) for k, v in vc.items()}
    return out


def _pick_numeric_columns(df: pd.DataFrame, columns: list[str] | None, min_share: float) -> list[str]:
    names = columns if columns else list(df.columns)
    picked: list[str] = []
    for c in names:
        if c not in df.columns:
            continue
        num = _series_numeric(df[c])
        if num.notna().mean() >= min_share and num.notna().sum() >= 10:
            picked.append(c)
    return picked


def tool_numeric_correlation(state: SessionState, args: dict[str, Any]) -> dict[str, Any]:
    df = state.ensure_df()
    cols = args.get("columns") or []
    if isinstance(cols, str):
        cols = [cols]
    min_share = float(args.get("min_non_null_share", 0.5))
    picked = _pick_numeric_columns(df, cols if cols else None, min_share)
    if len(picked) < 2:
        return {"columns_used": picked, "message": "Недостаточно числовых столбцов после coercing"}
    sub = pd.DataFrame({c: _series_numeric(df[c]) for c in picked})
    corr = sub.corr(method="pearson", min_periods=20)
    # JSON: replace nan with None
    return {
        "columns_used": picked,
        "correlation_matrix": corr.round(4).where(pd.notna(corr), None).to_dict(),
    }


def _two_group_mask(df: pd.DataFrame, gcol: str, a: str, b: str) -> tuple[pd.Series, pd.Series]:
    g = df[gcol].astype(str).str.strip()
    m1 = g == str(a).strip()
    m2 = g == str(b).strip()
    return m1, m2


def tool_ttest_groups(state: SessionState, args: dict[str, Any]) -> dict[str, Any]:
    df = state.ensure_df()
    ncol = args["numeric_column"]
    gcol = args["group_column"]
    m1, m2 = _two_group_mask(df, gcol, args["group_a"], args["group_b"])
    x = _series_numeric(df[ncol])
    a = x[m1].dropna()
    b = x[m2].dropna()
    if len(a) < 3 or len(b) < 3:
        raise ValueError("Мало наблюдений в одной из групп")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "statistic": float(t),
        "pvalue": float(p),
    }


def tool_mannwhitney_groups(state: SessionState, args: dict[str, Any]) -> dict[str, Any]:
    df = state.ensure_df()
    ncol = args["numeric_column"]
    gcol = args["group_column"]
    m1, m2 = _two_group_mask(df, gcol, args["group_a"], args["group_b"])
    x = _series_numeric(df[ncol])
    a = x[m1].dropna()
    b = x[m2].dropna()
    if len(a) < 3 or len(b) < 3:
        raise ValueError("Мало наблюдений в одной из групп")
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "median_a": float(a.median()),
        "median_b": float(b.median()),
        "statistic": float(u),
        "pvalue": float(p),
    }


def tool_chi_square_independence(state: SessionState, args: dict[str, Any]) -> dict[str, Any]:
    df = state.ensure_df()
    a, b = args["column_a"], args["column_b"]
    ct = pd.crosstab(df[a].astype(str).str.strip(), df[b].astype(str).str.strip())
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return {
        "chi2": float(chi2),
        "pvalue": float(p),
        "dof": int(dof),
        "table_shape": list(ct.shape),
    }


def tool_cramers_v(state: SessionState, args: dict[str, Any]) -> dict[str, Any]:
    df = state.ensure_df()
    a, b = args["column_a"], args["column_b"]
    ct = pd.crosstab(df[a].astype(str).str.strip(), df[b].astype(str).str.strip())
    chi2, _, _, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    v = np.sqrt(chi2 / (n * (min(k - 1, r - 1))))
    return {"cramers_v": float(v), "n": int(n)}


def tool_kmeans_cluster_summary(state: SessionState, args: dict[str, Any]) -> dict[str, Any]:
    df = state.ensure_df()
    feats = list(args["feature_columns"])
    n_clusters = int(args.get("n_clusters", 4))
    std = bool(args.get("standardize", True))
    rng = state.random_seed
    X = pd.DataFrame({c: _series_numeric(df[c]) for c in feats}).dropna()
    if len(X) < n_clusters * 5:
        raise ValueError("Слишком мало полных строк для k-means")
    arr = X.values
    if std:
        arr = StandardScaler().fit_transform(arr)
    km = KMeans(n_clusters=n_clusters, random_state=rng, n_init=10)
    labels = km.fit_predict(arr)
    sizes = pd.Series(labels).value_counts().sort_index()
    summary = []
    for k in range(n_clusters):
        mask = labels == k
        means = {feats[j]: float(X.values[mask, j].mean()) for j in range(len(feats))}
        summary.append({"cluster": int(k), "size": int(mask.sum()), "feature_means_raw": means})
    return {
        "n_used": int(len(X)),
        "n_clusters": n_clusters,
        "cluster_sizes": sizes.to_dict(),
        "clusters": summary,
        "inertia": float(km.inertia_),
    }


def tool_pca_projection_2d(state: SessionState, args: dict[str, Any]) -> dict[str, Any]:
    df = state.ensure_df()
    feats = list(args["feature_columns"])
    sample_rows = int(args.get("sample_rows", 500))
    std = bool(args.get("standardize", True))
    X = pd.DataFrame({c: _series_numeric(df[c]) for c in feats}).dropna()
    if len(X) < 10:
        raise ValueError("Мало строк после dropna")
    Xs = X
    if len(X) > sample_rows:
        Xs = X.sample(sample_rows, random_state=state.random_seed)
    arr = Xs.values
    if std:
        arr = StandardScaler().fit_transform(arr)
    pca = PCA(n_components=2, random_state=state.random_seed)
    xy = pca.fit_transform(arr)
    return {
        "n_used": int(len(Xs)),
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "coordinates_sample": [{"x": float(xy[i, 0]), "y": float(xy[i, 1])} for i in range(len(xy))],
    }


def tool_anova_numeric_by_category(state: SessionState, args: dict[str, Any]) -> dict[str, Any]:
    df = state.ensure_df()
    ncol = args["numeric_column"]
    ccol = args["category_column"]
    x = _series_numeric(df[ncol])
    g = df[ccol].astype(str).str.strip()
    groups = []
    for _, sub in df.assign(_x=x, _g=g).dropna(subset=["_x"]).groupby("_g")["_x"]:
        arr = sub.values
        if len(arr) >= 2:
            groups.append(arr)
    if len(groups) < 2:
        raise ValueError("Нужно минимум 2 группы с данными")
    f, p = stats.f_oneway(*groups)
    return {
        "n_groups": len(groups),
        "group_sizes": [len(g_) for g_ in groups],
        "f_statistic": float(f),
        "pvalue": float(p),
    }
