"""
Предобработка: типы столбцов, пропуски, кодирование категорий.
Устойчиво к пустым строкам, смешанным типам, мусору в object-столбцах.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def _empty_to_na(s: pd.Series) -> pd.Series:
    if s.dtype == object or str(s.dtype).startswith("string"):
        s = s.astype("string")
        s = s.str.strip()
        s = s.replace("", pd.NA)
    return s


def _coerce_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(s.astype(str).str.strip().replace("", np.nan), errors="coerce")


def _try_datetime(s: pd.Series) -> pd.Series | None:
    if len(s.dropna()) < 10:
        return None
    try:
        parsed = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
        if parsed.notna().mean() > 0.85:
            return parsed
    except Exception:
        pass
    return None


def classify_column(series: pd.Series, col: str) -> tuple[str, dict[str, Any]]:
    """Возвращает роль столбца и диагностику."""
    s = _empty_to_na(series.copy())
    n = len(s)
    meta: dict[str, Any] = {"n_rows": n}

    dt = _try_datetime(s)
    if dt is not None:
        return "datetime", {**meta, "role": "datetime", "missing_share": float(dt.isna().mean())}

    num = _coerce_numeric(s)
    num_share = float(num.notna().mean()) if n else 0.0
    nunique_num = num.dropna().nunique()

    # эвристика: «числовой» столбец
    if num_share >= 0.5 and nunique_num >= 3:
        return "numeric", {
            **meta,
            "role": "numeric",
            "coerced_non_null_share": num_share,
            "n_unique_numeric": int(nunique_num),
            "missing_share": float(num.isna().mean()),
        }

    # категориальный
    cat = s.astype(str)
    cat = cat.replace({"<NA>": np.nan, "nan": np.nan, "None": np.nan, "": np.nan})
    return "categorical", {
        **meta,
        "role": "categorical",
        "missing_share": float(cat.isna().mean()),
        "n_unique": int(cat.dropna().nunique()),
    }


def run_preprocess(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Строит очищенную таблицу и JSON-отчёт.

    Пишет в ``out_dir``:
    - ``report.json`` — типы, стратегии, карты кодирования
    - ``clean.csv`` — читаемые значения (категории строками, числа float)
    - ``encoded_features.csv`` — только числовые + label-encoded категории (для ML/EDA)
    - ``column_roles.json`` — краткая классификация
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _ = random_seed

    roles: dict[str, str] = {}
    per_col: dict[str, Any] = {}
    clean: dict[str, pd.Series] = {}
    encoded: dict[str, pd.Series] = {}
    encoders: dict[str, dict[str, int]] = {}

    for col in df.columns:
        try:
            role, role_info = classify_column(df[col], col)
        except Exception as e:
            logger.exception("column %s", col)
            role = "categorical"
            role_info = {"role": "categorical", "error": str(e)}

        s = _empty_to_na(df[col].copy())

        if role == "datetime":
            dt = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
            clean[col] = dt
            dtu = pd.to_datetime(dt, utc=True, errors="coerce")
            enc = dtu.astype(np.int64).astype(np.float64) / 1e9
            encoded[col] = enc
            roles[col] = "datetime"
            per_col[col] = {**role_info, "imputation": "none_datetime"}

        elif role == "numeric":
            num = _coerce_numeric(s)
            med = float(np.nanmedian(num)) if num.notna().any() else 0.0
            filled = num.fillna(med)
            clean[col] = filled
            encoded[col] = filled
            roles[col] = "numeric"
            per_col[col] = {
                **role_info,
                "imputation": "median",
                "imputation_value": med,
            }

        else:
            cat = s.astype(str).str.strip()
            cat = cat.replace({"<NA>": np.nan, "nan": np.nan, "None": np.nan, "": np.nan})
            mode = cat.mode(dropna=True)
            fill_val = str(mode.iloc[0]) if len(mode) else "__MISSING__"
            cat_filled = cat.fillna(fill_val)
            # убрать оставшиеся пустые строки
            cat_filled = cat_filled.replace("", fill_val)
            clean[col] = cat_filled

            le = LabelEncoder()
            cat_str = cat_filled.astype(str)
            uniq = sorted(cat_str.unique(), key=str)
            le.fit(uniq)
            codes = le.transform(cat_str)
            encoded[col] = pd.Series(codes, index=cat_filled.index, dtype=np.int64)
            mapping = {str(l): int(i) for i, l in enumerate(le.classes_)}
            encoders[col] = mapping
            roles[col] = "categorical"
            per_col[col] = {
                **role_info,
                "imputation": "mode_or_missing_label",
                "imputation_value": fill_val,
                "n_codes": len(mapping),
            }

    df_clean = pd.DataFrame(clean)
    df_enc = pd.DataFrame(encoded)

    report: dict[str, Any] = {
        "n_rows": int(len(df_clean)),
        "n_columns": int(len(df_clean.columns)),
        "column_roles": roles,
        "per_column": per_col,
        "label_encoders": encoders,
    }

    df_clean.to_csv(out_dir / "clean.csv", index=False, encoding="utf-8")
    df_enc.to_csv(out_dir / "encoded_features.csv", index=False, encoding="utf-8")
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "column_roles.json").write_text(
        json.dumps(roles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log_path = out_dir.parent / "logs" / "preprocess.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"ok rows={len(df_clean)} cols={len(df_clean.columns)}\n")
        f.write(json.dumps(roles, ensure_ascii=False, indent=2))

    return df_clean, df_enc, report
