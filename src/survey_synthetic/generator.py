from __future__ import annotations

import numpy as np
import pandas as pd

from survey_synthetic.schema import (
    CAT_POOLS_WIDE,
    DEFAULT_N_COLS,
    DEFAULT_N_ROWS,
    DEFAULT_SEED,
    EDUCATION_LABELS,
    GENDER_LABELS,
    INCOME_SUBJ_LABELS,
    REGION_LABELS,
    SETTLEMENT_LABELS,
    TRUST_COLUMNS,
)


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _latent_profile(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0, 1, size=n)


def _trust_values(n: int, skepticism: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_inst = len(TRUST_COLUMNS)
    loadings = rng.uniform(0.4, 0.9, size=n_inst)
    noise = rng.normal(0, 0.85, size=(n, n_inst))
    latent = loadings * skepticism[:, None] + noise
    # map to 1..4
    bins = np.quantile(latent, [0.25, 0.5, 0.75])
    cats = np.digitize(latent, bins)
    return np.clip(cats + 1, 1, 4).astype(np.int64)


def _inject_missing(
    df: pd.DataFrame,
    cols: list[str],
    missing_rate: float,
    rng: np.random.Generator,
) -> None:
    for c in cols:
        mask = rng.random(len(df)) < missing_rate
        df.loc[mask, c] = np.nan


def _inject_empty_strings(
    df: pd.DataFrame,
    cat_cols: list[str],
    rate: float,
    rng: np.random.Generator,
) -> None:
    for c in cat_cols:
        mask = rng.random(len(df)) < rate
        df.loc[mask, c] = ""


def _inject_mixed_numeric_noise(
    df: pd.DataFrame,
    trust_cols: list[str],
    rate: float,
    rng: np.random.Generator,
) -> None:
    for c in trust_cols:
        m = rng.random(len(df)) < rate
        idx = df.index[m]
        choice = rng.integers(0, 4, size=m.sum())
        junk = np.where(
            choice == 0,
            rng.choice(["99", "98", "отказ", "НЕТ ОТВЕТА"], size=m.sum()),
            np.where(choice == 1, rng.choice(["1", "2", "3", "4"], size=m.sum()), ""),
        )
        # перезаписать только часть замаскированных ячеек, чтобы сохранить некоторые числовые значения
        sub = rng.random(m.sum()) < 0.6
        bad_idx = idx[sub]
        bad_vals = junk[sub]
        df.loc[bad_idx, c] = bad_vals


def _core_column_count() -> int:
    return len(TRUST_COLUMNS) + 8


def _wide_survey_columns(n_rows: int, n_wide: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    числовые шкалы (1–5, 1–10), иногда float, и категории на кириллице;
    пропуски, пустые строки, редкий «мусор» в ячейках - как в реальных экспортах
    """
    if n_wide <= 0:
        return pd.DataFrame()

    out: dict[str, np.ndarray] = {}
    for j in range(n_wide):
        name = f"q_{j+1:03d}"
        kind = j % 7
        if kind == 0:
            x = rng.integers(1, 6, size=n_rows).astype(object)
        elif kind == 1:
            x = rng.integers(1, 11, size=n_rows).astype(object)
        elif kind == 2:
            x = np.round(rng.uniform(0, 10, size=n_rows), 1).astype(object)
        elif kind == 3:
            pool = CAT_POOLS_WIDE[j % len(CAT_POOLS_WIDE)]
            x = rng.choice(np.array(pool, dtype=object), size=n_rows)
        elif kind == 4:
            pool = CAT_POOLS_WIDE[(j + 1) % len(CAT_POOLS_WIDE)]
            x = rng.choice(np.array(pool, dtype=object), size=n_rows)
        else:
            pool = CAT_POOLS_WIDE[(j + 2) % len(CAT_POOLS_WIDE)]
            x = rng.choice(np.array(pool, dtype=object), size=n_rows)

        miss = rng.random(n_rows) < 0.06
        x = x.copy()
        x[miss] = np.nan

        if kind <= 2:
            stray = (~miss) & (rng.random(n_rows) < 0.012)
            x[stray] = rng.choice(["9", "не знаю", "", "98"], size=stray.sum())
        else:
            empty = (~miss) & (rng.random(n_rows) < 0.02)
            x[empty] = ""
            junk = (~miss) & (~empty) & (rng.random(n_rows) < 0.008)
            if junk.any():
                x[junk] = rng.choice(["???", "НЕТ ОТВЕТА"], size=junk.sum())

        out[name] = x

    return pd.DataFrame(out)


def generate_survey_dataframe(
    n_rows: int = DEFAULT_N_ROWS,
    seed: int | None = DEFAULT_SEED,
    *,
    n_cols: int = DEFAULT_N_COLS,
    missing_rate: float = 0.04,
    empty_cat_rate: float = 0.015,
    mixed_trust_rate: float = 0.02,
) -> pd.DataFrame:
    """

    Parameters
    ----------
    n_rows
        число респондентов
    n_cols
        итоговое число столбцов 
    seed
        генератор случайных чисел для воспроизводимости
    missing_rate
        доля NaN в целевых колонках
    empty_cat_rate
        доля пустых строк в категориальных столбцах (соцдем)
    mixed_trust_rate
        доля строк, в которых ячейки доверия могут стать ненужной строкой или цифрами строки
    """
    rng = _rng(seed)
    n = n_rows
    skepticism = _latent_profile(n, rng)
    trust_mat = _trust_values(n, skepticism, rng)

    trust_names = [t.name for t in TRUST_COLUMNS]
    data: dict[str, np.ndarray | list] = {
        **{name: trust_mat[:, j] for j, name in enumerate(trust_names)},
        "age": np.clip(rng.normal(42, 16, size=n), 18, 90).astype(int),
        "gender": rng.choice(GENDER_LABELS, size=n, p=[0.46, 0.52, 0.02]),
        "education": rng.choice(EDUCATION_LABELS, size=n),
        "income_subj": rng.choice(INCOME_SUBJ_LABELS, size=n),
        "settlement": rng.choice(SETTLEMENT_LABELS, size=n),
        "region_id": rng.choice(REGION_LABELS, size=n),
        "weight": rng.uniform(0.3, 1.8, size=n).round(4),
    }

    df = pd.DataFrame(data)

    # опционально
    t0 = pd.Timestamp("2022-01-01")
    span = (pd.Timestamp("2023-12-31") - t0).days + 1
    df["interview_date"] = t0 + pd.to_timedelta(rng.integers(0, span, size=n), unit="D")

    _inject_missing(df, trust_names, missing_rate, rng)
    for c in trust_names:
        df[c] = df[c].astype(object)
    socdem_cat = ["gender", "education", "income_subj", "settlement", "region_id"]
    _inject_empty_strings(df, socdem_cat, empty_cat_rate, rng)
    _inject_mixed_numeric_noise(df, trust_names, mixed_trust_rate, rng)

    n_core = _core_column_count()
    if n_cols < n_core:
        raise ValueError(f"n_cols must be >= {n_core} (core block), got {n_cols}")
    n_wide = n_cols - n_core
    if n_wide > 0:
        wide = _wide_survey_columns(n_rows, n_wide, rng)
        df = pd.concat([df, wide], axis=1)

    assert df.shape[1] == n_cols, (df.shape[1], n_cols)

    cols = df.columns.to_list()
    rng.shuffle(cols)
    return df[cols]
