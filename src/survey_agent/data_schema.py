from __future__ import annotations

import pandas as pd

#  trust_media, trust_police, …
TRUST_COLUMN_PREFIX = "trust_"

COLUMNS_EDUCATION = "education"
COLUMNS_AGE = "age"
COLUMNS_GENDER = "gender"
COLUMNS_SETTLEMENT = "settlement"
COLUMNS_REGION = "region_id"
COLUMNS_INCOME_SUBJ = "income_subj"

OPTIONAL_DEMOGRAPHIC_FOR_TESTS: tuple[str, ...] = (
    COLUMNS_EDUCATION,
    COLUMNS_AGE,
    COLUMNS_GENDER,
    COLUMNS_SETTLEMENT,
    COLUMNS_REGION,
    COLUMNS_INCOME_SUBJ,
)

MIN_TRUST_COLUMNS_FOR_CORE_EDA = 2


def trust_column_names(df: pd.DataFrame) -> list[str]:
    p = TRUST_COLUMN_PREFIX
    return [c for c in df.columns if str(c).startswith(p)]


def structure_hint() -> str:
    return (
        f"Ожидается таблица с числовыми (или приводимыми к числу) столбцами "
        f"«{TRUST_COLUMN_PREFIX}*» — шкалы доверия (минимум {MIN_TRUST_COLUMNS_FOR_CORE_EDA} для кластеров и корреляций). "
        f"Для расширенных тестов полезны: {', '.join(OPTIONAL_DEMOGRAPHIC_FOR_TESTS)}."
    )
