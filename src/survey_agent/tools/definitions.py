"""
Реестр инструментов для LLM (имя, описание, JSON Schema параметров).

Типичная таблица опроса: столбцы ``trust_*``, age, gender, education, …, q_* (имена не фиксированы в коде).
"""

from __future__ import annotations

from typing import Any

# Порядок — как в типичном цикле: профиль → снимок колонок → EDA → гипотезы/тесты.

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "dataset_profile",
        "description": (
            "Обзор датасета: число строк/столбцов, список имён колонок, доля пропусков, "
            "эвристика типа (numeric/categorical/datetime) по каждому столбцу. "
            "Вызывать в начале. Аргументы не нужны."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "column_snapshot",
        "description": (
            "Детали по одному столбцу: dtype, число пропусков, топ значений (для категорий), "
            "min/max/mean для числовых после coercing. Подходит для trust_*, gender, education, q_*."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "Имя столбца, например trust_media или gender",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Сколько уникальных значений показать для категорий",
                    "default": 8,
                },
            },
            "required": ["column"],
        },
    },
    {
        "name": "numeric_correlation",
        "description": (
            "Корреляция Пирсона между числовыми столбцами (после pd.to_numeric). "
            "Можно передать список имён или оставить пустым — возьмутся все coercible числовые."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Подмножество столбцов; пусто = авто",
                },
                "min_non_null_share": {
                    "type": "number",
                    "description": "Мин. доля непустых для включения столбца",
                    "default": 0.5,
                },
            },
            "required": [],
        },
    },
    {
        "name": "ttest_groups",
        "description": (
            "Независимый t-тест: сравнить числовую переменную (шкала доверия, age, q_*) "
            "между двумя группами категориального столбца (gender, education, …)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "numeric_column": {"type": "string"},
                "group_column": {"type": "string"},
                "group_a": {"type": "string", "description": "Подпись категории A (как в данных)"},
                "group_b": {"type": "string", "description": "Подпись категории B"},
            },
            "required": ["numeric_column", "group_column", "group_a", "group_b"],
        },
    },
    {
        "name": "mannwhitney_groups",
        "description": (
            "Mann–Whitney U: то же сравнение двух групп, если распределение не нормальное "
            "или мало наблюдений."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "numeric_column": {"type": "string"},
                "group_column": {"type": "string"},
                "group_a": {"type": "string"},
                "group_b": {"type": "string"},
            },
            "required": ["numeric_column", "group_column", "group_a", "group_b"],
        },
    },
    {
        "name": "chi_square_independence",
        "description": (
            "Хи-квадрат независимости для двух категориальных столбцов "
            "(например education × settlement или q_* × gender)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column_a": {"type": "string"},
                "column_b": {"type": "string"},
            },
            "required": ["column_a", "column_b"],
        },
    },
    {
        "name": "cramers_v",
        "description": (
            "Сила связи (Cramér's V) для той же пары категориальных столбцов, что и chi-square; "
            "удобно интерпретировать вместе с p-value."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column_a": {"type": "string"},
                "column_b": {"type": "string"},
            },
            "required": ["column_a", "column_b"],
        },
    },
    {
        "name": "kmeans_cluster_summary",
        "description": (
            "K-means по выбранным числовым столбцам: размеры кластеров, средние по фичам внутри кластера. "
            "Для поиска профилей «скептиков» и т.п."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "feature_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Например список trust_*",
                },
                "n_clusters": {"type": "integer", "default": 4},
                "standardize": {"type": "boolean", "default": True},
            },
            "required": ["feature_columns"],
        },
    },
    {
        "name": "pca_projection_2d",
        "description": (
            "PCA в 2 компоненты для числовых столбцов: доля объяснённой дисперсии и координаты точек "
            "(сэмпл для отчёта/графика)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "feature_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "sample_rows": {
                    "type": "integer",
                    "description": "Макс. число строк в ответе",
                    "default": 500,
                },
                "standardize": {"type": "boolean", "default": True},
            },
            "required": ["feature_columns"],
        },
    },
    {
        "name": "anova_numeric_by_category",
        "description": (
            "Однофакторный ANOVA F-test: числовая переменная по уровням одной категориальной "
            "(≥3 группы, например education)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "numeric_column": {"type": "string"},
                "category_column": {"type": "string"},
            },
            "required": ["numeric_column", "category_column"],
        },
    },
]


def tool_names() -> list[str]:
    return [s["name"] for s in TOOL_SPECS]
