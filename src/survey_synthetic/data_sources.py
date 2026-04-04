from __future__ import annotations

from pathlib import Path

from survey_synthetic.generator import generate_survey_dataframe


def resolve_input_csv(
    user_path: Path | None,
    *,
    fallback_path: Path | None = None,
    n_rows: int = 5000,
    seed: int = 42,
) -> Path:
    if user_path is not None:
        p = user_path.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"CSV не найден: {p}")
        return p

    if fallback_path is None:
        raise ValueError("Должен быть указан либо путь пользователя, либо запасной путь")

    fallback_path = fallback_path.expanduser().resolve()
    if not fallback_path.is_file():
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_survey_dataframe(n_rows=n_rows, seed=seed)
        df.to_csv(fallback_path, index=False, encoding="utf-8")

    return fallback_path
