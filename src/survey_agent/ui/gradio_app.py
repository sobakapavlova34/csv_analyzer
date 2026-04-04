"""
Чат в Gradio: загрузка CSV → полный пайплайн → гипотезы в диалоге.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr

from survey_agent.pipeline.full_pipeline import run_full_pipeline

logger = logging.getLogger(__name__)

WELCOME = (
    "привет! я сраный умный аналитик! я умею не только холдить таски в статусе \"в анализе\", "
    "но и делать что-то нормальное с данными. давай проверим, смогу ли я проанализировать "
    "твой фантастический датасет... "
)


def _hypotheses_to_markdown(data: dict[str, Any]) -> str:
    if data.get("source") == "skipped_no_api_key":
        return (
            "**Нет ключа API.** Добавь в `.env` переменную `BOTHUB_API_KEY` "
            "(или экспортируй в окружении), чтобы я смог вызвать модель и сгенерировать гипотезы."
        )
    if data.get("source") == "error":
        err = data.get("error", "неизвестная ошибка")
        hint = data.get("hint", "")
        return f"**Ошибка при генерации:** {err}\n\n{hint}".strip()

    hyps = data.get("hypotheses") or []
    if not hyps:
        return "**Пустой ответ:** гипотезы не распознаны. См. сырой JSON в артефактах."

    lines = ["### Гипотезы (как в `hypotheses.json`)", ""]
    for i, h in enumerate(hyps, 1):
        if not isinstance(h, dict):
            lines.append(f"{i}. {h}")
            continue
        title = h.get("title") or f"Гипотеза {i}"
        lines.append(f"**{i}. {title}**")
        if h.get("statement"):
            lines.append(h["statement"])
        if h.get("rationale"):
            lines.append(f"- *Обоснование:* {h['rationale']}")
        if h.get("suggested_test"):
            lines.append(f"- *Проверка:* {h['suggested_test']}")
        if h.get("variables_involved"):
            lines.append(f"- *Переменные:* {', '.join(map(str, h['variables_involved']))}")
        if h.get("grounding"):
            lines.append(f"- *Привязка к данным:* {h['grounding']}")
        lines.append("")
    lines.append("---")
    lines.append("```json")
    lines.append(json.dumps({"hypotheses": hyps}, ensure_ascii=False, indent=2))
    lines.append("```")
    return "\n".join(lines)


def run_analysis(csv_path: str | None) -> tuple[str, str]:
    """
    Запускает пайплайн; возвращает (markdown для чата, сырой JSON строкой для скачивания/отладки).
    """
    if not csv_path:
        return "Загрузи файл `.csv` выше.", ""

    path = Path(csv_path)
    if not path.is_file():
        return f"Файл не найден: {path}", ""

    tmp_root = Path(tempfile.mkdtemp(prefix="gradio_survey_"))
    try:
        manifest = run_full_pipeline(path, artifact_root=tmp_root)
        hp_path = Path(manifest["run_dir"]) / "hypotheses" / "hypotheses.json"
        raw = hp_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        md = _hypotheses_to_markdown(data)
        meta = f"\n\n_Артефакты прогона:_ `{manifest['run_dir']}`"
        return md + meta, raw
    except Exception as e:
        logger.exception("gradio pipeline")
        return f"**Сбой пайплайна:** `{type(e).__name__}: {e}`", ""
    # tmp_root оставляем для отладки; при желании можно shutil.rmtree


def _file_path(file: Any) -> str | None:
    if file is None:
        return None
    if isinstance(file, str):
        return file
    return getattr(file, "name", None)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Умный аналитик") as demo:
        gr.Markdown("# CSV → гипотезы\nЗагрузи таблицу и нажми кнопку.")

        # Gradio ≥4.44: формат сообщений — список dict с role / content
        chat = gr.Chatbot(
            label="Диалог",
            height=520,
            value=[{"role": "assistant", "content": WELCOME}],
        )
        file_csv = gr.File(label="Датасет (.csv)", file_types=[".csv"])
        raw_json = gr.State("")
        with gr.Row():
            btn = gr.Button("Сгенерировать гипотезы", variant="primary")

        def on_analyze(file, history: list | None, _raw: str):
            history = list(history) if history else [{"role": "assistant", "content": WELCOME}]
            path = _file_path(file)
            if not path:
                history.append({"role": "assistant", "content": "Сначала прикрепи CSV."})
                return history, ""

            fname = Path(path).name
            md, raw = run_analysis(path)
            history.append({"role": "user", "content": f"📎 Загружен файл: **{fname}**"})
            history.append({"role": "assistant", "content": md})
            return history, raw

        btn.click(
            on_analyze,
            inputs=[file_csv, chat, raw_json],
            outputs=[chat, raw_json],
        )

        gr.Markdown(
            "Нужны переменные `BOTHUB_API_KEY`, при сетевых ошибках — см. `BOTHUB_IGNORE_SYSTEM_PROXY` в `.env.example`."
        )

    return demo


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    app = build_app()
    app.launch(server_name=host, server_port=port, share=share)
