"""
Генерация содержательных гипотез на основе артефактов EDA (через LLM / BotHub).
Пишет JSON и Markdown; без ключа — заглушка с инструкцией.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from openai import APIConnectionError

from survey_agent.agent.llm_client import make_openai_client
from survey_agent.config import AgentConfig

logger = logging.getLogger(__name__)

HYPOTHESIS_SYSTEM = """Ты социолог-исследователь с сильным воображением и привычкой задавать неудобные вопросы данным. Ниже — предобработка и разведочный анализ: кластеры, корреляции институтов доверия, соцдем, предикторы, разрывы по образованию и т.д.

Сформулируй **от 7 до 10** проверяемых гипотез на русском.

### Главный приоритет: новизна и интерес
- Не ограничивайся «учебниковым» набором (СМИ×образование, полиция×суды, один скептический кластер). **Минимум половина** гипотез должны звучать **неожиданно**: контраст между институтами, «парадокс доверия», напряжение (высокое к одному — низкое к другому), гетерогенность по подгруппе или региону/типу поселения — **если** это опирается на числа или явные паттерны из JSON.
- Ищи **второй слой**: не только топ-корреляции, но и **слабые или отрицательные** связи, нетипичные пары из focus_pairs / correlation_tool, несовпадение профилей кластеров по разным осям доверия.
- Формулировки должны быть **содержательными**: намёк на механизм, иронию политической психологии или социального раскола — без воды и без общих фраз.

### Жёсткие правила качества
- Числа, r, доли, метки кластеров — **только из входных данных**; не придумывай значений.
- Запрещены пустые обобщения («кластеры различаются») без **конкретных институтов trust_* и/или соцдема**.
- Запрещены банальности вроде «мужчины и женщины различаются» без связки с **конкретными** шкалами доверия из данных.

### Формат ответа
Строго JSON-объект с ключом "hypotheses" — массив объектов:
{"hypotheses": [
  {"title": "...", "statement": "...", "rationale": "...", "suggested_test": "...", "variables_involved": ["..."], "grounding": "какой фрагмент анализа поддерживает идею (1 предложение)"}
]}
Поле "grounding" обязательно: привязка к данным (кластер N, корреляция X–Y, группа образования и т.д.).
"""


def _parse_hypotheses_response(text: str) -> list[Any]:
    """Достаёт список гипотез из JSON или из markdown-блока."""
    text = (text or "").strip()
    try:
        parsed = json.loads(text)
        hyps = parsed.get("hypotheses", parsed) if isinstance(parsed, dict) else []
        if isinstance(hyps, dict):
            hyps = list(hyps.values())
        return hyps if isinstance(hyps, list) else []
    except json.JSONDecodeError:
        pass
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.lower().startswith("json"):
                part = part[4:].strip()
            try:
                parsed = json.loads(part)
                if isinstance(parsed, dict) and "hypotheses" in parsed:
                    return parsed["hypotheses"] if isinstance(parsed["hypotheses"], list) else []
            except json.JSONDecodeError:
                continue
    return []


def _read_json(path: Path, max_chars: int = 25_000) -> str:
    if not path.is_file():
        return ""
    t = path.read_text(encoding="utf-8")
    return t if len(t) <= max_chars else t[:max_chars] + "\n…[truncated]"


def run_hypotheses_stage(
    run_dir: Path,
    cfg: AgentConfig | None,
    *,
    preprocess_subdir: str = "preprocess",
    eda_subdir: str = "eda",
) -> dict[str, Any]:
    """
    Читает ``preprocess/report.json``, ``eda/*.json`` и формирует гипотезы.

    Пишет в ``run_dir/hypotheses/``: ``hypotheses.json``, ``hypotheses.md``, ``hypothesis_prompt_context.txt``.
    """
    out = run_dir / "hypotheses"
    out.mkdir(parents=True, exist_ok=True)
    pre = run_dir / preprocess_subdir
    eda = run_dir / eda_subdir

    context_parts = [
        "=== preprocess/report.json ===",
        _read_json(pre / "report.json", 20_000),
        "\n=== eda/cluster_story.json (КЛЮЧЕВОЕ: кластеры, соцдем, пары институтов, предикторы) ===",
        _read_json(eda / "cluster_story.json", 35_000),
        "\n=== eda/cluster_profiles.json ===",
        _read_json(eda / "cluster_profiles.json", 18_000),
        "\n=== eda/eda_report.json ===",
        _read_json(eda / "eda_report.json", 8000),
        "\n=== eda/correlation_tool.json (фрагмент) ===",
        _read_json(eda / "correlation_tool.json", 10_000),
        "\n=== eda/clusters_kmeans.json (техн. сводка k-means) ===",
        _read_json(eda / "clusters_kmeans.json", 8000),
        "\n=== eda/distributions.json (фрагмент) ===",
        _read_json(eda / "distributions.json", 10_000),
    ]
    context = "\n".join(context_parts)
    (out / "hypothesis_prompt_context.txt").write_text(context, encoding="utf-8")

    result: dict[str, Any] = {"source": "llm", "hypotheses": []}

    if cfg is None:
        result["source"] = "skipped_no_api_key"
        result["message"] = "Задай BOTHUB_API_KEY для генерации гипотез через LLM."
        (out / "hypotheses.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "hypotheses.md").write_text(
            "# Гипотезы\n\nКлюч API не задан. Скопируй `.env.example` в `.env` и укажи `BOTHUB_API_KEY`.\n",
            encoding="utf-8",
        )
        (run_dir / "logs" / "hypotheses.log").write_text("skipped: no BOTHUB_API_KEY\n", encoding="utf-8")
        return result

    user_msg = (
        "Контекст исследования (опрос, доверие к институтам). "
        "Сформулируй смелые, разнообразные гипотезы по данным ниже — избегай шаблонного набора из очевидных тем\n\n"
        + context[:100_000]
    )

    messages = [
        {"role": "system", "content": HYPOTHESIS_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    # connection error часто из‑за HTTP_PROXY/ALL_PROXY; делаем второй проход с прямым httpx
    attempts: list[tuple[bool, bool]] = [
        (False, True),
        (True, True),
        (True, False),
    ]
    last_err: Exception | None = None
    for ignore_proxy, use_json_object in attempts:
        try:
            client = make_openai_client(cfg, ignore_system_proxy=ignore_proxy)
            hyp_temp = float(os.environ.get("BOTHUB_HYPOTHESIS_TEMPERATURE", "0.58"))
            kwargs: dict[str, Any] = {
                "model": cfg.model,
                "messages": messages,
                "temperature": max(0.0, min(1.5, hyp_temp)),
            }
            if use_json_object:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or "{}"
            hyps = _parse_hypotheses_response(text)
            if hyps:
                result["hypotheses"] = hyps
                result["llm_meta"] = {
                    "ignore_system_proxy": ignore_proxy,
                    "response_format_json_object": use_json_object,
                }
                break
            last_err = RuntimeError("Пустой или неразобранный ответ модели")
        except APIConnectionError as e:
            last_err = e
            logger.warning(
                "hypothesis LLM attempt failed (proxy=%s json=%s): %s",
                ignore_proxy,
                use_json_object,
                e,
            )
        except Exception as e:
            last_err = e
            logger.warning("hypothesis LLM attempt failed: %s", e)

    if not result["hypotheses"] and last_err is not None:
        result["source"] = "error"
        err_msg = str(last_err)
        if last_err.__cause__ is not None:
            err_msg = f"{err_msg} | cause: {last_err.__cause__!r}"
        result["error"] = err_msg
        result["hint"] = (
            "Если видишь Proxy/403/Connection error: задай BOTHUB_IGNORE_SYSTEM_PROXY=1 "
            "или убери HTTP_PROXY/HTTPS_PROXY из окружения. Проверь BOTHUB_BASE_URL и BOTHUB_MODEL."
        )
    elif not result["hypotheses"]:
        result["source"] = "error"
        result["error"] = "Не удалось разобрать ответ модели как JSON с ключом hypotheses."

    (out / "hypotheses.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = ["# Содержательные гипотезы", ""]
    if result.get("source") == "error":
        lines.append("**Ошибка:** см. `hypotheses.json`.")
        if result.get("hint"):
            lines.append("")
            lines.append(result["hint"])
        lines.append("")
    for i, h in enumerate(result.get("hypotheses") or [], 1):
        if isinstance(h, dict):
            lines.append(f"## {i}. {h.get('title', 'Гипотеза')}")
            lines.append("")
            lines.append(h.get("statement", h.get("text", "")))
            lines.append("")
            if h.get("rationale"):
                lines.append(f"**Обоснование:** {h['rationale']}")
            if h.get("suggested_test"):
                lines.append(f"**Проверка:** {h['suggested_test']}")
            if h.get("variables_involved"):
                lines.append(f"**Переменные:** {', '.join(map(str, h['variables_involved']))}")
            if h.get("grounding"):
                lines.append(f"**Привязка к данным:** {h['grounding']}")
            lines.append("")
    (out / "hypotheses.md").write_text("\n".join(lines), encoding="utf-8")

    (run_dir / "logs" / "hypotheses.log").write_text(
        json.dumps({"ok": result.get("source") != "error", "n": len(result.get("hypotheses", []))}, ensure_ascii=False),
        encoding="utf-8",
    )

    return result


def load_config_optional() -> AgentConfig | None:
    import os

    key = os.environ.get("BOTHUB_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    return AgentConfig.from_env()
