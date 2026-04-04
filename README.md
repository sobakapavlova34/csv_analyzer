###__ЗАПУСК ПАЙПЛАЙНА__###

```pip install -r requirements.txt```

```PYTHONPATH=src python scripts/run_pipeline.py --csv /любой/путь/к/файлу.csv```


###__ПОЛУЧЕНИЕ ОТЧЕТА__###

```PYTHONPATH=src python scripts/build_report.py --latest```


###__ЗАПУСК DUMMY-БОТА В ТЕСТОВОМ КОНТУРЕ__###

```PYTHONPATH=src python scripts/gradio_chat.py --host 0.0.0.0 --port 7860```


###__ГЕНЕРАЦИЯ ДАТАСЕТА__###

```PYTHONPATH=src python scripts/generate_dataset.py --out ./survey_synthetic.csv```


###__РЕЖИМ АГЕНТА С TOOL-CALLING ИЗ КОМАНДНОЙ СТРОКИ__###
```PYTHONPATH=src python scripts/run_agent.py --csv /путь/к/данным.csv```
