# LLM Basics + Prompting

Часть 1 из 5 в серии проектов по блоку `agent_junior`.

## Problem

Нужно понять, как настройки sampling (temperature) и техники промптинга
(few-shot, Chain-of-Thought) влияют на точность ответов LLM в задаче
question answering.

## Approach

- Датасет: HuggingFace `squad` (20 вопросов из validation split)
- Модель: `gpt-4o-mini` через OpenAI API
- Структурированный вывод: JSON через Pydantic-схемы (`schemas.py`)
- Протестированы 5 конфигураций:
  - base, temperature=0.0
  - base, temperature=0.7
  - base, temperature=1.2
  - few-shot (3 примера), temperature=0.0
  - Chain-of-Thought, temperature=0.0
- Метрика: accuracy — совпадение ответа модели с ground truth
  (с нормализацией регистра и пунктуации)

## Results

Точность по каждой конфигурации — в `report.json` после запуска.
Формат:
```
base (temp=0.0)                 XX.X%
few_shot (temp=0.0)             XX.X%
chain_of_thought (temp=0.0)     XX.X%
```

## How to run

```bash
pip install -r requirements.txt

# Windows PowerShell:
$env:OPENAI_API_KEY="твой_ключ"

python main.py
```

Результаты появятся в `results.jsonl` (сырые данные) и `report.json` (метрики).

## Структура проекта

```
llm-prompting-benchmark/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── main.py
    ├── schemas.py         # Pydantic-модели для JSON-ответов
    ├── data.py             # загрузка squad
    ├── prompts.py           # base / few-shot / CoT промпты
    ├── run_experiment.py      # прогон через API
    ├── evaluate.py              # подсчёт accuracy
    └── main.py                    # запуск всего пайплайна
```