"""
Прогон экспериментов: одни и те же вопросы через API
с разными temperature, few-shot и CoT настройками.

Установка:
    pip install openai pydantic

Перед запуском:
    установи переменную окружения OPENAI_API_KEY
    (Windows PowerShell:  $env:OPENAI_API_KEY="твой_ключ")
"""

import json
import time
from openai import OpenAI

from schemas import LLMAnswer, LLMAnswerWithReasoning, ExperimentResult
from prompts import build_messages
from data import load_squad_sample

client = OpenAI()

MODEL = "gpt-4o-mini"  # дешёвая модель — для экспериментов с промптингом достаточно


def call_llm(messages: list[dict], temperature: float) -> dict:
    """
    Один вызов LLM API. Возвращает распарсенный JSON-ответ (dict).

    response_format={"type": "json_object"} — просим модель гарантированно
    вернуть валидный JSON, а не текст с JSON внутри.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    raw_content = response.choices[0].message.content
    return json.loads(raw_content)


def run_single_experiment(
    question_data: dict,
    temperature: float,
    use_few_shot: bool,
    use_cot: bool,
) -> ExperimentResult:
    """
    Прогоняет один вопрос через одну конкретную конфигурацию
    (temperature + few-shot + CoT) и возвращает структурированный результат.
    """
    messages = build_messages(
        context=question_data["context"],
        question=question_data["question"],
        use_few_shot=use_few_shot,
        use_cot=use_cot,
    )

    raw_answer = call_llm(messages, temperature=temperature)

    # валидируем ответ через Pydantic — если модель вернула кривой JSON,
    # здесь сразу увидим ошибку, а не будем разбираться на этапе evaluate
    if use_cot:
        validated = LLMAnswerWithReasoning(**raw_answer)
    else:
        validated = LLMAnswer(**raw_answer)

    return ExperimentResult(
        question=question_data["question"],
        ground_truth=question_data["answer"],
        model_answer=validated.answer,
        confidence=validated.confidence,
        temperature=temperature,
        used_few_shot=use_few_shot,
        used_cot=use_cot,
    )


def run_full_experiment(n_questions: int = 20) -> list[ExperimentResult]:
    """
    Основной прогон: для каждого вопроса тестируем 5 конфигураций:

        1. temperature=0.0, base
        2. temperature=0.7, base
        3. temperature=1.2, base
        4. temperature=0.0, few-shot
        5. temperature=0.0, CoT

    Так проще всего увидеть влияние каждого фактора отдельно,
    не смешивая temperature с промптингом в одном сравнении.
    """
    questions = load_squad_sample(n=n_questions)
    results: list[ExperimentResult] = []

    configs = [
        {"temperature": 0.0, "use_few_shot": False, "use_cot": False},
        {"temperature": 0.7, "use_few_shot": False, "use_cot": False},
        {"temperature": 1.2, "use_few_shot": False, "use_cot": False},
        {"temperature": 0.0, "use_few_shot": True, "use_cot": False},
        {"temperature": 0.0, "use_few_shot": False, "use_cot": True},
    ]

    total = len(questions) * len(configs)
    done = 0

    for q in questions:
        for cfg in configs:
            try:
                result = run_single_experiment(q, **cfg)
                results.append(result)
            except Exception as e:
                # если модель вернула невалидный JSON или API упал —
                # не роняем весь эксперимент, просто логируем и идём дальше
                print(f"[WARN] Ошибка на вопросе '{q['question'][:50]}...': {e}")

            done += 1
            print(f"[{done}/{total}] обработано")
            time.sleep(0.5)  # небольшая пауза, чтобы не упереться в rate limit

    return results


def save_results(results: list[ExperimentResult], path: str = "results.jsonl"):
    """Сохраняет результаты построчно в jsonl — удобно для evaluate.py."""
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")
    print(f"Сохранено {len(results)} результатов в {path}")


if __name__ == "__main__":
    results = run_full_experiment(n_questions=20)
    save_results(results)