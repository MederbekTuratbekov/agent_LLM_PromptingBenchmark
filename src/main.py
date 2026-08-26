"""
Точка входа проекта LLM Basics + Prompting.

Запуск:
    python main.py

Что произойдёт:
    1. Загрузятся вопросы из squad
    2. Каждый вопрос прогонится через 5 конфигураций
       (base x3 temperature, few-shot, chain-of-thought)
    3. Результаты сохранятся в results.jsonl
    4. Посчитается точность по каждой конфигурации
    5. Отчёт выведется в консоль и сохранится в report.json
"""

from run_experiment import run_full_experiment, save_results
from evaluate import load_results, evaluate, print_report


def main():
    print("=== Запуск эксперимента: LLM Basics + Prompting ===\n")

    print("Шаг 1/3: прогон вопросов через API...")
    results = run_full_experiment(n_questions=20)
    save_results(results)

    print("\nШаг 2/3: загрузка результатов для оценки...")
    loaded = load_results("results.jsonl")

    print("Шаг 3/3: подсчёт метрик...")
    report = evaluate(loaded)
    print_report(report)

    print("\nГотово. Результаты: results.jsonl, report.json")


if __name__ == "__main__":
    main()