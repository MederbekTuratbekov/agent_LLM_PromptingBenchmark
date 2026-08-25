"""
Оценка результатов: считаем % правильных ответов
для каждой конфигурации (temperature / few-shot / CoT).

Это финальная метрика для резюме и понимания,
какая техника промптинга реально работает.
"""

import json
from collections import defaultdict

from schemas import ExperimentResult


def normalize(text: str) -> str:
    """
    Приводит текст к простому виду для сравнения:
    нижний регистр, без лишних пробелов и знаков препинания по краям.

    Нужно, потому что модель может ответить "Paris" вместо "paris",
    или "the river Nile" вместо "Nile" — это не должно считаться ошибкой
    при простом посимвольном сравнении.
    """
    return text.strip().lower().strip(".,!?\"'")


def is_answer_correct(model_answer: str, ground_truth: str) -> bool:
    """
    Простая проверка: считаем ответ верным, если строки совпадают
    после нормализации, ИЛИ правильный ответ содержится в ответе модели
    (модель могла ответить более развёрнуто).
    """
    norm_model = normalize(model_answer)
    norm_truth = normalize(ground_truth)

    if not norm_truth:
        return False

    return norm_truth in norm_model or norm_model == norm_truth


def load_results(path: str = "results.jsonl") -> list[ExperimentResult]:
    """Загружает результаты, сохранённые в run_experiment.py."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            results.append(ExperimentResult(**data))
    return results


def config_key(r: ExperimentResult) -> str:
    """Формирует читаемый ключ конфигурации для группировки."""
    if r.used_few_shot:
        return f"few_shot (temp={r.temperature})"
    if r.used_cot:
        return f"chain_of_thought (temp={r.temperature})"
    return f"base (temp={r.temperature})"


def evaluate(results: list[ExperimentResult]) -> dict:
    """
    Группирует результаты по конфигурации и считает точность (accuracy)
    для каждой группы.

    Возвращает словарь:
        {
            "base (temp=0.0)": {"correct": 15, "total": 20, "accuracy": 0.75},
            ...
        }
    """
    groups: dict[str, list[ExperimentResult]] = defaultdict(list)

    for r in results:
        r.is_correct = is_answer_correct(r.model_answer, r.ground_truth)
        groups[config_key(r)].append(r)

    report = {}
    for key, group_results in groups.items():
        correct = sum(1 for r in group_results if r.is_correct)
        total = len(group_results)
        report[key] = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 3) if total else 0.0,
        }

    return report


def print_report(report: dict):
    """Печатает отчёт в читаемом виде, отсортированный по точности."""
    print("\n=== РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ===\n")

    sorted_items = sorted(report.items(), key=lambda x: x[1]["accuracy"], reverse=True)

    for config, stats in sorted_items:
        bar = "█" * int(stats["accuracy"] * 20)
        print(f"{config:35s} {stats['accuracy']*100:5.1f}%  {bar}  ({stats['correct']}/{stats['total']})")

    print("\n--- Выводы ---")
    best = sorted_items[0]
    worst = sorted_items[-1]
    print(f"Лучшая конфигурация: {best[0]} — {best[1]['accuracy']*100:.1f}%")
    print(f"Худшая конфигурация: {worst[0]} — {worst[1]['accuracy']*100:.1f}%")


if __name__ == "__main__":
    results = load_results("results.jsonl")
    report = evaluate(results)
    print_report(report)

    # сохраняем отчёт отдельно — пригодится для README проекта
    with open("report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)