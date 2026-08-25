"""
Загрузка и подготовка данных из HuggingFace датасета squad.

squad = Stanford Question Answering Dataset:
каждый пример — это (контекст, вопрос, правильный ответ).
Удобен тем, что ответ короткий и однозначный —
легко автоматически проверить, угадала модель или нет.

Установка (если ещё не стоит):
    pip install datasets
"""

from datasets import load_dataset


def load_squad_sample(n: int = 25, seed: int = 42) -> list[dict]:
    """
    Загружает n случайных примеров из squad (validation split).

    Берём validation, а не train — там вопросы почище,
    и это ближе к тому, как модель встретит "новые" вопросы.

    Возвращает список словарей:
        {
            "context": str,   # текст, в котором есть ответ
            "question": str,  # сам вопрос
            "answer": str,    # правильный короткий ответ
        }
    """
    dataset = load_dataset("squad", split="validation")
    shuffled = dataset.shuffle(seed=seed)
    sample = shuffled.select(range(n))

    prepared = []
    for row in sample:
        # в squad answers — это список (может быть несколько вариантов
        # формулировки правильного ответа), берём первый
        answer_text = row["answers"]["text"][0] if row["answers"]["text"] else ""

        prepared.append({
            "context": row["context"],
            "question": row["question"],
            "answer": answer_text,
        })

    return prepared


def load_few_shot_examples(n: int = 3, seed: int = 123) -> list[dict]:
    """
    Отдельная небольшая выборка для few-shot промптинга.

    Важно: seed отличается от load_squad_sample, чтобы few-shot
    примеры НЕ пересекались с вопросами, на которых тестируем модель.
    Иначе эксперимент будет нечестным — модель просто "подсмотрит" ответ.
    """
    dataset = load_dataset("squad", split="train")
    shuffled = dataset.shuffle(seed=seed)
    sample = shuffled.select(range(n))

    examples = []
    for row in sample:
        answer_text = row["answers"]["text"][0] if row["answers"]["text"] else ""
        examples.append({
            "context": row["context"],
            "question": row["question"],
            "answer": answer_text,
        })

    return examples


if __name__ == "__main__":
    # быстрая проверка что всё грузится корректно
    samples = load_squad_sample(n=3)
    for s in samples:
        print(f"Q: {s['question']}")
        print(f"A: {s['answer']}")
        print(f"Context: {s['context'][:100]}...")
        print("---")