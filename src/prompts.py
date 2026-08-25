"""
Варианты промптов для сравнения в эксперименте.

Три техники, которые тестируем:
  1. base       — просто system prompt с требованием JSON
  2. few_shot   — + 2-3 примера "вопрос -> правильный ответ" перед основным вопросом
  3. cot        — Chain-of-Thought: просим модель сначала рассуждать, потом отвечать

Цель: понять, какая техника даёт больше точных ответов на squad.
"""

from data import load_few_shot_examples


BASE_SYSTEM_PROMPT = """Ты — ассистент для ответов на вопросы по тексту (question answering).

Тебе дают контекст (текст) и вопрос по этому тексту.
Твоя задача — найти в контексте точный ответ на вопрос.

Правила:
- Ответ должен быть коротким — обычно это слово, дата или короткая фраза из контекста
- Не придумывай ответ, если его нет в контексте
- Отвечай СТРОГО в формате JSON, без лишнего текста вокруг

Формат ответа:
{"answer": "<короткий ответ>", "confidence": "high" | "medium" | "low"}
"""


COT_SYSTEM_PROMPT = """Ты — ассистент для ответов на вопросы по тексту (question answering).

Тебе дают контекст (текст) и вопрос по этому тексту.

Прежде чем ответить, порассуждай пошагово:
- какая часть контекста относится к вопросу
- что именно спрашивается
- какой конкретно фрагмент текста является ответом

Затем дай короткий финальный ответ.

Отвечай СТРОГО в формате JSON, без лишнего текста вокруг:
{"reasoning": "<твоё рассуждение>", "answer": "<короткий финальный ответ>", "confidence": "high" | "medium" | "low"}
"""


def build_user_message(context: str, question: str) -> str:
    """Формирует user-сообщение с контекстом и вопросом."""
    return f"Контекст:\n{context}\n\nВопрос: {question}"


def build_few_shot_prefix(n_examples: int = 3) -> str:
    """
    Строит текстовый блок с примерами для few-shot промптинга.

    Вставляется в начало user-сообщения перед реальным вопросом.
    Формат примеров такой же, как ожидаемый JSON-ответ —
    модель видит паттерн "вопрос -> структурированный ответ".
    """
    examples = load_few_shot_examples(n=n_examples)

    blocks = []
    for ex in examples:
        blocks.append(
            f"Контекст:\n{ex['context']}\n\n"
            f"Вопрос: {ex['question']}\n"
            f'Ответ: {{"answer": "{ex["answer"]}", "confidence": "high"}}'
        )

    header = "Вот несколько примеров правильных ответов:\n\n"
    return header + "\n\n---\n\n".join(blocks) + "\n\n---\n\n"


def build_messages(
    context: str,
    question: str,
    use_few_shot: bool = False,
    use_cot: bool = False,
) -> list[dict]:
    """
    Собирает список messages для отправки в API.

    Комбинации, которые тестируем в run_experiment.py:
        - base                      (use_few_shot=False, use_cot=False)
        - base + few-shot           (use_few_shot=True,  use_cot=False)
        - base + CoT                (use_few_shot=False, use_cot=True)
    """
    system_prompt = COT_SYSTEM_PROMPT if use_cot else BASE_SYSTEM_PROMPT

    user_content = ""
    if use_few_shot:
        user_content += build_few_shot_prefix()

    user_content += build_user_message(context, question)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
