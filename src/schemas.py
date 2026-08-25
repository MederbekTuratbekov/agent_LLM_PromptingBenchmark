"""
Pydantic-схемы для структурированных ответов LLM.

Идея: заставляем модель отвечать строго в JSON-формате,
а не свободным текстом. Это даёт:
  - предсказуемый парсинг ответа в коде
  - возможность автоматически проверять корректность (validation)
  - основу для сравнения точности при разных настройках сэмплинга
"""

from enum import Enum
from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Насколько модель уверена в своём ответе."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LLMAnswer(BaseModel):
    """
    Базовая структура ответа для задачи question-answering.

    Используется в экспериментах без Chain-of-Thought —
    просто вопрос -> прямой ответ.
    """
    answer: str = Field(
        description="Прямой ответ на вопрос, коротко, без лишних слов"
    )
    confidence: ConfidenceLevel = Field(
        description="Насколько модель уверена: high / medium / low"
    )


class LLMAnswerWithReasoning(BaseModel):
    """
    Структура ответа для экспериментов с Chain-of-Thought.

    Отличие от LLMAnswer: есть поле reasoning — модель сначала
    объясняет ход мысли, потом даёт финальный ответ.
    Используем, чтобы сравнить: помогает ли CoT на сложных вопросах.
    """
    reasoning: str = Field(
        description="Пошаговое рассуждение перед тем, как дать финальный ответ"
    )
    answer: str = Field(
        description="Финальный короткий ответ на вопрос"
    )
    confidence: ConfidenceLevel = Field(
        description="Насколько модель уверена: high / medium / low"
    )


class ExperimentResult(BaseModel):
    """
    Результат одного прогона: вопрос + ответ модели + метаданные эксперимента.

    Одна такая запись = одна строка в итоговой таблице метрик.
    """
    question: str
    ground_truth: str = Field(description="Правильный ответ из датасета squad")
    model_answer: str
    confidence: ConfidenceLevel
    temperature: float
    used_few_shot: bool
    used_cot: bool
    is_correct: bool = Field(
        default=False,
        description="Заполняется на этапе evaluate.py после сравнения с ground_truth"
    )