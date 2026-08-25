from enum import Enum
from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LLMAnswer(BaseModel):
    answer: str = Field(
        description="Прямой ответ на вопрос, коротко, без лишних слов"
    )
    confidence: ConfidenceLevel = Field(
        description="Насколько модель уверена: high / medium / low"
    )


class LLMAnswerWithReasoning(BaseModel):
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
