from typing import Literal

from langchain_openai import ChatOpenAI

from config.settings import settings


LLMTask = Literal[
    "default",
    "rewriting",
    "generation",
    "grading",
    "grading_escalation",
]


def _normalize_openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized[: -len("/chat/completions")]
    return normalized


def _model_for_task(task: LLMTask = "default") -> str:
    llm_settings = settings.llm
    model_map: dict[LLMTask, str] = {
        "default": llm_settings.default_model_name,
        "rewriting": llm_settings.rewriting_model_name,
        "generation": llm_settings.generation_model_name,
        "grading": llm_settings.grading_model_name,
        "grading_escalation": llm_settings.grading_escalation_model_name,
    }
    return model_map[task]


def get_llm(task: LLMTask = "default"):
    return ChatOpenAI(
        model=_model_for_task(task),
        api_key=settings.llm.api_key,  # type: ignore[arg-type]
        base_url=_normalize_openai_base_url(settings.llm.base_url),
        temperature=settings.llm.temperature,
    )
