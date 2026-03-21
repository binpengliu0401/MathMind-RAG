from app.utils.constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
from langchain_openai import ChatOpenAI


def _normalize_openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized[: -len("/chat/completions")]
    return normalized


def get_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,  # type: ignore
        base_url=_normalize_openai_base_url(LLM_BASE_URL),
        temperature=LLM_TEMPERATURE,
    )
