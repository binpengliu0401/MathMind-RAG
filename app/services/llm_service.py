from langchain_openai import ChatOpenAI
from app.utils.constants import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL, LLM_TEMPERATURE


def get_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,  # type: ignore
        base_url=LLM_BASE_URL,
        temperature=LLM_TEMPERATURE,
    )
