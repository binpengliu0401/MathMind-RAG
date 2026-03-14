from langchain_openai import ChatOpenAI
from app.utils.constants import LLM_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE


def get_llm():
    return ChatOpenAI(
        model=LLM_MODEL_NAME,
        api_key=LLM_API_KEY,  # type: ignore
        temperature=LLM_TEMPERATURE,
    )
