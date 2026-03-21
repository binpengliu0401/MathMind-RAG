from unittest.mock import patch

from app.services import llm_service


def test_normalize_openai_base_url_strips_chat_completions_suffix():
    assert (
        llm_service._normalize_openai_base_url(
            "https://api.siliconflow.cn/v1/chat/completions"
        )
        == "https://api.siliconflow.cn/v1"
    )


def test_normalize_openai_base_url_strips_trailing_slash_only():
    assert (
        llm_service._normalize_openai_base_url("https://example.test/v1/")
        == "https://example.test/v1"
    )


@patch("app.services.llm_service.ChatOpenAI")
@patch("app.services.llm_service.LLM_TEMPERATURE", 0.35)
@patch("app.services.llm_service.LLM_BASE_URL", "https://example.test/v1")
@patch("app.services.llm_service.LLM_MODEL", "test-model")
@patch("app.services.llm_service.LLM_API_KEY", "test-key")
def test_get_llm_constructs_chat_openai_with_project_settings(mock_chat_openai):
    client = llm_service.get_llm()

    mock_chat_openai.assert_called_once_with(
        model="test-model",
        api_key="test-key",
        base_url="https://example.test/v1",
        temperature=0.35,
    )
    assert client is mock_chat_openai.return_value


@patch("app.services.llm_service.ChatOpenAI")
@patch("app.services.llm_service.LLM_TEMPERATURE", 0.0)
@patch("app.services.llm_service.LLM_BASE_URL", "https://example.test/v1/chat/completions")
@patch("app.services.llm_service.LLM_MODEL", "test-model")
@patch("app.services.llm_service.LLM_API_KEY", None)
def test_get_llm_forwards_none_api_key_without_local_validation(mock_chat_openai):
    llm_service.get_llm()

    mock_chat_openai.assert_called_once_with(
        model="test-model",
        api_key=None,
        base_url="https://example.test/v1",
        temperature=0.0,
    )
