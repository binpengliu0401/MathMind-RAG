from types import SimpleNamespace
from unittest.mock import patch

from app.services import llm_service


def _make_settings(
    *,
    api_key="test-key",
    base_url="https://example.test/v1",
    temperature=0.35,
    default_model_name="default-model",
    rewriting_model_name="rewriting-model",
    generation_model_name="generation-model",
    grading_model_name="grading-model",
    grading_escalation_model_name="grading-escalation-model",
):
    return SimpleNamespace(
        llm=SimpleNamespace(
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            default_model_name=default_model_name,
            rewriting_model_name=rewriting_model_name,
            generation_model_name=generation_model_name,
            grading_model_name=grading_model_name,
            grading_escalation_model_name=grading_escalation_model_name,
        )
    )


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


def test_model_for_task_returns_expected_profile():
    fake_settings = _make_settings()
    with patch.object(llm_service, "settings", fake_settings):
        assert llm_service._model_for_task("default") == "default-model"
        assert llm_service._model_for_task("rewriting") == "rewriting-model"
        assert llm_service._model_for_task("generation") == "generation-model"
        assert llm_service._model_for_task("grading") == "grading-model"
        assert (
            llm_service._model_for_task("grading_escalation")
            == "grading-escalation-model"
        )


@patch("app.services.llm_service.ChatOpenAI")
def test_get_llm_constructs_chat_openai_with_project_settings(mock_chat_openai):
    fake_settings = _make_settings()

    with patch.object(llm_service, "settings", fake_settings):
        client = llm_service.get_llm("generation")

    mock_chat_openai.assert_called_once_with(
        model="generation-model",
        api_key="test-key",
        base_url="https://example.test/v1",
        temperature=0.35,
    )
    assert client is mock_chat_openai.return_value


@patch("app.services.llm_service.ChatOpenAI")
def test_get_llm_forwards_none_api_key_without_local_validation(mock_chat_openai):
    fake_settings = _make_settings(
        api_key=None,
        base_url="https://example.test/v1/chat/completions",
        temperature=0.0,
        default_model_name="test-model",
    )

    with patch.object(llm_service, "settings", fake_settings):
        llm_service.get_llm()

    mock_chat_openai.assert_called_once_with(
        model="test-model",
        api_key=None,
        base_url="https://example.test/v1",
        temperature=0.0,
    )
