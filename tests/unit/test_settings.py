from unittest.mock import patch

from app.utils import constants
from config import settings as settings_module


def _isolated_env(overrides: dict[str, str]):
    return patch.dict("os.environ", overrides, clear=True)


def test_load_settings_uses_llm_model_as_default_for_all_tasks():
    with _isolated_env(
        {
            "LLM_API_KEY": "test-api-key",
            "LLM_MODEL": "default-model",
            "LLM_BASE_URL": "https://example.test/v1",
            "LLM_TEMPERATURE": "0.25",
            "RAG_MAX_RETRIES": "4",
            "RAG_HALLUCINATION_THRESHOLD": "0.85",
            "RAG_TOP_K_DOCS": "8",
            "RAG_FAISS_INDEX_PATH": "./tmp/index",
        }
    ):
        loaded = settings_module.load_settings()

    assert loaded.llm.api_key == "test-api-key"
    assert loaded.llm.default_model_name == "default-model"
    assert loaded.llm.rewriting_model_name == loaded.llm.default_model_name
    assert loaded.llm.generation_model_name == loaded.llm.default_model_name
    assert loaded.llm.grading_model_name == loaded.llm.default_model_name
    assert (
        loaded.llm.grading_escalation_model_name == loaded.llm.default_model_name
    )
    assert loaded.llm.base_url == "https://example.test/v1"
    assert loaded.llm.temperature == 0.25
    assert loaded.rag.max_retries == 4
    assert loaded.rag.hallucination_threshold == 0.85
    assert loaded.rag.top_k_docs == 8
    assert loaded.rag.faiss_index_path == "./tmp/index"


def test_load_settings_prefers_task_specific_model_env_vars():
    with _isolated_env(
        {
            "LLM_MODEL": "default-model",
            "LLM_MODEL_REWRITING": "rewrite-model",
            "LLM_MODEL_GENERATION": "generation-model",
            "LLM_MODEL_GRADING": "grading-model",
            "LLM_MODEL_GRADING_ESCALATION": "grading-escalation-model",
        }
    ):
        loaded = settings_module.load_settings()

    assert loaded.llm.default_model_name == "default-model"
    assert loaded.llm.rewriting_model_name == "rewrite-model"
    assert loaded.llm.generation_model_name == "generation-model"
    assert loaded.llm.grading_model_name == "grading-model"
    assert loaded.llm.grading_escalation_model_name == "grading-escalation-model"


def test_load_settings_uses_openrouter_defaults_when_model_env_vars_are_missing():
    with _isolated_env(
        {
        }
    ):
        loaded = settings_module.load_settings()

    assert loaded.llm.default_model_name == "qwen/qwen-turbo"
    assert loaded.llm.rewriting_model_name == "qwen/qwen-turbo"
    assert loaded.llm.generation_model_name == "qwen/qwen-turbo"
    assert loaded.llm.grading_model_name == "qwen/qwen-turbo"
    assert loaded.llm.grading_escalation_model_name == "qwen/qwen-turbo"
    assert loaded.llm.base_url == "https://openrouter.ai/api/v1"


def test_load_settings_uses_task_specific_models_without_global_llm_model():
    with _isolated_env(
        {
            "LLM_MODEL_REWRITING": "rewrite-model",
            "LLM_MODEL_GENERATION": "generation-model",
            "LLM_MODEL_GRADING": "grading-model",
        }
    ):
        loaded = settings_module.load_settings()

    assert loaded.llm.default_model_name == "qwen/qwen-turbo"
    assert loaded.llm.rewriting_model_name == "rewrite-model"
    assert loaded.llm.generation_model_name == "generation-model"
    assert loaded.llm.grading_model_name == "grading-model"
    assert loaded.llm.grading_escalation_model_name == "grading-model"


def test_constants_are_sourced_from_shared_settings():
    assert constants.LLM_API_KEY == settings_module.settings.llm.api_key
    assert constants.LLM_MODEL == settings_module.settings.llm.default_model_name
    assert (
        constants.LLM_MODEL_REWRITING
        == settings_module.settings.llm.rewriting_model_name
    )
    assert (
        constants.LLM_MODEL_GENERATION
        == settings_module.settings.llm.generation_model_name
    )
    assert (
        constants.LLM_MODEL_GRADING
        == settings_module.settings.llm.grading_model_name
    )
    assert (
        constants.LLM_MODEL_GRADING_ESCALATION
        == settings_module.settings.llm.grading_escalation_model_name
    )
    assert constants.LLM_BASE_URL == settings_module.settings.llm.base_url
    assert constants.LLM_TEMPERATURE == settings_module.settings.llm.temperature
    assert constants.MAX_RETRIES == settings_module.settings.rag.max_retries
    assert (
        constants.HALLUCINATION_THRESHOLD
        == settings_module.settings.rag.hallucination_threshold
    )
    assert constants.TOP_K_DOCS == settings_module.settings.rag.top_k_docs
    assert constants.FAISS_INDEX_PATH == settings_module.settings.rag.faiss_index_path
