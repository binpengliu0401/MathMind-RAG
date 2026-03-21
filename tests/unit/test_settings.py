from unittest.mock import patch

from app.utils import constants
from config import settings as settings_module


def test_load_settings_accepts_repo_llm_env_names():
    with patch.dict(
        "os.environ",
        {
            "LLM_API_KEY": "repo-key",
            "LLM_MODEL": "repo-model",
            "LLM_BASE_URL": "https://example.test/v1",
            "LLM_TEMPERATURE": "0.25",
            "RAG_MAX_RETRIES": "4",
            "RAG_HALLUCINATION_THRESHOLD": "0.85",
            "RAG_TOP_K_DOCS": "8",
            "RAG_FAISS_INDEX_PATH": "./tmp/index",
        },
        clear=False,
    ):
        loaded = settings_module.load_settings()

    assert loaded.llm.api_key == "repo-key"
    assert loaded.llm.model_name == "repo-model"
    assert loaded.llm.base_url == "https://example.test/v1"
    assert loaded.llm.temperature == 0.25
    assert loaded.rag.max_retries == 4
    assert loaded.rag.hallucination_threshold == 0.85
    assert loaded.rag.top_k_docs == 8
    assert loaded.rag.faiss_index_path == "./tmp/index"


def test_constants_are_sourced_from_shared_settings():
    assert constants.LLM_API_KEY == settings_module.settings.llm.api_key
    assert constants.LLM_MODEL == settings_module.settings.llm.model_name
    assert constants.LLM_BASE_URL == settings_module.settings.llm.base_url
    assert constants.LLM_TEMPERATURE == settings_module.settings.llm.temperature
    assert constants.MAX_RETRIES == settings_module.settings.rag.max_retries
    assert constants.HALLUCINATION_THRESHOLD == settings_module.settings.rag.hallucination_threshold
    assert constants.TOP_K_DOCS == settings_module.settings.rag.top_k_docs
    assert constants.FAISS_INDEX_PATH == settings_module.settings.rag.faiss_index_path
