from config.settings import settings

# LLM
LLM_API_KEY = settings.llm.api_key
LLM_MODEL = settings.llm.model_name
LLM_BASE_URL = settings.llm.base_url
LLM_TEMPERATURE = settings.llm.temperature

# Router
HALLUCINATION_THRESHOLD = settings.rag.hallucination_threshold
MAX_RETRIES = settings.rag.max_retries

# Retrieval
FAISS_INDEX_PATH = settings.rag.faiss_index_path
TOP_K_DOCS = settings.rag.top_k_docs
