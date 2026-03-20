from dotenv import load_dotenv
import os

load_dotenv()

# LLM
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = "qwen-plus"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_TEMPERATURE = 0.0

# Router
HALLUCINATION_THRESHOLD = 0.7
MAX_RETRIES = 2

# Retrieval
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
TOP_K_DOCS = 5
