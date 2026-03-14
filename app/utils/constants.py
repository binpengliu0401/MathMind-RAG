from dotenv import load_dotenv
import os

load_dotenv()


# LLM
LLM_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
LLM_TEMPERATURE = 0.0

# Router
HALLUCINATION_THRESHOLD = 0.7
MAX_RETRIES = 2

# Retrieval
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
TOP_K_DOCS = 5
