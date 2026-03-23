# MathMind-RAG

A LangGraph-based Agentic RAG system for AI and Math academic paper question answering,
featuring a hallucination-detection feedback loop with automatic query retry.

```
User Query
    │
    ▼
[Node 1] Query Rewriting
    │
    ▼
[Node 2] Retrieval (FAISS)
    │
    ▼
[Node 3] Generation (LLM)
    │
    ▼
[Node 4] Hallucination Grading
    │
    ▼
[Node 5] Conditional Router ── score >= 0.7 ──► Output
    │
    └── score < 0.7 & retries < max ──► back to Node 1
```

---

## Project Structure

```
MathMind-RAG/
├── app/                          # Core RAG pipeline
│   ├── graph/
│   │   ├── state.py              # GraphState schema — shared contract
│   │   ├── builder.py            # LangGraph graph assembly
│   │   └── router.py             # Conditional Router — Node 5
│   ├── nodes/
│   │   ├── rewriting.py          # Node 1 — Query Rewriting
│   │   ├── retrieval.py          # Node 2 — FAISS Retrieval
│   │   ├── generation.py         # Node 3 — LLM Generation
│   │   └── grading.py            # Node 4 — Hallucination Grading
│   ├── dataset_processing/       # Data pipeline
│   │   ├── dataset_loader.py     # Load parquet → List[Document]
│   │   ├── embedder.py           # BAAI/bge-base-en-v1.5 embeddings
│   │   └── vector_store.py       # FAISS index build / search / save
│   ├── services/
│   │   ├── llm_service.py        # Qwen via DashScope (LangChain-compatible)
│   │   └── retriever.py          # RAGRetriever — end-to-end retrieval pipeline
│   └── utils/
│       ├── constants.py          # Environment variables and defaults
│       └── tracer.py             # build_trace_entry() helper
│
├── backend/                      # WebSocket + FastAPI server
│   ├── src/
│   │   ├── api/
│   │   │   ├── routes.py         # REST endpoints
│   │   │   └── websocket.py      # WebSocket handler
│   │   ├── engines/
│   │   │   ├── core_engine.py    # Full RAG pipeline engine
│   │   │   └── fake_engine.py    # Mock engine for frontend demo
│   │   ├── schemas/
│   │   │   ├── events.py         # WebSocket event types
│   │   │   └── messages.py       # Request / response models
│   │   ├── services/
│   │   │   └── session_service.py
│   │   ├── config.py             # Backend-specific config
│   │   ├── dependencies.py       # FastAPI dependency injection
│   │   └── main.py               # FastAPI app factory
│   └── run.py                    # Backend entry point
│
├── web/                          # React frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── components/       # UI components (answer-card, reasoning-panel, etc.)
│   │   │   ├── hooks/            # use-rag-session.ts
│   │   │   ├── lib/              # WebSocket client, transport, logger
│   │   │   └── types/            # TypeScript type definitions
│   │   └── styles/               # CSS and theme
│   ├── index.html
│   └── package.json
│
├── config/                       # Centralized configuration
│   ├── logging.py                # Logging setup
│   └── settings.py               # Settings model (reads from .env)
│
├── scripts/
│   ├── setup_data.py             # Auto-download dataset and build FAISS index
│   └── build_index.py            # Manual FAISS index builder
│
├── data/                         # Not in git
│   ├── train-00000-of-00001.parquet   # AI/Math paper dataset
│   └── index/                         # Generated FAISS index files
│
├── tests/
│   ├── unit/                     # pytest unit tests (45 tests, all passing)
│   │   ├── test_rewriting.py
│   │   ├── test_retrieval.py
│   │   ├── test_grading.py
│   │   ├── test_grading_execution.py
│   │   ├── test_grading_execution_live.py
│   │   ├── test_workflow.py
│   │   ├── test_llm_service.py
│   │   ├── test_llm_service_live.py
│   │   ├── test_backend_service.py
│   │   └── test_settings.py
│   └── eval/                     # Quantitative evaluation scripts
│       ├── eval_behavior.py
│       ├── eval_rewrite.py
│       └── eval_rewriting_assert.py
│
├── main.py                       # Pipeline entry point — exposes run_workflow()
├── conftest.py                   # pytest path configuration
├── requirements.txt
├── .env.example                  # Environment variable template — copy to .env
└── README.md
```

---

## Setup

```bash
git clone <repo-url>
cd MathMind-RAG

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Configure environment:**

```bash
cp .env.example .env
```

Then open `.env` and set your API key:

```dotenv
LLM_API_KEY=your_openrouter_api_key_here
```

This is the only field you must fill in. All other values have sensible defaults
and can be left as-is for local development.

Get your OpenRouter API key at: <https://openrouter.ai/>

---

## Data & Index Setup (First Time Only)

Run the setup script to automatically download the dataset and build the FAISS index:

```bash
python -m scripts.setup_data
```

This will:

1. Download `train-00000-of-00001.parquet` from HuggingFace if not present
2. Build `data/index/faiss_flat.index` and `data/index/documents.pkl` if not present

To rebuild the index manually:

```bash
python -m scripts.build_index
```

Dataset source: <https://huggingface.co/datasets/fzyzcjy/ai_math_paper_list>

---

## Environment Variables

Copy `.env.example` to `.env` — never commit `.env` to git.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `LLM_API_KEY` | **Yes** | OpenRouter API key | — |
| `RAG_ENGINE_MODE` | No | `core` for full pipeline, `fake` for UI testing only | `core` |
| `RAG_FAISS_INDEX_PATH` | No | Path to FAISS index directory | `./data/faiss_index` |
| `LLM_MODEL` | No | Default model for all LLM tasks | `qwen-turbo` |
| `LLM_MODEL_REWRITING` | No | Query rewriting model override | `LLM_MODEL` |
| `LLM_MODEL_GENERATION` | No | Answer generation model override | `LLM_MODEL` |
| `LLM_MODEL_GRADING` | No | Hallucination grading model override | `LLM_MODEL` |
| `LLM_MODEL_GRADING_ESCALATION` | No | Future stronger-model grading fallback | `LLM_MODEL_GRADING` |
| `LLM_BASE_URL` | No | OpenAI-compatible gateway endpoint | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `BACKEND_PORT` | No | Backend server port | `8000` |
| `FRONTEND_PORT` | No | Frontend dev server port | `5173` |
| `ALLOWED_ORIGINS` | No | CORS allowed origins | `http://127.0.0.1:5173,http://localhost:5173` |

If you use a shared gateway such as OpenRouter, keep `LLM_API_KEY` and `LLM_BASE_URL`
shared, and switch models per task with the task-specific `LLM_MODEL_*` variables.

All variables not listed above also have defaults defined in `config/settings.py`.

---

## Running the System

**Step 1 — Initialize data** (first time only):

```bash
python -m scripts.setup_data
```

**Step 2 — Start backend** (Terminal 1):

```bash
python -m backend.run
```

**Step 3 — Start frontend** (Terminal 2):

```bash
cd web
npm install   # first time only
npm run dev
```

Frontend available at `http://localhost:5173`.  
Backend running at `http://localhost:8000`.

---

## CLI Usage

Run the full pipeline from the command line without the frontend:

```bash
python main.py "What is chain of thought prompting?"
```

---

## Recommended Demo Queries

The dataset covers AI and Math research papers. The following queries work well:

- `What is chain of thought prompting?`
- `How does zero-shot reasoning work in LLMs?`
- `What is reinforcement learning from human feedback?`
- `How do large language models solve math problems?`

---

## GraphState Contract

All nodes share a single `GraphState`. Do not add or rename fields without
discussing with the system architect first.

| Field | Type | Description |
|-------|------|-------------|
| `query` | `str` | Original user question |
| `rewritten_query` | `str` | Rewritten query for retrieval |
| `failed_queries` | `Annotated[list, operator.add]` | All previously attempted queries |
| `retrieved_docs` | `List[Document]` | Retrieved LangChain Document objects |
| `answer` | `str` | Generated answer |
| `hallucination_score` | `float` | 0.0–1.0, higher = more grounded |
| `retry_count` | `int` | Current retry count |
| `max_retries` | `int` | Max retries allowed (default: 2) |
| `final_decision` | `str` | `"output"` / `"retry"` / `"stop"` |
| `error_message` | `Optional[str]` | Write here on error, do not raise |
| `execution_trace` | `Annotated[list, operator.add]` | Append-only execution log |

**Important:**

- `retrieved_docs` must be `List[Document]` from `langchain_core.documents`
- On error, write to `error_message` and return gracefully — do not raise exceptions
- Use `build_trace_entry()` from `app/utils/tracer.py` for trace entries
- Score threshold: `>= 0.7` acceptable, `< 0.7` triggers retry

---

## Unit Tests

```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run evaluation scripts (requires LLM API)
python -m tests.eval.eval_behavior
python -m tests.eval.eval_rewrite
```

---

## Dataset

**ai_math_paper_list** — 1220 AI and Math academic papers.  
Source: <https://huggingface.co/datasets/fzyzcjy/ai_math_paper_list>  
Each paper's abstract is used as a retrieval unit. Title is stored as metadata.
