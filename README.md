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
│   ├── utils/
│   │   ├── constants.py          # Environment variables and defaults
│   │   └── tracer.py             # build_trace_entry() helper
│   ├── api/                      # FastAPI routes
│   └── main.py                   # CLI entry point
│
├── backend/                      # WebSocket + FastAPI server
│   ├── src/
│   │   ├── api/
│   │   │   ├── routes.py         # REST endpoints
│   │   │   └── websocket.py      # WebSocket handler
│   │   ├── engines/
│   │   │   ├── core_engine.py    # Full RAG pipeline engine
│   │   │   └── fake_engine.py    # Mock engine for frontend demo
│   │   ├── schemas/              # Request / response models
│   │   └── main.py               # FastAPI app factory
│   └── run.py                    # Backend entry point
│
├── web/                          # React frontend
│   ├── src/
│   │   ├── app/                  # Components, hooks, types
│   │   └── styles/               # CSS and theme
│   ├── index.html
│   └── package.json
│
├── config/                       # Centralized configuration
│   ├── logging.py                # Logging setup
│   └── settings.py               # Pydantic settings model
│
├── scripts/
│   └── build_index.py            # One-time FAISS index builder
│
├── data/                         # Not in git
│   ├── train-00000-of-00001.parquet   # AI/Math paper dataset
│   └── index/                         # Generated FAISS index files
│
├── tests/
│   ├── unit/                     # pytest unit tests
│   │   ├── test_rewriting.py
│   │   ├── test_retrieval.py
│   │   ├── test_grading.py
│   │   ├── test_workflow.py
│   │   ├── test_llm_service.py
│   │   ├── test_llm_service_live.py
│   │   └── test_backend_service.py
│   └── eval/                     # Quantitative evaluation scripts
│       ├── eval_behavior.py
│       ├── eval_rewrite.py
│       └── eval_rewriting_assert.py
│
├── conftest.py                   # pytest path configuration
├── requirements.txt
├── .env.example
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

cp .env.example .env
# Fill in your own API keys in .env
```

---

## Data & Index Setup (First Time Only)

Download the dataset from HuggingFace and place it in `data/`:

```
data/train-00000-of-00001.parquet
```

Dataset source: <https://huggingface.co/datasets/fzyzcjy/ai_math_paper_list>

Then build the FAISS index:

```bash
python -m scripts.build_index
```

This generates `data/index/faiss_flat.index` and `data/index/documents.pkl`.

---

## Environment Variables

See `.env.example`. Each person fills in their own `.env` — never commit this file.

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | DashScope API key for Qwen |
| `RAG_ENGINE_MODE` | `fake` for frontend demo, `core` for full pipeline |
| `FAISS_INDEX_PATH` | Path to FAISS index directory |

---

## Running the System

**Backend** (Terminal 1):

```bash
python -m backend.run
```

**Frontend** (Terminal 2):

```bash
cd web
npm install   # first time only
npm run dev
```

Frontend available at `http://localhost:5173`.  
Backend running at `http://localhost:8000`.

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
- LLM: Qwen (`qwen-plus`) via DashScope, LangChain-compatible interface
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
