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
app/
├── graph/
│   ├── state.py              # GraphState schema
│   ├── builder.py            # LangGraph assembly
│   └── router.py             # Conditional Router — Node 5
├── nodes/
│   ├── rewriting.py          # Node 1 — Query Rewriting (Chen)
│   ├── retrieval.py          # Node 2 — FAISS Retrieval (Li)
│   ├── generation.py         # Node 3 — LLM Generation (Liu)
│   └── grading.py            # Node 4 — Hallucination Grading (Hu)
├── dataset_processing/       # Data pipeline (Li)
│   ├── dataset_loader.py
│   ├── embedder.py
│   └── vector_store.py
├── services/
│   ├── llm_service.py        # Qwen via DashScope (LangChain-compatible)
│   └── retriever.py          # RAGRetriever — FAISS index management
├── utils/
│   ├── tracer.py
│   └── constants.py
├── api/
└── main.py                   # CLI entry point

backend/                      # FastAPI + WebSocket server (Hu)
├── src/
│   ├── api/
│   ├── engines/              # core / fake engine modes
│   ├── schemas/
│   └── main.py
└── run.py                    # Backend entry point

web/                          # React frontend (Hu)
├── src/
└── package.json

config/                       # Centralized configuration
├── logging.py
└── settings.py

scripts/
└── build_index.py            # One-time FAISS index builder

data/
├── train-00000-of-00001.parquet   # AI/Math paper dataset (not in git)
└── index/                         # FAISS index files (not in git)

tests/
├── unit/
│   ├── test_rewriting.py
│   ├── test_retrieval.py
│   ├── test_grading.py
│   ├── test_workflow.py
│   ├── test_llm_service.py
│   ├── test_llm_service_live.py
│   └── test_backend_service.py
└── eval/
    ├── eval_behavior.py
    ├── eval_rewrite.py
    └── eval_rewriting_assert.py
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
| `LLM_API_KEY` | API key for the configured LLM provider |
| `RAG_ENGINE_MODE` | `fake` for frontend demo, `core` for full pipeline |
| `FAISS_INDEX_PATH` | Path to FAISS index directory |

---

## Running the System

**Backend:**

```bash
python -m backend.run
```

**Frontend** (separate terminal):

```bash
cd web
npm install   # first time only
npm run dev
```

Frontend available at `http://localhost:5173`.

---

## GraphState Contract

All nodes share a single `GraphState`. Do not add or rename fields without
discussing with the system architect (Liu) first.

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

**ai_math_paper_list** — 1220 AI and Math academic papers from HuggingFace.  
Source: <https://huggingface.co/datasets/fzyzcjy/ai_math_paper_list>  
Each paper's abstract is used as a retrieval unit. Title is stored as metadata.
