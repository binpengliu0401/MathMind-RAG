from __future__ import annotations
import time
from typing import Any, AsyncIterator

from app.graph.builder import rag_graph
from app.utils.constants import MAX_RETRIES
from backend.src.engines.base import RAGEngine
from backend.src.schemas.events import (
    AnswerReplacedEvent,
    DocumentsRetrievedEvent,
    GradingCompletedEvent,
    HallucinationResult,
    QueryRewrittenEvent,
    RetrievedDoc,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    ServerEvent,
    StepChangedEvent,
)


def _doc_to_event(index: int, doc: Any) -> RetrievedDoc:
    metadata = getattr(doc, "metadata", {}) or {}
    source = str(metadata.get("source") or metadata.get("title") or f"Document {index}")
    snippet = str(getattr(doc, "page_content", "")).strip()
    return RetrievedDoc(
        id=str(metadata.get("id") or index),
        source=source,
        snippet=snippet,
        relevant=True,
    )


class CoreRAGEngine(RAGEngine):
    async def run(self, query: str) -> AsyncIterator[ServerEvent]:  # type: ignore
        run_id = f"run-{int(time.time() * 1000)}"
        yield RunStartedEvent(runId=run_id, query=query)

        initial_state = {
            "query": query,
            "rewritten_query": "",
            "failed_queries": [],
            "retrieved_docs": [],
            "answer": "",
            "hallucination_score": 0.0,
            "retry_count": 0,
            "max_retries": MAX_RETRIES,
            "final_decision": "",
            "error_message": None,
            "execution_trace": [],
        }

        final_state = {}

        try:
            async for chunk in rag_graph.astream(initial_state): # type: ignore
                node_name = list(chunk.keys())[0]
                node_output = chunk[node_name]
                final_state.update(node_output)

                if node_name == "rewriting":
                    yield StepChangedEvent(step="rewriting")
                    yield QueryRewrittenEvent(
                        rewrittenQuery=node_output.get("rewritten_query", "")
                    )

                elif node_name == "retrieval":
                    yield StepChangedEvent(step="retrieval")
                    docs = [
                        _doc_to_event(i + 1, doc)
                        for i, doc in enumerate(node_output.get("retrieved_docs", []))
                    ]
                    yield DocumentsRetrievedEvent(retrievedDocs=docs)

                elif node_name == "generation":
                    yield StepChangedEvent(step="generation")
                    yield AnswerReplacedEvent(answer=node_output.get("answer", ""))

                elif node_name == "grading":
                    yield StepChangedEvent(step="grading")
                    score = float(node_output.get("hallucination_score", 0.0))
                    yield GradingCompletedEvent(
                        hallucinationResult=HallucinationResult(
                            score=score,
                            explanation=(
                                "Evaluation derived from the current RAG workflow output. "
                                "Backend core integration does not yet expose unsupported claims."
                            ),
                        )
                    )

        except Exception as e:
            yield RunFailedEvent(error=str(e))
            return

        error_message = final_state.get("error_message")
        if error_message:
            yield RunFailedEvent(error=error_message)
            return

        yield RunCompletedEvent()
