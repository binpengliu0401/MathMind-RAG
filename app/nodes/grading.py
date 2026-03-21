import time
from typing import Any
from typing import Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.graph.state import GraphState
from app.services.llm_service import get_llm
from app.utils.tracer import build_trace_entry
from config.logging import get_logger

logger = get_logger(__name__)


class GradingResult(BaseModel):
    score: float
    verdict: Literal["grounded", "partially_grounded", "unsupported"]
    explanation: str
    unsupported_claims: list[str] = Field(default_factory=list)


GRADING_SYSTEM_PROMPT = """You are a strict hallucination grader for a Retrieval-Augmented Generation system.
Your task is to judge grounding, not universal truth.

You must follow these rules:
1. Use ONLY the retrieved documents as evidence.
2. Do NOT use outside knowledge.
3. If a claim in the answer is not supported by the retrieved documents, treat it as unsupported.
4. Missing evidence means unsupported.
5. Semantic similarity alone is not enough; the support must be specific enough to justify the answer.
6. Give a lower score when unsupported claims appear.
7. Return structured output only.
8. The verdict field must be exactly one of: grounded, partially_grounded, unsupported.
9. unsupported_claims must list every distinct unsupported claim as a separate short sentence.
10. If a sentence contains multiple unsupported claims, split them into separate list items.
11. Do not write prose in the verdict field.

Scoring:
- 1.0 means the answer is fully grounded in the retrieved documents.
- 0.0 means the answer is completely unsupported by the retrieved documents.

Verdict rules:
- grounded: all material claims are supported by the retrieved documents.
- partially_grounded: some claims are supported, but one or more distinct claims are unsupported.
- unsupported: the answer is mostly or entirely unsupported by the retrieved documents.
"""


def _format_retrieved_docs(docs: list[Document]) -> str:
    formatted_docs = []
    for index, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source") or metadata.get("title") or f"Document {index}"
        content = str(getattr(doc, "page_content", "")).strip()
        formatted_docs.append(f"[DOC {index}] source={source}\n{content}")
    return "\n\n".join(formatted_docs)


def _build_grading_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", GRADING_SYSTEM_PROMPT),
            (
                "human",
                "Answer:\n{answer}\n\nRetrieved Documents:\n{context}",
            ),
        ]
    )


def _compute_score(score: Any) -> float:
    try:
        numeric_score = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError("Grading score is missing or non-numeric.") from exc
    return max(0.0, min(1.0, numeric_score))


def _normalize_verdict(verdict: Any, score: float, unsupported_claims: list[str]) -> str:
    if isinstance(verdict, str):
        normalized = verdict.strip().lower()
        if normalized in {"grounded", "partially_grounded", "unsupported"}:
            return normalized

        if "partially" in normalized:
            return "partially_grounded"
        if "not fully" in normalized or "contains an unsupported claim" in normalized:
            return "partially_grounded"
        if "not supported" in normalized or "unsupported" in normalized:
            return "unsupported"
        if "fully supported" in normalized or "fully grounded" in normalized:
            return "grounded"
        if "grounded" in normalized and unsupported_claims:
            return "partially_grounded"
        if "grounded" in normalized:
            return "grounded"

    if unsupported_claims:
        return "unsupported" if score <= 0.25 else "partially_grounded"
    if score >= 0.85:
        return "grounded"
    if score <= 0.25:
        return "unsupported"
    return "partially_grounded"


def _normalize_result(result: Any) -> GradingResult:
    if result is None:
        raise ValueError("Grading result is missing.")

    if isinstance(result, GradingResult):
        normalized = result
    elif hasattr(result, "model_dump"):
        normalized = GradingResult.model_validate(result.model_dump())
    else:
        normalized = GradingResult.model_validate(result)

    unsupported_claims = [
        claim.strip()
        for claim in normalized.unsupported_claims
        if isinstance(claim, str) and claim.strip()
    ]

    score = _compute_score(normalized.score)
    verdict = _normalize_verdict(normalized.verdict, score, unsupported_claims)

    return GradingResult(
        score=score,
        verdict=verdict,  # type: ignore[arg-type]
        explanation=(normalized.explanation or "").strip(),
        unsupported_claims=unsupported_claims,
    )


def _grade_answer(answer: str, retrieved_docs: list[Document]) -> GradingResult:
    logger.info(
        "Invoking grading LLM answer_chars=%d doc_count=%d",
        len(answer.strip()),
        len(retrieved_docs),
    )
    prompt = _build_grading_prompt()
    llm = get_llm()
    structured_llm = llm.with_structured_output(GradingResult)
    chain = prompt | structured_llm
    result = chain.invoke(
        {
            "answer": answer,
            "context": _format_retrieved_docs(retrieved_docs),
        }
    )
    return _normalize_result(result)


def _preview_text(text: str, limit: int = 120) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit] + "..."


def grade_hallucination(state: GraphState) -> dict:
    """
    Node 4: Hallucination Grading
    Evaluates whether the generated answer is supported by retrieved docs.
    """
    start = time.time()
    answer = str(state.get("answer", "") or "")
    retrieved_docs = state.get("retrieved_docs", []) or []

    try:
        logger.info(
            "Grading started answer_present=%s doc_count=%d",
            bool(answer.strip()),
            len(retrieved_docs),
        )

        if not answer.strip():
            logger.info("Skipping grading LLM because answer is empty")
            result = GradingResult(
                score=0.0,
                verdict="unsupported",
                explanation="Empty answer cannot be grounded in retrieved documents.",
                unsupported_claims=[],
            )
        elif not retrieved_docs:
            logger.info("Skipping grading LLM because no retrieved documents are available")
            result = GradingResult(
                score=0.0,
                verdict="unsupported",
                explanation="No retrieved documents were available for grounding.",
                unsupported_claims=[],
            )
        else:
            result = _grade_answer(answer, retrieved_docs)

        score = _compute_score(result.score)
        unsupported_claim_count = len(result.unsupported_claims)
        latency = round((time.time() - start) * 1000, 2)

        logger.info(
            "Grading completed score=%.2f verdict=%s unsupported_claims=%d doc_count=%d explanation=%s",
            score,
            result.verdict,
            unsupported_claim_count,
            len(retrieved_docs),
            _preview_text(result.explanation),
        )

        return {
            "hallucination_score": score,
            "execution_trace": [
                build_trace_entry(
                    node="grading",
                    status="success",
                    latency_ms=latency,
                    summary=(
                        f"score={score:.2f}, verdict={result.verdict}, "
                        f"unsupported_claims={unsupported_claim_count}"
                    ),
                    key_output={
                        "hallucination_score": score,
                        "verdict": result.verdict,
                        "unsupported_claim_count": unsupported_claim_count,
                        "doc_count": len(retrieved_docs),
                    },
                )
            ],
        }
    except Exception as exc:
        latency = round((time.time() - start) * 1000, 2)
        logger.exception("Hallucination grading failed")
        return {
            "hallucination_score": 0.0,
            "error_message": f"Hallucination grading failed: {str(exc)}",
            "execution_trace": [
                build_trace_entry(
                    node="grading",
                    status="error",
                    latency_ms=latency,
                    summary="grading failed; returned score=0.0",
                    key_output={
                        "hallucination_score": 0.0,
                        "doc_count": len(retrieved_docs),
                    },
                )
            ],
        }
