import re
import time
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.graph.state import GraphState
from app.services.llm_service import get_llm
from app.utils.tracer import build_trace_entry
from config.logging import get_logger


logger = get_logger(__name__)

ClaimLabel = Literal["supported", "partial", "unsupported"]
VerdictLabel = Literal["grounded", "partially_grounded", "unsupported"]
CLAIM_SCORE_MAP: dict[ClaimLabel, float] = {
    "supported": 1.0,
    "partial": 0.5,
    "unsupported": 0.0,
}


class ClaimAssessment(BaseModel):
    claim: str
    label: ClaimLabel
    explanation: str = ""


class GradingResult(BaseModel):
    score: float | None = None
    verdict: VerdictLabel
    explanation: str
    claims: list[ClaimAssessment] = Field(default_factory=list)
    supported_claims: list[str] = Field(default_factory=list)
    partial_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    claim_count: int = 0


GRADING_SYSTEM_PROMPT = """You are a strict hallucination grader for a Retrieval-Augmented Generation system.
Your task is to judge grounding, not universal truth.

You must follow these rules:
1. Use ONLY the retrieved documents as evidence.
2. Do NOT use outside knowledge.
3. Missing evidence means unsupported.
4. Semantic similarity alone is not enough. A claim must have specific support in the documents.
5. Split the answer into material claims and evaluate each claim independently.
6. Label each claim as exactly one of: supported, partial, unsupported.
7. supported means the claim is clearly backed by the documents.
8. partial means the claim is only partly supported, overly broad, or missing important support details.
9. unsupported means the claim is not supported by the documents.
10. Return one claim assessment per claim.
11. unsupported_claims must include every distinct unsupported claim separately.
12. partial_claims must include every distinct partially supported claim separately.
13. supported_claims must include every distinct supported claim separately.
14. The verdict field must be exactly one of: grounded, partially_grounded, unsupported.
15. grounded requires all material claims to be supported.
16. partially_grounded means there is at least one partial or unsupported claim, but not most claims.
17. unsupported means most or all claims are unsupported.
18. Do not use prose in the verdict field. Return structured output only.
"""


def _format_retrieved_docs(docs: list[Document]) -> str:
    formatted_docs = []
    for index, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source") or metadata.get("title") or f"Document {index}"
        content = str(getattr(doc, "page_content", "")).strip()
        formatted_docs.append(f"[DOC {index}] source={source}\n{content}")
    return "\n\n".join(formatted_docs)


def _is_meaningful_claim(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text).strip(" ,;:.!?-")
    if len(cleaned) < 12:
        return False
    if not re.search(r"[A-Za-z]", cleaned):
        return False

    lowered = cleaned.lower()
    meaningless_fragments = {
        "and",
        "but",
        "while",
        "whereas",
        "for example",
        "such as",
        "which is",
        "that is",
        "in addition",
        "however",
    }
    if lowered in meaningless_fragments:
        return False

    if re.fullmatch(r"(and|but|while|whereas)\b", lowered):
        return False

    return True


def _split_into_claims(answer: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", answer).strip()
    if not normalized:
        return []

    # Stage 1: coarse sentence splitting on terminal punctuation.
    sentences = [
        part.strip(" ,;")
        for part in re.split(r"[.!?]+(?:\s+|$)", normalized)
        if part.strip(" ,;")
    ]

    claims: list[str] = []
    clause_pattern = re.compile(
        r";+|,\s+(?:and|but|while|whereas)\s+", flags=re.IGNORECASE
    )

    for sentence in sentences:
        # Stage 2: split only on semicolons and selected conjunction-led comma clauses.
        parts = [
            part.strip(" ,;")
            for part in clause_pattern.split(sentence)
            if part.strip(" ,;")
        ]
        for part in parts:
            if _is_meaningful_claim(part):
                claims.append(part)

    cleaned_claims: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        cleaned = re.sub(r"\s+", " ", claim).strip(" ,;:.!?")
        key = cleaned.lower()
        if key and key not in seen:
            seen.add(key)
            cleaned_claims.append(cleaned)

    return cleaned_claims or [normalized]


def _format_claims(claims: list[str]) -> str:
    return "\n".join(
        f"[CLAIM {index}] {claim}" for index, claim in enumerate(claims, start=1)
    )


def _build_grading_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", GRADING_SYSTEM_PROMPT),
            (
                "human",
                "Answer:\n{answer}\n\nClaims:\n{claims}\n\nRetrieved Documents:\n{context}",
            ),
        ]
    )


def _compute_score(score: Any) -> float:
    try:
        numeric_score = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError("Grading score is missing or non-numeric.") from exc
    return max(0.0, min(1.0, numeric_score))


def _aggregate_claim_score(claims: list[ClaimAssessment]) -> float:
    if not claims:
        return 0.0
    total = sum(CLAIM_SCORE_MAP[claim.label] for claim in claims)
    return total / len(claims)


def _normalize_claims(
    raw_claims: list[ClaimAssessment | dict[str, Any]],
) -> list[ClaimAssessment]:
    normalized_claims: list[ClaimAssessment] = []
    for raw_claim in raw_claims:
        if hasattr(raw_claim, "model_dump"):
            claim = ClaimAssessment.model_validate(raw_claim.model_dump())  # type: ignore
        else:
            claim = ClaimAssessment.model_validate(raw_claim)
        cleaned_text = claim.claim.strip()
        if not cleaned_text:
            continue
        normalized_claims.append(
            ClaimAssessment(
                claim=cleaned_text,
                label=claim.label,
                explanation=claim.explanation.strip(),
            )
        )
    return normalized_claims


def _derive_claim_lists(
    claims: list[ClaimAssessment],
) -> tuple[list[str], list[str], list[str]]:
    supported = [claim.claim for claim in claims if claim.label == "supported"]
    partial = [claim.claim for claim in claims if claim.label == "partial"]
    unsupported = [claim.claim for claim in claims if claim.label == "unsupported"]
    return supported, partial, unsupported


def _normalize_verdict(
    verdict: Any,
    score: float,
    partial_claims: list[str],
    unsupported_claims: list[str],
    claim_count: int,
) -> VerdictLabel:
    has_non_grounded_claims = bool(partial_claims or unsupported_claims)

    if isinstance(verdict, str):
        normalized = verdict.strip().lower()
        if normalized == "grounded" and has_non_grounded_claims:
            return "partially_grounded"
        if (
            normalized == "unsupported"
            and claim_count > 0
            and not unsupported_claims
            and score > 0.25
        ):
            return "partially_grounded"
        if normalized in {"grounded", "partially_grounded", "unsupported"}:
            return normalized  # type: ignore[return-value]
        if "partially" in normalized:
            return "partially_grounded"
        if "unsupported" in normalized or "not supported" in normalized:
            return "unsupported"
        if "grounded" in normalized:
            return "grounded"

    if claim_count == 0:
        return "unsupported"
    if unsupported_claims and len(unsupported_claims) >= max(1, (claim_count + 1) // 2):
        return "unsupported"
    if unsupported_claims or partial_claims or score < 0.999:
        return "partially_grounded"
    return "grounded"


def _normalize_result(result: Any) -> GradingResult:
    if result is None:
        raise ValueError("Grading result is missing.")

    if isinstance(result, GradingResult):
        normalized = result
    elif hasattr(result, "model_dump"):
        normalized = GradingResult.model_validate(result.model_dump())
    else:
        normalized = GradingResult.model_validate(result)

    normalized_claims = _normalize_claims(normalized.claims)  # type: ignore
    if normalized_claims:
        supported_claims, partial_claims, unsupported_claims = _derive_claim_lists(
            normalized_claims
        )
        score = _aggregate_claim_score(normalized_claims)
        claim_count = len(normalized_claims)
    else:
        supported_claims = [
            claim.strip() for claim in normalized.supported_claims if claim.strip()
        ]
        partial_claims = [
            claim.strip() for claim in normalized.partial_claims if claim.strip()
        ]
        unsupported_claims = [
            claim.strip() for claim in normalized.unsupported_claims if claim.strip()
        ]
        claim_count = normalized.claim_count or (
            len(supported_claims) + len(partial_claims) + len(unsupported_claims)
        )
        score = _compute_score(normalized.score)

    verdict = _normalize_verdict(
        normalized.verdict,
        score,
        partial_claims,
        unsupported_claims,
        claim_count,
    )

    return GradingResult(
        score=score,
        verdict=verdict,
        explanation=(normalized.explanation or "").strip(),
        claims=normalized_claims,
        supported_claims=supported_claims,
        partial_claims=partial_claims,
        unsupported_claims=unsupported_claims,
        claim_count=claim_count,
    )


def _should_escalate(result: GradingResult) -> bool:
    # Reserved for a future stronger-model fallback without changing the node contract.
    return False


def _grade_answer(answer: str, retrieved_docs: list[Document]) -> GradingResult:
    claims = _split_into_claims(answer)
    claims = claims[:8]
    logger.info(
        "Invoking grading LLM answer_chars=%d doc_count=%d claim_count=%d",
        len(answer.strip()),
        len(retrieved_docs),
        len(claims),
    )
    prompt = _build_grading_prompt()
    llm = get_llm("grading")
    structured_llm = llm.with_structured_output(GradingResult)
    chain = prompt | structured_llm
    result = chain.invoke(
        {
            "answer": answer,
            "claims": _format_claims(claims),
            "context": _format_retrieved_docs(retrieved_docs),
        }
    )
    normalized_result = _normalize_result(result)
    if _should_escalate(normalized_result):
        logger.info(
            "Grading escalation hook triggered but no fallback model is configured"
        )
    return normalized_result


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
                claim_count=0,
            )
        elif not retrieved_docs:
            logger.info(
                "Skipping grading LLM because no retrieved documents are available"
            )
            result = GradingResult(
                score=0.0,
                verdict="unsupported",
                explanation="No retrieved documents were available for grounding.",
                claim_count=max(1, len(_split_into_claims(answer))),
            )
        else:
            result = _normalize_result(_grade_answer(answer, retrieved_docs))

        score = _compute_score(result.score)
        unsupported_claim_count = len(result.unsupported_claims)
        claim_count = result.claim_count
        latency = round((time.time() - start) * 1000, 2)

        logger.info(
            "Grading completed score=%.2f verdict=%s unsupported_claims=%d claim_count=%d doc_count=%d explanation=%s",
            score,
            result.verdict,
            unsupported_claim_count,
            claim_count,
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
                        "claim_count": claim_count,
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
                        "unsupported_claim_count": 0,
                        "claim_count": 0,
                    },
                )
            ],
        }
