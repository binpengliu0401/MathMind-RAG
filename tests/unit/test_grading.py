# ============================================================
# Test for: Node 4 — Hallucination Grading
# Run:      python -m pytest tests/test_grading.py -v
# Owner:    Member D (Hu) — run this after your implementation
# ============================================================

from unittest.mock import patch

from langchain_core.documents import Document

from app.nodes import grading
from app.nodes.grading import grade_hallucination


def make_state(answer: str, retrieved_docs: list) -> dict:  # type: ignore
    """Build a minimal state dict for testing."""
    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs,
    }


def make_document(content: str) -> Document:  # type: ignore
    """Helper to create a test Document."""
    return Document(page_content=content, metadata={"source": "test_doc"})


@patch("app.nodes.grading._grade_answer")
def test_returns_required_keys(mock_grade_answer):
    """Output dict must contain: hallucination_score, execution_trace."""
    mock_grade_answer.return_value = grading.GradingResult(
        score=0.82,
        verdict="grounded",
        explanation="The answer is supported by the retrieved document.",
        unsupported_claims=[],
    )

    state = make_state(
        "RAG improves answer grounding.",
        [make_document("RAG uses retrieved documents to ground the answer.")],
    )
    result = grade_hallucination(state)  # type: ignore

    assert "hallucination_score" in result
    assert "execution_trace" in result


@patch("app.nodes.grading._grade_answer")
def test_score_is_float_between_0_and_1(mock_grade_answer):
    """hallucination_score must be a float in range [0.0, 1.0]."""
    mock_grade_answer.return_value = grading.GradingResult(
        score=1.02,
        verdict="grounded",
        explanation="The answer is fully grounded.",
        unsupported_claims=[],
    )

    state = make_state(
        "RAG improves answer grounding.",
        [make_document("RAG uses retrieved documents to ground the answer.")],
    )
    result = grade_hallucination(state)  # type: ignore

    assert isinstance(result["hallucination_score"], float)
    assert 0.0 <= result["hallucination_score"] <= 1.0
    assert result["hallucination_score"] == 1.0


@patch("app.nodes.grading._grade_answer")
def test_grounded_answer_scores_high(mock_grade_answer):
    """An answer directly based on retrieved docs should score >= 0.7."""
    mock_grade_answer.return_value = grading.GradingResult(
        score=0.88,
        verdict="grounded",
        explanation="The answer is directly supported by the retrieved document.",
        unsupported_claims=[],
    )

    state = make_state(
        "RAG retrieves relevant documents before generation.",
        [make_document("RAG systems retrieve relevant documents before generating an answer.")],
    )
    result = grade_hallucination(state)  # type: ignore

    assert result["hallucination_score"] >= 0.7


@patch("app.nodes.grading._grade_answer")
def test_ungrounded_answer_scores_low(mock_grade_answer):
    """An answer unrelated to retrieved docs should score < 0.7."""
    mock_grade_answer.return_value = grading.GradingResult(
        score=0.21,
        verdict="unsupported",
        explanation="The answer includes claims not supported by the retrieved documents.",
        unsupported_claims=["The answer claims benchmark results not found in the docs."],
    )

    state = make_state(
        "This paper proves Transformers outperform every baseline on MMLU.",
        [make_document("The retrieved document discusses retrieval-augmented generation basics.")],
    )
    result = grade_hallucination(state)  # type: ignore

    assert result["hallucination_score"] < 0.7


@patch("app.nodes.grading._grade_answer")
def test_trace_entry_format(mock_grade_answer):
    """execution_trace must have exactly 1 entry with keys:
    node, status, latency_ms, summary, key_output."""
    mock_grade_answer.return_value = grading.GradingResult(
        score=0.74,
        verdict="grounded",
        explanation="Most claims are supported by the retrieved documents.",
        unsupported_claims=[],
    )

    state = make_state(
        "RAG combines retrieval and generation.",
        [make_document("RAG combines retrieval with language generation.")],
    )
    result = grade_hallucination(state)  # type: ignore

    trace = result["execution_trace"]
    assert isinstance(trace, list)
    assert len(trace) == 1

    entry = trace[0]
    assert entry["node"] == "grading"
    assert "status" in entry
    assert "latency_ms" in entry
    assert "summary" in entry
    assert "key_output" in entry
    assert "hallucination_score" in entry["key_output"]


@patch("app.nodes.grading._grade_answer", side_effect=RuntimeError("LLM unavailable"))
def test_failure_returns_zero_score_and_error_message(mock_grade_answer):
    state = make_state(
        "RAG combines retrieval and generation.",
        [make_document("RAG combines retrieval with language generation.")],
    )
    result = grade_hallucination(state)  # type: ignore

    assert result["hallucination_score"] == 0.0
    assert "error_message" in result
    assert result["execution_trace"][0]["status"] == "error"
    mock_grade_answer.assert_called_once()


@patch("app.nodes.grading._grade_answer")
def test_empty_docs_does_not_raise(mock_grade_answer):
    state = make_state("RAG combines retrieval and generation.", [])
    result = grade_hallucination(state)  # type: ignore

    assert result["hallucination_score"] == 0.0
    assert "error_message" not in result
    assert result["execution_trace"][0]["status"] == "success"
    mock_grade_answer.assert_not_called()


@patch("app.nodes.grading._grade_answer")
def test_empty_answer_does_not_raise(mock_grade_answer):
    state = make_state("", [make_document("RAG combines retrieval with generation.")])
    result = grade_hallucination(state)  # type: ignore

    assert result["hallucination_score"] == 0.0
    assert "error_message" not in result
    assert result["execution_trace"][0]["status"] == "success"
    mock_grade_answer.assert_not_called()
