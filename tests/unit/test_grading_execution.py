from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.nodes.grading import GradingResult, grade_hallucination

CASE_DATA = [
    {
        "name": "fully_grounded_rag_definition",
        "question": "What is Retrieval-Augmented Generation (RAG), and why is it useful?",
        "retrieved_docs": [
            "[DOC 1]\nRetrieval-Augmented Generation (RAG) is a method that combines information retrieval with text generation. Instead of relying only on a model's parametric memory, RAG retrieves external documents relevant to a user query and uses them as evidence during answer generation.",
            "[DOC 2]\nRAG is useful because it can improve factual grounding and reduce unsupported answers. By conditioning generation on retrieved evidence, the system can produce responses that are more transparent and easier to verify.",
        ],
        "answer": "Retrieval-Augmented Generation (RAG) combines document retrieval with text generation. It first retrieves relevant external documents for a user query and then uses those documents as evidence when generating an answer. This is useful because it improves factual grounding and helps reduce unsupported answers by making the response easier to verify.",
        "expected_verdict": "grounded",
        "expected_score_range": {"min": 0.85, "max": 1.0},
        "unsupported_claims_truth": [],
    },
    {
        "name": "partially_grounded_rag_definition",
        "question": "What is Retrieval-Augmented Generation (RAG), and why is it useful?",
        "retrieved_docs": [
            "[DOC 1]\nRetrieval-Augmented Generation (RAG) is a method that combines information retrieval with text generation. Instead of relying only on a model's parametric memory, RAG retrieves external documents relevant to a user query and uses them as evidence during answer generation.",
            "[DOC 2]\nRAG is useful because it can improve factual grounding and reduce unsupported answers. By conditioning generation on retrieved evidence, the system can produce responses that are more transparent and easier to verify.",
        ],
        "answer": "Retrieval-Augmented Generation (RAG) combines retrieval and generation by bringing in external documents during answering. It is useful because it improves factual grounding and makes responses easier to verify. In addition, RAG lowers computational cost dramatically and guarantees that the final answer is always correct.",
        "expected_verdict": "partially_grounded",
        "expected_score_range": {"min": 0.4, "max": 0.7},
        "unsupported_claims_truth": [
            "RAG lowers computational cost dramatically.",
            "RAG guarantees that the final answer is always correct.",
        ],
    },
    {
        "name": "unsupported_rag_definition",
        "question": "What is Retrieval-Augmented Generation (RAG), and why is it useful?",
        "retrieved_docs": [
            "[DOC 1]\nRetrieval-Augmented Generation (RAG) is a method that combines information retrieval with text generation. Instead of relying only on a model's parametric memory, RAG retrieves external documents relevant to a user query and uses them as evidence during answer generation.",
            "[DOC 2]\nRAG is useful because it can improve factual grounding and reduce unsupported answers. By conditioning generation on retrieved evidence, the system can produce responses that are more transparent and easier to verify.",
        ],
        "answer": "RAG was invented by OpenAI in 2024 as a replacement for transformers. It stores all world knowledge in a symbolic database and eliminates the need for retrieval entirely. Its main advantage is that it guarantees perfect accuracy in medical and legal domains.",
        "expected_verdict": "unsupported",
        "expected_score_range": {"min": 0.0, "max": 0.25},
        "unsupported_claims_truth": [
            "RAG was invented by OpenAI in 2024.",
            "RAG is a replacement for transformers.",
            "It stores all world knowledge in a symbolic database.",
            "It eliminates the need for retrieval entirely.",
            "It guarantees perfect accuracy in medical and legal domains.",
        ],
    },
]


def make_document(raw_doc: str) -> Document:
    lines = raw_doc.splitlines()
    source = lines[0].strip()[1:-1] if lines else "DOC"
    content = "\n".join(lines[1:]).strip()
    return Document(page_content=content, metadata={"source": source})


def make_generation_state(case: dict) -> dict:  # type: ignore
    return {
        "query": case["question"],
        "answer": case["answer"],
        "retrieved_docs": [make_document(raw_doc) for raw_doc in case["retrieved_docs"]],
    }


def make_mock_grading_result(case: dict) -> GradingResult:
    midpoint_score = (
        case["expected_score_range"]["min"] + case["expected_score_range"]["max"]
    ) / 2
    verdict = case["expected_verdict"]
    unsupported_claims = case["unsupported_claims_truth"]
    if unsupported_claims:
        explanation = (
            "Some answer claims are supported, but unsupported claims remain in the response."
            if verdict == "partially_grounded"
            else "The answer is not supported by the retrieved documents."
        )
    else:
        explanation = "The answer is fully supported by the retrieved documents."

    return GradingResult(
        score=midpoint_score,
        verdict=verdict,
        explanation=explanation,
        unsupported_claims=unsupported_claims,
    )


@pytest.mark.parametrize("case", CASE_DATA, ids=[case["name"] for case in CASE_DATA])
@patch("app.nodes.grading._build_grading_prompt")
@patch("app.nodes.grading.get_llm")
def test_grading_execution_from_generation_outputs(
    mock_get_llm, mock_build_prompt, case, caplog
):
    fake_prompt = MagicMock()
    fake_chain = MagicMock()
    fake_prompt.__or__.return_value = fake_chain
    mock_build_prompt.return_value = fake_prompt

    fake_llm = MagicMock()
    fake_structured_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured_llm
    mock_get_llm.return_value = fake_llm

    fake_chain.invoke.return_value = make_mock_grading_result(case)

    state = make_generation_state(case)

    with caplog.at_level("INFO"):
        result = grade_hallucination(state)  # type: ignore

    assert case["expected_score_range"]["min"] <= result["hallucination_score"] <= case["expected_score_range"]["max"]
    assert result["execution_trace"][0]["node"] == "grading"
    assert result["execution_trace"][0]["status"] == "success"
    assert result["execution_trace"][0]["key_output"]["verdict"] == case["expected_verdict"]
    assert (
        result["execution_trace"][0]["key_output"]["unsupported_claim_count"]
        == len(case["unsupported_claims_truth"])
    )

    fake_llm.with_structured_output.assert_called_once()
    fake_chain.invoke.assert_called_once()
    invoke_payload = fake_chain.invoke.call_args.args[0]
    assert invoke_payload["answer"] == case["answer"]
    assert "[DOC 1] source=DOC 1" in invoke_payload["context"]
    assert "[DOC 2] source=DOC 2" in invoke_payload["context"]

    logs = caplog.text
    assert "Grading started" in logs
    assert "Invoking grading LLM" in logs
    assert f"verdict={case['expected_verdict']}" in logs
    assert f"unsupported_claims={len(case['unsupported_claims_truth'])}" in logs


def test_grading_execution_logs_empty_input_short_circuit(caplog):
    state = {
        "query": "What is Retrieval-Augmented Generation (RAG), and why is it useful?",
        "answer": "",
        "retrieved_docs": [make_document(CASE_DATA[0]["retrieved_docs"][0])],
    }

    with caplog.at_level("INFO"):
        result = grade_hallucination(state)  # type: ignore

    assert result["hallucination_score"] == 0.0
    assert result["execution_trace"][0]["status"] == "success"
    assert "Skipping grading LLM because answer is empty" in caplog.text
