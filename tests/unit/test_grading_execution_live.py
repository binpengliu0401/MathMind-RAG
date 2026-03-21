import pytest
from langchain_core.documents import Document

from app.nodes.grading import grade_hallucination
from app.utils.constants import LLM_API_KEY


pytestmark = pytest.mark.skipif(
    not LLM_API_KEY,
    reason="LLM_API_KEY is not configured for live grading execution tests.",
)


CASE_DATA = [
    {
        "name": "fully_grounded_rag_definition",
        "question": "What is Retrieval-Augmented Generation (RAG), and why is it useful?",
        "retrieved_docs": [
            "[DOC 1]\nRetrieval-Augmented Generation (RAG) is a method that combines information retrieval with text generation. Instead of relying only on a model's parametric memory, RAG retrieves external documents relevant to a user query and uses them as evidence during answer generation.",
            "[DOC 2]\nRAG is useful because it can improve factual grounding and reduce unsupported answers. By conditioning generation on retrieved evidence, the system can produce responses that are more transparent and easier to verify.",
        ],
        "answer": "Retrieval-Augmented Generation (RAG) combines document retrieval with text generation. It first retrieves relevant external documents for a user query and then uses those documents as evidence when generating an answer. This is useful because it improves factual grounding and helps reduce unsupported answers by making the response easier to verify.",
    },
    {
        "name": "partially_grounded_rag_definition",
        "question": "What is Retrieval-Augmented Generation (RAG), and why is it useful?",
        "retrieved_docs": [
            "[DOC 1]\nRetrieval-Augmented Generation (RAG) is a method that combines information retrieval with text generation. Instead of relying only on a model's parametric memory, RAG retrieves external documents relevant to a user query and uses them as evidence during answer generation.",
            "[DOC 2]\nRAG is useful because it can improve factual grounding and reduce unsupported answers. By conditioning generation on retrieved evidence, the system can produce responses that are more transparent and easier to verify.",
        ],
        "answer": "Retrieval-Augmented Generation (RAG) combines retrieval and generation by bringing in external documents during answering. It is useful because it improves factual grounding and makes responses easier to verify. In addition, RAG lowers computational cost dramatically and guarantees that the final answer is always correct.",
    },
    {
        "name": "unsupported_rag_definition",
        "question": "What is Retrieval-Augmented Generation (RAG), and why is it useful?",
        "retrieved_docs": [
            "[DOC 1]\nRetrieval-Augmented Generation (RAG) is a method that combines information retrieval with text generation. Instead of relying only on a model's parametric memory, RAG retrieves external documents relevant to a user query and uses them as evidence during answer generation.",
            "[DOC 2]\nRAG is useful because it can improve factual grounding and reduce unsupported answers. By conditioning generation on retrieved evidence, the system can produce responses that are more transparent and easier to verify.",
        ],
        "answer": "RAG was invented by OpenAI in 2024 as a replacement for transformers. It stores all world knowledge in a symbolic database and eliminates the need for retrieval entirely. Its main advantage is that it guarantees perfect accuracy in medical and legal domains.",
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


def _evaluate_case(case: dict) -> dict:
    result = grade_hallucination(make_generation_state(case))  # type: ignore
    trace_entry = result["execution_trace"][0]
    print(
        f"\n[{case['name']}] "
        f"score={result['hallucination_score']:.2f} "
        f"verdict={trace_entry['key_output'].get('verdict', 'n/a')} "
        f"unsupported_claim_count={trace_entry['key_output'].get('unsupported_claim_count', 'n/a')}"
    )
    print(f"summary={trace_entry['summary']}")
    if "error_message" in result:
        print(f"error={result['error_message']}")
    return result


@pytest.mark.parametrize("case", CASE_DATA, ids=[case["name"] for case in CASE_DATA])
def test_live_grading_execution_reports_real_result(case):
    result = _evaluate_case(case)

    assert "error_message" not in result
    assert isinstance(result["hallucination_score"], float)
    assert 0.0 <= result["hallucination_score"] <= 1.0
    assert result["execution_trace"][0]["node"] == "grading"
    assert result["execution_trace"][0]["status"] == "success"


def test_live_grading_execution_preserves_expected_ordering():
    fully_grounded = _evaluate_case(CASE_DATA[0])
    partially_grounded = _evaluate_case(CASE_DATA[1])
    unsupported = _evaluate_case(CASE_DATA[2])

    assert fully_grounded["hallucination_score"] > partially_grounded["hallucination_score"]
    assert partially_grounded["hallucination_score"] > unsupported["hallucination_score"]
