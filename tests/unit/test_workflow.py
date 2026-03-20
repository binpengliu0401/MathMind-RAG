# Owner: Liu
# Responsibility: End-to-end workflow tests for the full RAG graph

import pytest
from unittest.mock import patch, MagicMock
from app.main import run_workflow


# Mock helper 

def make_mock_chain(answer: str = "This is a fake answer for testing."):
    # Build a fake chain that mimics GENERATION_PROMPT | llm behavior
    fake_response = MagicMock()
    fake_response.content = answer

    fake_chain = MagicMock()
    fake_chain.invoke.return_value = fake_response

    return fake_chain


def patch_generation(mock_get_llm, mock_prompt, answer: str = "This is a fake answer for testing."):
    # Apply mock to both get_llm and GENERATION_PROMPT to intercept the | chain
    fake_chain = make_mock_chain(answer)
    mock_prompt.__or__ = MagicMock(return_value=fake_chain)
    mock_get_llm.return_value = MagicMock()


# Test

@patch("app.nodes.generation.GENERATION_PROMPT")
@patch("app.nodes.generation.get_llm")
def test_graph_runs_without_error(mock_get_llm, mock_prompt):
    """Full graph should run end-to-end without raising any exception."""
    patch_generation(mock_get_llm, mock_prompt)
    result = run_workflow("What is the Transformer architecture?")
    assert result is not None


@patch("app.nodes.generation.GENERATION_PROMPT")
@patch("app.nodes.generation.get_llm")
def test_result_contains_required_fields(mock_get_llm, mock_prompt):
    """Final state must contain all required output fields."""
    patch_generation(mock_get_llm, mock_prompt)
    result = run_workflow("What is attention mechanism?")

    assert "answer" in result
    assert "hallucination_score" in result
    assert "retry_count" in result
    assert "final_decision" in result
    assert "execution_trace" in result
    assert "rewritten_query" in result


@patch("app.nodes.generation.GENERATION_PROMPT")
@patch("app.nodes.generation.get_llm")
def test_stub_grading_triggers_retry(mock_get_llm, mock_prompt):
    """Grading stub returns 0.5 < threshold 0.7, router should trigger retry."""
    patch_generation(mock_get_llm, mock_prompt)
    result = run_workflow("What is BERT?")

    assert result["retry_count"] > 0


@patch("app.nodes.generation.GENERATION_PROMPT")
@patch("app.nodes.generation.get_llm")
def test_stub_grading_triggers_stop(mock_get_llm, mock_prompt):
    """After max retries, final_decision should be stop."""
    patch_generation(mock_get_llm, mock_prompt)
    result = run_workflow("What is GPT?")

    assert result["final_decision"] == "stop"


@patch("app.nodes.generation.GENERATION_PROMPT")
@patch("app.nodes.generation.get_llm")
def test_execution_trace_has_all_nodes(mock_get_llm, mock_prompt):
    """Trace should contain entries for all 5 nodes."""
    patch_generation(mock_get_llm, mock_prompt)
    result = run_workflow("What is ResNet?")

    node_names = [entry["node"] for entry in result["execution_trace"]]
    assert "rewriting" in node_names
    assert "retrieval" in node_names
    assert "generation" in node_names
    assert "grading" in node_names
    assert "router" in node_names


@patch("app.nodes.generation.GENERATION_PROMPT")
@patch("app.nodes.generation.get_llm")
def test_answer_is_nonempty_string(mock_get_llm, mock_prompt):
    """answer field should be a non-empty string."""
    patch_generation(mock_get_llm, mock_prompt, answer="This is a fake answer for testing.")
    result = run_workflow("What is self-attention?")

    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0