# ============================================================
# Test for: Node 4 — Hallucination Grading
# Run:      python -m pytest tests/test_grading.py -v
# Owner:    Member D (Hu) — run this after your implementation
# ============================================================

import pytest
from langchain_core.documents import Document
from app.nodes.grading import grade_hallucination


def make_state(answer: str, retrieved_docs: list) -> dict:
    """Build a minimal state dict for testing."""
    pass


def make_document(content: str) -> Document:
    """Helper to create a test Document."""
    pass


def test_returns_required_keys():
    """Output dict must contain: hallucination_score, execution_trace."""
    pass


def test_score_is_float_between_0_and_1():
    """hallucination_score must be a float in range [0.0, 1.0]."""
    pass


def test_grounded_answer_scores_high():
    """An answer directly based on retrieved docs should score >= 0.7."""
    pass


def test_ungrounded_answer_scores_low():
    """An answer unrelated to retrieved docs should score < 0.7."""
    pass


def test_trace_entry_format():
    """execution_trace must have exactly 1 entry with keys:
    node, status, latency_ms, summary, key_output."""
    pass