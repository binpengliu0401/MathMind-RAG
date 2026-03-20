# ============================================================
# Test for: Node 1 — Query Rewriting
# Run:      python -m pytest tests/test_rewriting.py -v
# Owner:    Member C (Chen) — run this after your implementation
# ============================================================

import pytest
from app.nodes.rewriting import rewrite_query


def make_state(query: str, failed_queries: list = None) -> dict:  # type: ignore
    """Build a minimal state dict for testing."""
    state = {"query": query}
    if failed_queries is not None:
        state["failed_queries"] = failed_queries  # type: ignore
    return state


def test_returns_required_keys():
    """Output dict must contain: rewritten_query, failed_queries, execution_trace."""
    state = make_state("What is the attention mechanism?")
    result = rewrite_query(state)  # type: ignore

    assert "rewritten_query" in result
    assert "failed_queries" in result
    assert "execution_trace" in result


def test_rewritten_query_is_nonempty_string():
    """rewritten_query must be a non-empty string."""
    state = make_state("Explain transformer models")
    result = rewrite_query(state)  # type: ignore

    assert isinstance(result["rewritten_query"], str)
    assert len(result["rewritten_query"].strip()) > 0


def test_rewritten_query_differs_from_original():
    """Rewritten query should not be identical to the original input (unless fallback occurs)."""
    original = "how does that 17 year vaswani model prevent seeing future?"
    state = make_state(original)
    result = rewrite_query(state)  # type: ignore

    # +++ 打印错误信息 +++
    if "error_message" in result:
        print(f"\n[致命错误提示] 大模型调用失败: {result['error_message']}")

    # 因为我们使用了大模型提取关键词，正常情况下一定会被改写
    assert result["rewritten_query"] != original


def test_failed_queries_contains_new_query():
    """failed_queries must include the newly rewritten query (for LangGraph to append)."""
    state = make_state("What dataset trained GPT-3?")
    result = rewrite_query(state)  # type: ignore

    new_query = result["rewritten_query"]
    assert len(result["failed_queries"]) == 1
    assert result["failed_queries"][0] == new_query


def test_retry_avoids_previous_queries():
    """On retry, rewritten_query must not repeat any query in failed_queries."""
    original = "What is Scaling Law?"
    # 模拟上一轮失败的历史记录
    fake_failed_history = ["Scaling Law language model", "Scaling Law neural network"]

    state = make_state(original, failed_queries=fake_failed_history)
    result = rewrite_query(state)  # type: ignore

    new_query = result["rewritten_query"]
    # 大模型必须进行反思，给出与历史不同的查询词
    assert new_query not in fake_failed_history


def test_trace_entry_format():
    """execution_trace must have exactly 1 entry with keys:
    node, status, latency_ms, summary, key_output."""
    state = make_state("Test execution trace format")
    result = rewrite_query(state)  # type: ignore

    trace = result["execution_trace"]
    assert isinstance(trace, list)
    assert len(trace) == 1

    entry = trace[0]
    assert entry["node"] == "rewriting"
    assert "status" in entry
    assert "latency_ms" in entry
    assert "summary" in entry
    assert "key_output" in entry
    assert "rewritten_query" in entry["key_output"]
