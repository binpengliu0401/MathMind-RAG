# ============================================================
# Test for: Node 1 — Query Rewriting
# Run:      python -m pytest tests/eval_rewriting_assert.py -v
# Owner:    Member C (Chen)
# ============================================================

import pytest
from app.nodes.rewriting import rewrite_query


def make_state(query: str, failed_queries: list = None) -> dict: # type: ignore
    """Build a minimal state dict for testing."""
    state = {"query": query}
    if failed_queries is not None:
        state["failed_queries"] = failed_queries # type: ignore
    return state


def test_case1_remove_colloquial_filler():
    """测试小白问题：必须去除口语化废话，并尽量提取或保留核心学术动作"""
    state = make_state("how to make the model reason step by step without giving it any examples?")
    result = rewrite_query(state) # type: ignore
    rewritten = result["rewritten_query"].lower()
    print(f"\ncase 1 [真实改写结果]: {rewritten}")

    # 断言 1：不能包含低级口语词
    assert "how to make" not in rewritten
    assert "without giving it" not in rewritten
    # 断言 2：必须保留或映射出核心学术概念
    assert "step-by-step" in rewritten or "zero-shot" in rewritten or "chain of thought" in rewritten


def test_case2_preserve_perfect_query():
    """测试完美问题（不惩罚好用户）：原问题极好时，不能过度压缩，必须保留核心实体"""
    original = "what is the MMLU benchmark with 57 tasks for evaluating massive multitask language understanding?"
    state = make_state(original)
    result = rewrite_query(state) # type: ignore
    rewritten = result["rewritten_query"].lower()
    print(f"\ncase 2 [真实改写结果]: {rewritten}")

    # 断言：必须保留具体的专有名词和数字，不能被无脑压缩掉
    assert "mmlu" in rewritten
    assert "57 tasks" in rewritten or "57" in rewritten
    # 不应该比原问题短太多（防止过度压缩）
    assert len(rewritten) > len(original) * 0.5


def test_case3_jargon_preservation():
    """测试学术黑话保全：绝对不能把行业的约定俗成替换成普通的同义词"""
    state = make_state("combining reasoning and acting in LLMs")
    result = rewrite_query(state) # type: ignore
    rewritten = result["rewritten_query"].lower()
    print(f"\ncase 3 [真实改写结果]: {rewritten}")

    # 断言：必须原封不动地保留 "reasoning" 和 "acting"
    assert "reasoning" in rewritten
    assert "acting" in rewritten or "action" in rewritten


def test_case4_retry_logic_avoids_failed_history():
    """测试反思重试逻辑：必须避开 failed_queries 中的词汇，实现视角转换"""
    original = "math problem solving algorithms"
    failed_history = ["math problem solving algorithms", "mathematical logic reasoning neural networks"]
    state = make_state(original, failed_queries=failed_history)

    result = rewrite_query(state) # type: ignore
    rewritten = result["rewritten_query"].lower()
    print(f"\ncase 4 [真实改写结果]: {rewritten}")

    # 断言：新的 query 绝对不能和失败历史上任何一条完全一致
    for failed in failed_history:
        assert rewritten != failed.lower()

    # 断言：它必须依然包含对原始问题的某种呼应（不能完全跑题）
    assert "math" in rewritten or "algorithm" in rewritten or "deduction" in rewritten or "reasoning" in rewritten


def test_returns_required_contract_keys():
    """测试契约：必须严格遵守 GraphState 的输入输出规范"""
    state = make_state("test query")
    result = rewrite_query(state) # type: ignore

    assert "rewritten_query" in result
    assert "failed_queries" in result
    assert "execution_trace" in result
    assert len(result["execution_trace"]) == 1
