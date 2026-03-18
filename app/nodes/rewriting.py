# ============================================================
# Owner:        Member C (Chen)
# Responsibility: Query Rewriting Node (Node 1)
# Status:       Stub — awaiting implementation by Member C
# ============================================================
#
# YOUR TASK:
#   Implement the rewrite_query() function below.
#
# INPUT  (read from state):
#   - state["query"]          str   original user question
#   - state["failed_queries"] list  queries already tried (empty on first run)
#                                   use this on retry to avoid repeating same rewrite
#
# OUTPUT (return as dict):
#   - "rewritten_query"  str        the improved query for retrieval
#   - "failed_queries"   list[str]  append the NEW rewritten query to this list
#   - "execution_trace"  list[dict] append one trace entry (see format below)
#
# TRACE FORMAT (append exactly one entry per call):
#   {
#       "node": "rewriting",
#       "status": "success",          # or "error"
#       "latency_ms": 123,
#       "summary": "Rewrote query: ...",
#       "key_output": {
#           "rewritten_query": "..."
#       }
#   }
#
# IMPORTANT:
#   - Do NOT raise exceptions — write to state["error_message"] and return
#   - Do NOT modify state directly — always return a dict
#   - "failed_queries" uses Annotated[list, operator.add] — just return a
#     list with the new query, LangGraph will auto-append it
#
# ============================================================

import time
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import GraphState
from app.utils.tracer import build_trace_entry
from app.services.llm_service import get_llm


# ------------------------------------------------------------
# 1. 定义强制大模型输出的 JSON 数据结构 (Structured Output)
# ------------------------------------------------------------
class RewriteOutput(BaseModel):
    """强制大模型输出的结构体，只允许包含一句完整的学术检索陈述/问句"""
    search_phrase: str = Field(
        description="A concise, highly professional academic search phrase optimized for dense vector search. Do NOT output a list of keywords. Extract the true academic intent and correct any colloquial or inaccurate terms."
    )


# ------------------------------------------------------------
# 2. 定义纯英文的 System Prompts (适应英文学术语境)
# ------------------------------------------------------------

# 原版
# FIRST_TRY_SYSTEM_PROMPT = """You are an expert academic search query rewriting assistant.
# Your task is to extract the most important entities, academic terms, and concepts from the user's vague query.
# Return a list of precise keywords optimized for a semantic vector database search (e.g., FAISS).
# Ignore conversational filler words. Focus on algorithms, authors, metrics, and core mechanisms."""

# # 改版
# FIRST_TRY_SYSTEM_PROMPT = """You are an expert academic search query rewriting assistant.
# Your task is to rewrite the user's vague query into a highly specific, descriptive academic search query.
# Instead of just listing disconnected keywords, combine the core entities, authors, and technical terms into a coherent search phrase optimized for dense vector retrieval.
# Do not use conversational filler words like "tell me" or "what is"."""

# # T 提议加强版
# FIRST_TRY_SYSTEM_PROMPT = """You are an expert academic search query rewriting assistant.
# Your task is to rewrite the user's vague query into a highly specific, descriptive academic search phrase optimized for dense vector retrieval.
# Instead of listing disconnected keywords, seamlessly combine the core entities—specifically focusing on algorithms, authors, evaluation metrics, and core mechanisms—into a single coherent semantic phrase.
# Strictly exclude conversational filler words (e.g., 'tell me about', 'what is', 'how does')."""

# # AI/Math 改进版
# FIRST_TRY_SYSTEM_PROMPT = """You are an expert academic search query rewriting assistant specializing in AI, Large Language Models (LLMs), and Mathematical Reasoning.
# Your task is to rewrite the user's vague query into a highly specific, descriptive academic search phrase optimized for dense vector retrieval.
# Instead of listing disconnected keywords, seamlessly combine the core entities—specifically focusing on AI models (e.g., PaLM, GPT-3), prompting techniques (e.g., Chain of Thought, ReAct), benchmark datasets (e.g., GSM8K, MMLU), and core mathematical reasoning mechanisms—into a single coherent semantic phrase.
# Strictly exclude conversational filler words (e.g., 'tell me about', 'what is', 'how does')."""

# # 防向量稀释版
# FIRST_TRY_SYSTEM_PROMPT = """You are an expert academic search query rewriting assistant.
# Your task is to rewrite the user's vague query into a highly specific, descriptive academic search phrase.
# CRITICAL CONSTRAINTS:
# 1. Do not use conversational filler words (e.g., 'what is', 'tell me').
# 2. Keep it concise. Output a MAXIMUM of 4 to 6 core words or entities. Vector databases perform poorly with overly long keyword stuffing."""

# # 稠密向量语义优化版 (Dense Semantic Rewrite)
# FIRST_TRY_SYSTEM_PROMPT = """You are an expert academic search assistant specializing in AI and Big Data.
# The user will provide a vague query. Your task is to rewrite it into a highly professional, coherent academic statement or precise question that captures the core semantic intent.
#
# CRITICAL CONSTRAINTS FOR DENSE VECTOR RETRIEVAL (e.g., BGE/BERT):
# 1. DO NOT output a disconnected list of keywords. Output a natural, flowing academic sentence.
# 2. DO NOT hallucinate specific datasets (e.g., GSM8K), model names (e.g., GPT-3), or metrics UNLESS they are heavily implied by the user's query.
# 3. Translate informal words into academic terminology (e.g., "make model think" -> "elicit reasoning capabilities").
# 4. Ensure the output resembles a sentence you would find in an abstract of a paper.
#
# Output ONLY the rewritten sentence."""

# 最终版
FIRST_TRY_SYSTEM_PROMPT = """You are an expert AI/Math academic search assistant.
Your task is to optimize the user's query for a dense vector database (BGE/BERT).

CRITICAL CONSTRAINTS:
1. ALIGN & CORRECT TERMINOLOGY: Extract the true academic intent. If the user uses colloquialisms or inaccurate terms (e.g., calling an API interaction "reinforcement learning"), correct it to the standard academic jargon. If they use correct jargon, keep it.
2. EXPAND INTELLIGENTLY: Add 1 or 2 universally accepted academic terms/mechanisms that directly underlie the user's concept.
3. NO HALLUCINATION: DO NOT add specific dataset names (e.g., GSM8K) or model names (e.g., GPT-3) UNLESS the user explicitly mentioned them.
4. CONCISE FORMAT: Output a short, punchy search phrase (e.g., "zero-shot chain of thought reasoning"). NO conversational filler ("What is", "How to").
5. If the user's original query is already a highly professional, well-structured academic phrase (e.g., containing specific metrics, full dataset names), DO NOT compress it. Act as a pass-through and retain its full descriptive richness.

Output ONLY the optimized search phrase."""

# # 原版
# RETRY_SYSTEM_PROMPT = """You are an expert academic search query rewriting assistant.
# The user previously searched for: "{original_query}"
# The system already tried the following queries but failed to find relevant documents:
# {failed_queries_str}
#
# You MUST change the search strategy. Think of alternative academic synonyms, broader concepts, narrower concepts, or related underlying technologies.
# Provide a completely new set of keywords that do NOT overlap entirely with the failed attempts."""

# # 修改版
# RETRY_SYSTEM_PROMPT = """You are an expert academic search query rewriting assistant.
# The user previously searched for: "{original_query}"
# The system already tried the following queries but failed to find relevant documents:
# {failed_queries_str}
#
# Since the previous general terms failed, you MUST change the search strategy.
# Do NOT use abstract or overly broad terms. Instead, dive deeper into specific sub-components, exact mathematical variables, specific algorithms, or niche metrics related to the original query.
# Generate a highly specific, coherent search phrase that avoids the failed terms but targets the deep technical details."""

# # 防向量稀释版
# RETRY_SYSTEM_PROMPT = """You are an expert academic search query rewriting assistant.
# The user previously searched for: "{original_query}"
# Failed attempts: {failed_queries_str}
#
# Since previous terms failed, you MUST dive deeper into specific mathematical variables, niche metrics, or exact sub-components.
# CRITICAL CONSTRAINTS:
# 1. Avoid broad, abstract terms.
# 2. Output a MAXIMUM of 3 to 5 highly specific words.
#
# EXAMPLE:
# User: neural network performance
# Failed: accuracy, loss
# Good Output: F1-score perplexity cross-entropy"""

# # AI/Math 改进版
# RETRY_SYSTEM_PROMPT = """You are an expert academic search query rewriting assistant specializing in AI and LLMs.
# The user previously searched for: "{original_query}"
# Failed attempts: {failed_queries_str}
#
# Since previous terms failed, you MUST dive deeper into niche evaluation metrics, exact algorithms, specific datasets, or mathematical techniques related to the query.
# CRITICAL CONSTRAINTS:
# 1. Avoid broad, abstract AI terms.
# 2. Output a MAXIMUM of 3 to 5 highly specific technical words.
#
# EXAMPLE:
# User: LLM math test
# Failed: math dataset evaluation, LLM mathematical ability
# Good Output: GSM8K AQUA-RAT MultiArith benchmark"""

# 稠密向量语义优化版 - 反思重试 (Dense Semantic Rewrite - Retry)
RETRY_SYSTEM_PROMPT = """You are an expert academic search assistant specializing in AI and Big Data.
The user previously searched for: "{original_query}"
The system already tried the following queries but failed to find relevant documents:
{failed_queries_str}

Since previous attempts failed, you MUST change the semantic perspective.
CRITICAL CONSTRAINTS FOR DENSE VECTOR RETRIEVAL (e.g., BGE/BERT):
1. Pivot the terminology: Use alternative academic synonyms, broader conceptual framing, or describe the underlying mechanism differently.
2. DO NOT hallucinate specific datasets, model names, or niche metrics unless explicitly requested by the user. Abstracts often do not contain them.
3. Output a single, natural, flowing academic sentence or phrase (similar to what you would read in a paper's abstract).
4. DO NOT output a disconnected list of keywords.

Output ONLY the new rewritten sentence."""


# ------------------------------------------------------------
# 3. 核心节点函数
# ------------------------------------------------------------
def rewrite_query(state: GraphState) -> dict:
    """
    Node 1: Query Rewriting
    Rewrites the user query to improve retrieval quality.
    On retry, uses failed_queries to avoid repeating previous attempts.
    """
    start = time.time()

    # 防御性编程：安全获取原始问题和失败历史
    original_query = state.get("query", "")
    failed_queries = state.get("failed_queries", []) or []

    status = "success"
    error_msg = None
    summary = ""
    rewritten = original_query  # 默认 fallback 值为原问题

    try:
        # 获取封装好的 LLM 实例，并绑定 Pydantic 结构化输出
        llm = get_llm()
        structured_llm = llm.with_structured_output(RewriteOutput)

        # 路由分支：判断是首次执行还是反思重试
        if not failed_queries:
            # 分支 A: 首次改写
            prompt = ChatPromptTemplate.from_messages([
                ("system", FIRST_TRY_SYSTEM_PROMPT),
                ("human", "{original_query}")
            ])
            chain = prompt | structured_llm
            result = chain.invoke({"original_query": original_query})
            summary_prefix = "First try"
        else:
            # 分支 B: 触发重试与反思
            failed_queries_str = "\n".join([f"- {fq}" for fq in failed_queries])
            prompt = ChatPromptTemplate.from_messages([
                ("system", RETRY_SYSTEM_PROMPT),
                ("human", "Original Query: {original_query}")
            ])
            chain = prompt | structured_llm
            result = chain.invoke({
                "original_query": original_query,
                "failed_queries_str": failed_queries_str
            })
            summary_prefix = f"Retry #{len(failed_queries)}"

        # 校验大模型是否成功返回了 keywords 列表
        if result and result.search_phrase:
            # 直接提取生成的学术句子，不再需要 join 拼接
            rewritten = result.search_phrase
            summary = f"[{summary_prefix}] Rewrote query to semantic phrase."
        else:
            summary = f"[{summary_prefix}] LLM returned empty phrase, fallback to original query."

    except Exception as e:
        # 绝对不抛出异常导致图崩溃！记录错误并触发原样放行的 Fallback
        status = "error"
        error_msg = f"Query rewriting failed: {str(e)}"
        summary = "Error during rewriting, fell back to original query."

    # 计算节点总耗时 (毫秒)
    latency = round((time.time() - start) * 1000, 2)

    # 严格按照契约组装返回字典
    response = {
        "rewritten_query": rewritten,
        "failed_queries": [rewritten],  # LangGraph 会自动 Append
        "execution_trace": [build_trace_entry(
            node="rewriting",
            status=status,
            latency_ms=latency,
            summary=summary,
            key_output={"rewritten_query": rewritten}
        )]
    }

    # 仅在发生报错时写入 error_message (Liu 框架的约定)
    if error_msg:
        response["error_message"] = error_msg

    return response