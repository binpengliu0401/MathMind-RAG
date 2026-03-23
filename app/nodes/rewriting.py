import time
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import GraphState
from app.utils.tracer import build_trace_entry
from app.services.llm_service import get_llm


class RewriteOutput(BaseModel):
    search_phrase: str = Field(
        description="A concise, highly professional academic search phrase optimized for dense vector search. Do NOT output a list of keywords. Extract the true academic intent and correct any colloquial or inaccurate terms."
    )


FIRST_TRY_SYSTEM_PROMPT = """You are an expert AI/Math academic search assistant.
Your task is to optimize the user's query for a dense vector database (BGE/BERT).

CRITICAL CONSTRAINTS:
1. ALIGN & CORRECT TERMINOLOGY: Extract the true academic intent. If the user uses colloquialisms or inaccurate terms (e.g., calling an API interaction "reinforcement learning"), correct it to the standard academic jargon. If they use correct jargon, keep it.
2. EXPAND INTELLIGENTLY: Add 1 or 2 universally accepted academic terms/mechanisms that directly underlie the user's concept.
3. NO HALLUCINATION: DO NOT add specific dataset names (e.g., GSM8K) or model names (e.g., GPT-3) UNLESS the user explicitly mentioned them.
4. CONCISE FORMAT: Output a short, punchy search phrase (e.g., "zero-shot chain of thought reasoning"). NO conversational filler ("What is", "How to").
5. If the user's original query is already a highly professional, well-structured academic phrase (e.g., containing specific metrics, full dataset names), DO NOT compress it. Act as a pass-through and retain its full descriptive richness.

Output ONLY the optimized search phrase."""

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


def rewrite_query(state: GraphState) -> dict:
    """
    Node 1: Query Rewriting
    Rewrites the user query to improve retrieval quality.
    On retry, uses failed_queries to avoid repeating previous attempts.
    """
    start = time.time()

    original_query = state.get("query", "")
    failed_queries = state.get("failed_queries", []) or []

    status = "success"
    error_msg = None
    summary = ""
    rewritten = original_query

    try:
        llm = get_llm("rewriting")
        structured_llm = llm.with_structured_output(RewriteOutput)

        if not failed_queries:
            prompt = ChatPromptTemplate.from_messages(
                [("system", FIRST_TRY_SYSTEM_PROMPT), ("human", "{original_query}")]
            )
            chain = prompt | structured_llm
            result = chain.invoke({"original_query": original_query})
            summary_prefix = "First try"
        else:
            failed_queries_str = "\n".join([f"- {fq}" for fq in failed_queries])
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", RETRY_SYSTEM_PROMPT),
                    ("human", "Original Query: {original_query}"),
                ]
            )
            chain = prompt | structured_llm
            result = chain.invoke(
                {
                    "original_query": original_query,
                    "failed_queries_str": failed_queries_str,
                }
            )
            summary_prefix = f"Retry #{len(failed_queries)}"

        if result and result.search_phrase:  # type: ignore
            rewritten = result.search_phrase  # type: ignore
            summary = f"[{summary_prefix}] Rewrote query to semantic phrase."
        else:
            summary = f"[{summary_prefix}] LLM returned empty phrase, fallback to original query."

    except Exception as e:
        status = "error"
        error_msg = f"Query rewriting failed: {str(e)}"
        summary = "Error during rewriting, fell back to original query."

    latency = round((time.time() - start) * 1000, 2)

    response = {
        "rewritten_query": rewritten,
        "failed_queries": [rewritten],  # LangGraph 会自动 Append
        "execution_trace": [
            build_trace_entry(
                node="rewriting",
                status=status,
                latency_ms=latency,
                summary=summary,
                key_output={"rewritten_query": rewritten},
            )
        ],
    }

    if error_msg:
        response["error_message"] = error_msg

    return response
