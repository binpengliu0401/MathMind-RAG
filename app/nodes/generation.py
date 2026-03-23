# Owner: Liu
# Responsibility: Generation Node — Node 3
# Input:  rewritten_query, retrieved_docs
# Output: answer, execution_trace

import time
from langchain_core.prompts import ChatPromptTemplate
from app.graph.state import GraphState
from app.services.llm_service import get_llm
from app.utils.tracer import build_trace_entry

# prompt
GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an academic assistant. Answer the question based ONLY 
on the provided context. If the context does not contain enough information 
to answer the question, say "I cannot find sufficient information in the 
provided documents."

Context:
{context}""",
        ),
        ("human", "{question}"),
    ]
)


# Helper
def format_docs(docs) -> str:
    return "\n\n".join(f"[Doc {i+1}] {doc.page_content}" for i, doc in enumerate(docs))


# Node
def generate_answer(state: GraphState) -> dict:
    start = time.time()
    try:
        rewritten_query = state["rewritten_query"]
        retrieved_docs = state["retrieved_docs"]

        context = format_docs(retrieved_docs)
        llm = get_llm("generation")

        # LCEL
        chain = GENERATION_PROMPT | llm

        response = chain.invoke({"context": context, "question": rewritten_query})

        answer = response.content
        latency = round((time.time() - start) * 1000, 2)

        return {
            "answer": answer,
            "execution_trace": [
                build_trace_entry(
                    node="generation",
                    status="success",
                    latency_ms=latency,
                    summary=f"Generated answer from {len(retrieved_docs)} retrieved docs",
                    key_output={
                        "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer,  # type: ignore
                        "doc_count": len(retrieved_docs),
                    },
                )
            ],
        }
    except Exception as e:
        latency = round((time.time() - start) * 1000, 2)
        return {
            "answer": "",
            "error_message": f"Generation failed: {str(e)}",
            "execution_trace": [
                build_trace_entry(
                    node="generation",
                    status="error",
                    latency_ms=latency,
                    summary=f"Generation failed: {str(e)}",
                    key_output={},
                )
            ],
        }
