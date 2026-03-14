# Owner: Liu
# Responsibility: Conditional Router — Node 5
# Input:  hallucination_score, retry_count, max_retries
# Output: "retry" | "output" | "stop"

import time
from app.graph.state import GraphState
from app.utils.tracer import build_trace_entry
from app.utils.constants import HALLUCINATION_THRESHOLD, MAX_RETRIES

# Evaluate hallucination score and determine whether to output, retry, or stop
def route_decision(state: GraphState) -> dict:
    start = time.time()

    score = state["hallucination_score"]
    retry_count = state["retry_count"]
    max_retries = state.get("max_retries", MAX_RETRIES)

    # Decision logic
    if score >= HALLUCINATION_THRESHOLD:
        decision = "output"
        summary = f"Socore {score} >= threashold {HALLUCINATION_THRESHOLD} - outputting answer"
    elif retry_count < max_retries:
        decision = "retry"
        retry_count += 1
        summary = f"Score {score} < threashold {HALLUCINATION_THRESHOLD} - retrying (attempt {retry_count} / {max_retries})"
    else:
        decision = "stop"
        summary = f"Score {score} < threashold {HALLUCINATION_THRESHOLD} - max retries reached, stopping"

    latency = round((time.time() - start) * 1000, 2)

    return {
        "final_decision": decision,
        "retry_count": retry_count,
        "execution_trace": [
            build_trace_entry(
                node="router",
                status="success",
                latency_ms=latency,
                summary=summary,
                key_output={
                    "decision": decision,
                    "score": score,
                    "threshold": HALLUCINATION_THRESHOLD,
                    "retry_count": retry_count,
                },
            )
        ],
    }


# Read final_decision from state and return the name of the next node for LangGraph routing
def get_next_node(state: GraphState) -> str:
    decision = state["final_decision"]

    if decision == "retry":
        return "rewriting"
    else:
        return "output"
