# Owner: Liu
# Responsibility: Execution trace helper — used by all nodes
from typing import List


def build_trace_entry(
    node: str, status: str, latency_ms: float, summary: str, key_output: dict = {}
) -> dict:
    """
    Build a single execution trace entry.

    Args:
        node:        node name, e.g. "generation"
        status:      "success" or "error"
        latency_ms:  time taken in milliseconds
        summary:     one-line human-readable description
        key_output:  key results to surface for frontend / demo display

    Returns:
        A dict to be appended to execution_trace in GraphState.
    """
    return {
        "node": node,
        "status": status,
        "latency_ms": latency_ms,
        "summary": summary,
        "key_output": key_output,
    }


# Pretty-print the full execution trace to terminal for demo purposes
def print_trace(state: dict):
    trace: List[dict] = state.get("execution_trace", [])

    print("\n" + "═" * 55)
    print("  RAG SYSTEM — EXECUTION TRACE")
    print("═" * 55)

    for i, entry in enumerate(trace):
        status_icon = "✓" if entry["status"] == "success" else "✗"
        print(
            f"  [{i+1}] {entry['node']:<25} {entry['latency_ms']:>7.1f}ms  {status_icon}"
        )
        print(f"       {entry['summary']}")

    print("─" * 55)
    print(f"  Query:    {state.get('query', '')[:60]}")
    print(f"  Rewrite:  {state.get('rewritten_query', '')[:60]}")
    print(
        f"  Score:    {state.get('hallucination_score', 0.0):.2f}   "
        f"Retries: {state.get('retry_count', 0)}   "
        f"Decision: {state.get('final_decision', '-')}"
    )
    print("─" * 55)
    print(f"  Answer:")
    answer = state.get("answer", "")
    print(f"  {answer[:200]}{'...' if len(answer) > 200 else ''}")
    print("═" * 55 + "\n")
