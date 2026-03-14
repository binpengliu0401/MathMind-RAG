# Owner: Liu
# Responsibility: LangGraph assembly — connects all nodes into a complete graph

from langgraph.graph import StateGraph, END
from app.graph.state import GraphState
from app.graph.router import route_decision, get_next_node
from app.nodes.rewriting import rewrite_query
from app.nodes.retrieval import retrieve_docs
from app.nodes.generation import generate_answer
from app.nodes.grading import grade_hallucination
from app.utils.constants import MAX_RETRIES


def build_graph():

    # Initialize graph with state schema
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("rewriting", rewrite_query)
    graph.add_node("retrieval", retrieve_docs)
    graph.add_node("generation", generate_answer)
    graph.add_node("grading", grade_hallucination)
    graph.add_node("router", route_decision)

    # Define edges
    graph.set_entry_point("rewriting")
    graph.add_edge("rewriting", "retrieval")
    graph.add_edge("retrieval", "generation")
    graph.add_edge("generation", "grading")
    graph.add_edge("grading", "router")

    # Conditonal edge from router
    graph.add_conditional_edges(
        "router", get_next_node, {"rewriting": "rewriting", "output": END}
    )

    return graph.compile()


rag_graph = build_graph()
