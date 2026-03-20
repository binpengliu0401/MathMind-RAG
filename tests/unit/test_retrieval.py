import os
import tempfile
import numpy as np
import pytest
from langchain_core.documents import Document


def test_extract_documents():
    from app.dataset_processing.dataset_loader import (
        load_paper_parquet,
        extract_documents,
    )

    parquet_path = os.environ.get("PAPER_PARQUET", "data/train-00000-of-00001.parquet")

    if not os.path.exists(parquet_path):
        pytest.skip(f"Data file does not exist: {parquet_path}")

    df = load_paper_parquet(parquet_path)
    docs = extract_documents(df)

    assert len(docs) > 0, "At least one Document should be extracted."

    for doc in docs[:10]:
        assert isinstance(doc, Document), f"Expected Document type, got: {type(doc)}"
        assert len(doc.page_content.strip()) > 0, "page_content should not be empty"
        assert "source" in doc.metadata, "metadata must contain 'source' field"
        assert isinstance(doc.metadata["source"], str), "source should be str"


def test_embedder():
    from app.dataset_processing.embedder import Embedder

    embedder = Embedder(model_name="all-MiniLM-L6-v2")

    texts = ["Hello world", "Attention is all you need"]
    doc_embs = embedder.embed_documents(texts)
    assert doc_embs.shape == (2, embedder.embedding_dim)

    norms = np.linalg.norm(doc_embs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    query_emb = embedder.embed_query("What is attention?")
    assert query_emb.shape == (1, embedder.embedding_dim)


def test_vector_store_flat():
    from app.dataset_processing.vector_store import VectorStore

    dim = 64
    n = 100
    store = VectorStore(embedding_dim=dim, index_type="flat")

    embeddings = np.random.randn(n, dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    docs = [
        Document(page_content=f"Document {i}", metadata={"source": f"doc_{i}"})
        for i in range(n)
    ]

    store.build_index(embeddings, docs)
    assert store.index.ntotal == n

    query = np.random.randn(1, dim).astype(np.float32)
    query = query / np.linalg.norm(query)
    results = store.search(query, top_k=5)

    assert len(results) == 5
    for doc in results:
        assert isinstance(doc, Document)
        assert "score" in doc.metadata

    with tempfile.TemporaryDirectory() as tmpdir:
        idx_path = os.path.join(tmpdir, "test.index")
        doc_path = os.path.join(tmpdir, "test.pkl")

        store.save(idx_path, doc_path)
        assert os.path.exists(idx_path)
        assert os.path.exists(doc_path)

        store2 = VectorStore(embedding_dim=dim, index_type="flat")
        store2.load(idx_path, doc_path)
        assert store2.index.ntotal == n
        assert len(store2.documents) == n


def test_retrieval_node_contract():
    from app.nodes.retrieval import create_retrieval_node

    class MockRetriever:
        def retrieve(self, query, top_k=5):
            return [
                Document(
                    page_content=f"Mock result for: {query}",
                    metadata={"source": "mock_paper", "authors": [], "score": 0.95},
                )
            ]

    node = create_retrieval_node(MockRetriever())

    state = {
        "query": "What is chain of thought?",
        "rewritten_query": "chain of thought prompting zero-shot reasoning LLM",
        "failed_queries": [],
        "retrieved_docs": [],
        "answer": "",
        "hallucination_score": 0.0,
        "retry_count": 0,
        "max_retries": 3,
        "final_decision": "",
        "error_message": None,
        "execution_trace": [],
    }

    result = node(state)

    assert "retrieved_docs" in result, "Result must contain retrieved_docs"
    assert "execution_trace" in result, "Result must contain execution_trace"

    assert isinstance(result["retrieved_docs"], list)
    for doc in result["retrieved_docs"]:
        assert isinstance(
            doc, Document
        ), "Each item in retrieved_docs must be a Document"

    assert isinstance(result["execution_trace"], list)
    assert len(result["execution_trace"]) == 1
    trace = result["execution_trace"][0]
    assert trace["node"] == "retrieval"
    assert "latency" in trace
    assert "docs_retrieved" in trace
