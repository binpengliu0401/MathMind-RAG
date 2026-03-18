# python -m tests.eval_rewrite

import os
import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from app.nodes.rewriting import rewrite_query

load_dotenv()

print("⏳ Loading local embedding model (BAAI/bge-base-en-v1.5)...")
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")


def cosine_similarity(v1: list, v2: list) -> float:
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def run_evaluation():
    print("\n🚀 Starting Quantitative Evaluation for Query Rewriting Node...\n")

    # 【优化点1】：测试用例必须包含隐式实体，不能全是代词，否则无状态 RAG 必死
    test_cases = [
        {
            "id": 1,
            # 模糊问题 (对应 Abstract 1: Zero-shot CoT)
            "vague_query": "how to make the model reason step by step without giving it any examples?",
            "target_doc": "While these successes are often attributed to LLMs' ability for few-shot learning, we show that LLMs are decent zero-shot reasoners by simply adding ``Let's think step by step'' before each answer. Experimental results demonstrate that our Zero-shot-CoT... significantly outperforms zero-shot LLM performances on diverse benchmark reasoning tasks including arithmetics (MultiArith, GSM8K, AQUA-RAT, SVAMP)..."
        },
        {
            "id": 2,
            # 一般问题 (对应 Abstract 3: ReAct)
            "vague_query": "combining reasoning and acting in language models to solve tasks",
            "target_doc": "In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two... We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness... Concretely, on question answering (HotpotQA) and fact verification (Fever)..."
        },
        {
            "id": 3,
            # 好问题 (对应 Abstract 2: MMLU) - 测试系统是否能保持高分不帮倒忙
            "vague_query": "what is the MMLU benchmark with 57 tasks for evaluating massive multitask language understanding?",
            "target_doc": "We propose a new test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more... By comprehensively evaluating the breadth and depth of a model's academic and professional understanding, our test can be used to analyze models across many tasks and to identify important shortcomings."
        },
        {
            "id": 4,
            # 反思测试 (对应 Abstract 1: Zero-shot CoT)
            "vague_query": "zero-shot reasoning in LLMs",
            # 故意屏蔽掉最明显的宏观词，逼迫大模型去想 GSM8K, MultiArith 等具体的数据集或 "Let's think step by step" 这种具体 prompt
            "failed_queries": ["zero-shot reasoning in LLMs", "zero-shot-CoT algorithm", "step-by-step reasoning"],
            "target_doc": "While these successes are often attributed to LLMs' ability for few-shot learning, we show that LLMs are decent zero-shot reasoners by simply adding ``Let's think step by step'' before each answer. Experimental results demonstrate that our Zero-shot-CoT... significantly outperforms zero-shot LLM performances on diverse benchmark reasoning tasks including arithmetics (MultiArith, GSM8K, AQUA-RAT, SVAMP)..."
        }
    ]

    total_improvement = 0.0

    for case in test_cases:
        print(f"--- Test Case #{case['id']} ---")

        # 传入 query，如果 case 里有 failed_queries 也一并传入
        state = {"query": case["vague_query"]}
        if "failed_queries" in case:
            state["failed_queries"] = case["failed_queries"]

        result = rewrite_query(state)
        rewritten_query = result.get("rewritten_query", "")
        print(f"✨ Rewritten Query: {rewritten_query}")

        # 【优化点2】：严格遵守队友 Li 的架构契约，为 Query 加上 BGE 的专用前缀
        bge_prefix = "Represent this sentence: "
        doc_vec = embedder.embed_query(case["target_doc"])  # Document 不加
        vague_vec = embedder.embed_query(bge_prefix + case["vague_query"])  # 加上前缀
        rewritten_vec = embedder.embed_query(bge_prefix + rewritten_query)  # 加上前缀

        sim_original = cosine_similarity(vague_vec, doc_vec)
        sim_rewritten = cosine_similarity(rewritten_vec, doc_vec)

        improvement = ((sim_rewritten - sim_original) / sim_original) * 100
        total_improvement += improvement

        print(f"📊 Original Similarity:  {sim_original:.4f}")
        print(f"📈 Rewritten Similarity: {sim_rewritten:.4f}")
        print(f"🔥 Improvement:          +{improvement:.2f}%\n")

    avg_improvement = total_improvement / len(test_cases)
    print("==================================================")
    print(f"🏆 EVALUATION COMPLETE: Average Similarity Lift = +{avg_improvement:.2f}%")
    print("==================================================")


if __name__ == "__main__":
    run_evaluation()
