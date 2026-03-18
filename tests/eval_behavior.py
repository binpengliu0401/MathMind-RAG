# 运行命令: python -m tests.eval_behavior

from app.nodes.rewriting import rewrite_query


def run_behavioral_eval():
    print("\n🚀 Starting Behavioral Evaluation for Query Rewriting Node...\n")

    test_cases = [
        {
            "id": 1,
            "scenario": "小白/口语化查询 (测试意图抽取)",
            "query": "how to make the model reason step by step without giving it any examples?",
        },
        {
            "id": 2,
            "scenario": "学术黑话保全 (测试防退化)",
            "query": "combining reasoning and acting in LLMs",
        },
        {
            "id": 3,
            "scenario": "极度宽泛查询 (测试具象化与防幻觉)",
            "query": "improving accuracy on tests",
        },
        {
            "id": 4,
            "scenario": "真实的重试场景 (测试强制发散与避坑)",
            "query": "math problem solving algorithms",
            "failed_queries": ["math problem solving algorithms", "mathematical logic reasoning neural networks"]
        }
    ]

    for case in test_cases:
        print(f"--- Test Case #{case['id']}: {case['scenario']} ---")
        print(f"👤 Original Query: {case['query']}")
        if "failed_queries" in case:
            print(f"🚫 Failed History:  {case['failed_queries']}")

        # 组装 State
        state = {"query": case["query"]}
        if "failed_queries" in case:
            state["failed_queries"] = case["failed_queries"]

        # 调用你的工位 1 核心函数
        result = rewrite_query(state)
        rewritten_query = result.get("rewritten_query", "ERROR/FALLBACK")

        print(f"✨ Rewritten Phrase: {rewritten_query}\n")


if __name__ == "__main__":
    run_behavioral_eval()