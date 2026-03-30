# ============================================================
# RAG 引擎独立测试脚本
# ============================================================
# 用法: python -m LangChain.test_rag

import os
import sys

# 确保可以导入 LangChain 包
sys.path.insert(0, "E:\\Master\\ultralytics-main")


from LangChain.rag_engine import YuYuanRAG
from LangChain import config

def test_rag_functionality():
    print("="*50)
    print("开始测试 YuYuanRAG 核心功能")
    print("="*50)

    # 1. 初始化引擎
    print("\n[步骤 1] 初始化 RAG 引擎...")
    rag = YuYuanRAG()

    # 2. 检查索引状态
    if not rag.is_ready():
        print("[步骤 2] 索引未就绪，开始构建...")
        kb_path = config.KNOWLEDGE_BASE_PATH
        if not os.path.exists(kb_path):
            print(f"[错误] 知识库文件不存在: {kb_path}")
            return
        rag.build_from_text(kb_path)
    else:
        print("[步骤 2] 索引已成功加载。")

    # 3. 测试基本检索 (不带重排序)
    print("\n[步骤 3] 测试基本检索 (rerank=False)...")
    query = "玉玲珑是什么？"
    results = rag.retrieve(query, k=2, rerank=False)
    print(f"查询: {query}")
    for i, res in enumerate(results):
        print(f"  结果 {i+1}: {res[:100]}...")

    # 4. 测试带分数的检索
    print("\n[步骤 4] 测试带分数的检索...")
    query = "龙墙在哪里？"
    results_with_scores = rag.retrieve_with_score(query, k=2, threshold=0.1)
    print(f"查询: {query}")
    for i, (res, score) in enumerate(results_with_scores):
        print(f"  结果 {i+1} [分数: {score:.4f}]: {res[:100]}...")

    # 5. 测试语义切分器验证 (仅限逻辑验证)
    print("\n[步骤 5] 验证 RecursiveCharacterTextSplitter 配置...")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    test_text = "这是第一句。这是第二句，带逗号。这是第三句！\n这是新的一行。"
    chunks = splitter.split_text(test_text)
    print(f"测试文本切分结果: {chunks}")

    print("\n" + "="*50)
    print("RAG 功能测试完成")
    print("="*50)

if __name__ == "__main__":
    try:
        test_rag_functionality()
    except Exception as e:
        print(f"\n[测试失败] 捕获到异常: {e}")
        import traceback
        traceback.print_exc()
