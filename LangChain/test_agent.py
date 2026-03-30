# ============================================================
# 独立测试脚本 - 验证 LangChain ReAct Agent 功能
# ============================================================
# 用法: python -m LangChain.test_agent

import sys
import os

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LangChain.rag_engine import YuYuanRAG
from LangChain.guidance_agent import YuYuanGuidanceAgent
from LangChain import config


def test_rag_build():
    """测试 1: 构建 RAG 索引"""
    print("\n" + "=" * 60)
    print("测试 1: RAG 索引构建（使用 RecursiveCharacterTextSplitter）")
    print("=" * 60)

    rag = YuYuanRAG()

    # 检查是否已有索引
    if rag.is_ready():
        print("[OK] 索引已存在，直接加载")
    else:
        print("[INFO] 索引不存在，开始构建...")
        rag.build_from_text(config.KNOWLEDGE_BASE_PATH)
        print("[OK] 索引构建完成")

    return rag


def test_rag_retrieve(rag: YuYuanRAG):
    """测试 2: RAG 检索功能"""
    print("\n" + "=" * 60)
    print("测试 2: RAG 检索")
    print("=" * 60)

    test_queries = ["玉玲珑", "左铁狮", "龙墙", "点春堂"]

    for query in test_queries:
        print(f"\n检索: '{query}'")
        results = rag.retrieve(query, k=2)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {r[:60]}...")
        if not results:
            print("  [无结果]")


def test_agent_guidance(agent: YuYuanGuidanceAgent):
    """测试 3: ReAct Agent 生成讲解"""
    print("\n" + "=" * 60)
    print("测试 3: ReAct Agent 生成讲解（模拟 YOLO 检测）")
    print("=" * 60)

    # 模拟 YOLO 检测结果
    test_cases = [
        {
            "name": "检测到玉玲珑",
            "detections": [
                {"class": "stone1", "confidence": 0.92, "bbox": [100, 100, 500, 400]}
            ]
        },
        {
            "name": "检测到左铁狮",
            "detections": [
                {"class": "LeftIronLion", "confidence": 0.88, "bbox": [50, 150, 200, 400]}
            ]
        },
        {
            "name": "低置信度检测（应跳过）",
            "detections": [
                {"class": "stone2", "confidence": 0.60, "bbox": [100, 100, 300, 300]}
            ]
        },
        {
            "name": "无检测（应返回 None）",
            "detections": []
        }
    ]

    for case in test_cases:
        print(f"\n[测试] {case['name']}")
        print(f"  输入: {case['detections']}")

        result = agent.generate_guidance(case['detections'])

        if result is None:
            print("  输出: None（符合预期）")
        else:
            print(f"  输出类型: {result.get('type')}")
            print(f"  显示文本: {result.get('display_text')}")
            print(f"  动作指令: {result.get('action_cmd')}")
            print(f"  震动模式: {result.get('vibrate')}")


def test_agent_auto_hint(agent: YuYuanGuidanceAgent):
    """测试 4: 自动视觉注释"""
    print("\n" + "=" * 60)
    print("测试 4: 自动视觉注释")
    print("=" * 60)

    detections = [
        {"class": "LeftIronLion", "confidence": 0.90, "bbox": [20, 100, 100, 350]}
    ]

    print(f"输入: {detections}")
    hint = agent.generate_auto_hint(detections)

    if hint:
        print(f"输出: {hint}")
    else:
        print("输出: None")


def test_rag_splitter():
    """测试 5: 验证 RecursiveCharacterTextSplitter 切分效果"""
    print("\n" + "=" * 60)
    print("测试 5: 验证 RecursiveCharacterTextSplitter 切分效果")
    print("=" * 60)

    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        sample_text = """玉玲珑：豫园中心假山，太湖石堆砌，高约5米，玲珑剔透。左铁狮：大门左侧，蹲狮形象，铸铁材质，清代遗物。右铁狮：大门右侧，蹲狮形象，与左狮对称。龙墙：豫园外墙，巨龙盘旋形态，寓意吉祥。"""

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=10,
            separators=["\n\n", "\n", "。", "，", " ", ""],
            length_function=len
        )

        docs = splitter.split_text(sample_text)
        print(f"原始文本长度: {len(sample_text)} 字符")
        print(f"切分后文档数: {len(docs)}")
        print("\n切分结果:")
        for i, doc in enumerate(docs, 1):
            print(f"  [块 {i}] {doc}")

        print("\n[OK] RecursiveCharacterTextSplitter 工作正常")
        return True
    except Exception as e:
        print(f"[错误] {e}")
        return False


def main():
    print("=" * 60)
    print("豫园 XR LangChain ReAct Agent 独立测试")
    print("=" * 60)

    # 检查 API Key
    if config.DOUBao_API_KEY == "your-doubao-api-key":
        print("\n[警告] 请先在 LangChain/config.py 中设置 DouBao API Key")
        print("跳过 LLM 相关测试，仅测试 RAG 功能...\n")
        skip_llm = True
    else:
        skip_llm = False

    # 测试 5: 切分器验证（不依赖索引）
    test_rag_splitter()

    # 测试 1: RAG 索引
    rag = test_rag_build()

    # 测试 2: RAG 检索
    test_rag_retrieve(rag)

    if skip_llm:
        print("\n[跳过] LLM 测试（API Key 未配置）")
        print("\n请设置 config.py 中的 DOUBao_API_KEY 后重新运行完整测试")
    else:
        # 测试 3: ReAct Agent 讲解生成
        print("\n[INFO] 正在初始化 ReAct Agent（首次运行需下载模型，约几分钟）...")
        agent = YuYuanGuidanceAgent(
            rag_engine=rag,
            verbose=True  # 开启 ReAct 思考过程打印
        )
        test_agent_guidance(agent)

        # 测试 4: 自动注释
        test_agent_auto_hint(agent)

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n关键更新：")
    print("1. RAG 使用 RecursiveCharacterTextSplitter 进行语义切分")
    print("2. Agent 使用 create_react_agent 创建标准 ReAct Agent")
    print("3. Agent 支持工具调用（YuYuanKnowledge）进行知识检索")


if __name__ == "__main__":
    main()
