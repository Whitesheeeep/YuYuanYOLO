"""
真实场景测试：YuYuanGuidanceAgent.
====================================
用途：用真实图片和真实问题验证 Agent 端到端行为。
      输出结构化日志供人工审查。

配置方法：修改下方 SCENARIOS 列表，填入
  - image_path : 图片绝对路径（str），或 None（纯文字查询）
  - query      : 用户提问（str）

运行方式：
  pytest LangChain/test_real_scenarios.py -v -s
  pytest LangChain/test_real_scenarios.py -v -s -k "multiturn"
"""

import os

import pytest
from dotenv import load_dotenv

from LangChain.guidance_agent import YuYuanGuidanceAgent

load_dotenv()

# ============================================================================
# ★ 在这里配置你的测试场景
# ============================================================================

# 单轮场景：每条是一次独立调用
SCENARIOS = [
    {
        "id": "scene_01",
        "desc": "识别景点 + 历史介绍",
        "image_path": r"D:\master\Yuyuan2\YuyuanYOLO\LangChain\test_imgs\test_1.jpg",
        "query": "这是豫园的哪个景点？请简单介绍它的历史。",
    },
    {
        "id": "scene_02",
        "desc": "建筑细节描述",
        "image_path": r"D:\master\Yuyuan2\YuyuanYOLO\LangChain\test_imgs\test_3.jpg",
        "query": "图中建筑有什么特色？用听障游客能理解的方式描述。",
    },
    {
        "id": "scene_03",
        "desc": "纯文字问答（无图）",
        "image_path": None,
        "query": "九曲桥弯曲的原因是什么？",
    },
    {
        "id": "scene_04",
        "desc": "问路 / 导航引导",
        "image_path": r"D:\master\Yuyuan2\YuyuanYOLO\LangChain\test_imgs\test_3.jpg",
        "query": "我站在这里，往哪个方向走能到三穗堂？",
    },
]

# 多轮场景：每条是连续多轮对话，共用同一 session_id
MULTITURN_SCENARIOS = [
    {
        "id": "multiturn_01",
        "desc": "先看图识景点，再追问历史",
        "turns": [
            {
                "image_path": r"D:\master\Yuyuan2\YuyuanYOLO\LangChain\test_imgs\test_1.jpg",
                "query": "这是哪里？",
            },
            {
                "image_path": None,
                "query": "刚才你说的地方，有什么历史典故？",
            },
            {
                "image_path": None,
                "query": "这里适合听障游客拍照的最佳角度是哪里？",
            },
        ],
    },
    {
        "id": "multiturn_02",
        "desc": "连续两张不同景点图，测试记忆切换",
        "turns": [
            {
                "image_path": r"D:\master\Yuyuan2\YuyuanYOLO\LangChain\test_imgs\test_1.jpg",
                "query": "请描述这个景点。",
            },
            {
                "image_path": r"D:\master\Yuyuan2\YuyuanYOLO\LangChain\test_imgs\test_3.jpg",
                "query": "这张图和上一张有什么不同？",
            },
        ],
    },
]

# ============================================================================
# 辅助输出
# ============================================================================


def _section(title: str):
    print(f"\n{'═' * 64}")
    print(f"  {title}")
    print(f"{'═' * 64}")


def _show(label: str, value, truncate: int = 200):
    val = str(value)
    if len(val) > truncate:
        val = val[:truncate] + f"...（共 {len(str(value))} 字符）"
    print(f"  {label:<30} {val}")


def _check_image(path):
    """检查图片路径是否可访问，返回 (ok, reason)。."""
    if path is None:
        return True, "纯文字查询，无需图片"
    if not os.path.exists(path):
        return False, f"文件不存在: {path}"
    size = os.path.getsize(path)
    if size == 0:
        return False, "文件为空"
    return True, f"文件大小 {size / 1024:.1f} KB"


# ============================================================================
# 单轮场景测试
# ============================================================================


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
@pytest.mark.asyncio
async def test_real_scenario(scenario):
    sid = scenario["id"]
    desc = scenario["desc"]
    image_path = scenario["image_path"]
    query = scenario["query"]

    _section(f"[{sid}] {desc}")
    _show("query", query)
    _show("image_path", image_path or "（无图，纯文字）")

    # 图片可用性检查
    ok, reason = _check_image(image_path)
    _show("图片状态", reason)
    if not ok:
        pytest.skip(f"图片不可用，跳过: {reason}")

    agent = YuYuanGuidanceAgent()

    if image_path:
        answer = await agent.generate_guidance_with_image(
            user_query=query,
            image_path=image_path,
            session_id=sid,
        )
    else:
        answer = await agent.generate_guidance(
            user_query=query,
            session_id=sid,
        )

    _show("Agent 回答", answer)
    _show("回答长度 (字符)", len(answer))
    _show("是否兜底文案", answer == "导览助手暂时无法连接，请稍后再试。")

    assert isinstance(answer, str), "返回值应为字符串"
    assert answer.strip() != "", "返回值不应为空"
    assert answer != "导览助手暂时无法连接，请稍后再试。", "不应触发兜底文案"
    print(f"\n  ✓ [{sid}] 通过")


# ============================================================================
# 多轮场景测试
# ============================================================================


@pytest.mark.parametrize("scenario", MULTITURN_SCENARIOS, ids=[s["id"] for s in MULTITURN_SCENARIOS])
@pytest.mark.asyncio
async def test_real_multiturn_scenario(scenario):
    sid = scenario["id"]
    desc = scenario["desc"]
    turns = scenario["turns"]

    _section(f"[{sid}] {desc}  （共 {len(turns)} 轮）")

    # 预检所有图片
    for i, turn in enumerate(turns):
        ok, reason = _check_image(turn["image_path"])
        _show(f"  轮次 {i + 1} 图片状态", reason)
        if not ok:
            pytest.skip(f"轮次 {i + 1} 图片不可用，跳过: {reason}")

    agent = YuYuanGuidanceAgent()
    answers = []

    for i, turn in enumerate(turns):
        image_path = turn["image_path"]
        query = turn["query"]

        print(f"\n  {'─' * 56}")
        print(f"  轮次 {i + 1} / {len(turns)}")
        print(f"  {'─' * 56}")
        _show("  query", query)
        _show("  image_path", image_path or "（无图）")

        if image_path:
            answer = await agent.generate_guidance_with_image(
                user_query=query,
                image_path=image_path,
                session_id=sid,
            )
        else:
            answer = await agent.generate_guidance(
                user_query=query,
                session_id=sid,
            )

        _show("  Agent 回答", answer)
        _show("  回答长度 (字符)", len(answer))
        answers.append(answer)

        assert isinstance(answer, str) and answer.strip(), f"轮次 {i + 1} 返回值为空或非字符串"
        assert answer != "导览助手暂时无法连接，请稍后再试。", f"轮次 {i + 1} 触发兜底文案"

    print(f"\n  ✓ [{sid}] {len(turns)} 轮全部通过")
    print("\n  【对话摘要】")
    for i, (turn, and) in enumerate(zip(turns, answers)):
        print(f"    Q{i + 1}: {turn['query']}")
        preview = and[:80].replace("\n", " ")
        print(f"    A{i + 1}: {preview}{'...' if len(and) > 80 else ''}")
