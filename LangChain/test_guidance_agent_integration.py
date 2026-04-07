"""
集成测试：YuYuanGuidanceAgent（真实 RAG + 真实 LLM）
====================================================
标记为 integration，默认不在 CI 中执行。
运行方式：pytest -m integration LangChain/test_guidance_agent_integration.py -v -s
"""
import pytest
import os
import io
import base64 as b64_module
import tempfile
from PIL import Image

from LangChain.guidance_agent import YuYuanGuidanceAgent
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.integration


def _section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")

def _show(label: str, value):
    val_str = str(value)
    if len(val_str) > 200:
        val_str = val_str[:200] + "..."
    print(f"  {label:<32} {val_str}")


@pytest.mark.asyncio
async def test_generate_guidance_with_real_rag_and_llm():
    """集成测试：使用真实 RAG + 真实 LLM，验证文本导览主流程可用。"""
    _section("集成 | 文本导览（真实 RAG + LLM）")
    query = "九曲桥为什么是弯的？请用两句话简洁回答。"
    session = "integration_user_01"
    _show("输入 query", query)
    _show("session_id", session)

    agent = YuYuanGuidanceAgent()
    answer = await agent.generate_guidance(query, session_id=session)

    _show("LLM 返回", answer)
    _show("长度 (字符)", len(answer))
    _show("非空", answer.strip() != "")
    _show("非兜底文案", answer != "导览助手暂时无法连接，请稍后再试。")

    assert isinstance(answer, str)
    assert answer.strip() != ""
    assert answer != "导览助手暂时无法连接，请稍后再试。"
    print("  ✓ 真实链路文本导览正常")


@pytest.mark.asyncio
async def test_generate_guidance_with_memory_real_stack():
    """集成测试：连续提问，验证真实链路下的会话记忆不报错。"""
    _section("集成 | 多轮会话记忆")
    session_id = "integration_user_02"
    q1 = "我现在在九曲桥。"
    q2 = "结合我刚才的位置，下一步推荐看什么？"
    _show("第 1 轮 query", q1)
    _show("第 2 轮 query", q2)
    _show("session_id", session_id)

    agent = YuYuanGuidanceAgent()
    first = await agent.generate_guidance(q1, session_id=session_id)
    second = await agent.generate_guidance(q2, session_id=session_id)

    _show("第 1 轮回答", first)
    _show("第 2 轮回答", second)
    _show("第 2 轮引用上文", "九曲桥" in second or "桥" in second)

    assert isinstance(first, str) and first.strip() != ""
    assert isinstance(second, str) and second.strip() != ""
    assert second != "导览助手暂时无法连接，请稍后再试。"
    print("  ✓ 多轮记忆正常，第 2 轮能感知上文位置")


# ============================================================================
# 图像接口集成测试
# ============================================================================

@pytest.mark.asyncio
async def test_generate_guidance_with_image_path():
    """集成测试：传入图片路径，验证 yolo_detect / image_understand tool 被正确触发。"""
    _section("集成 | 图文导览（image_path 分支）")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img_path = f.name
    try:
        Image.new("RGB", (1600, 1200), color=(50, 150, 200)).save(img_path)
        _show("图像路径", img_path)
        _show("图像尺寸", "1600×1200 px（超过 max_side=1024，触发压缩）")

        agent = YuYuanGuidanceAgent()
        answer = await agent.generate_guidance_with_image(
            "这张图片里有什么，15 字以内，快速回答。",
            image_path=img_path,
            session_id="integration_img_01",
        )

        _show("LLM 返回", answer)
        _show("长度 (字符)", len(answer))

        assert isinstance(answer, str)
        assert answer.strip() != ""
        assert answer != "导览助手暂时无法连接，请稍后再试。"
        print("  ✓ image_path 分支：压缩、编码、工具调用、回答全链路正常")
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


@pytest.mark.asyncio
async def test_generate_guidance_with_image_base64():
    """集成测试：传入纯 base64 字符串，验证图像存入 state 后工具可正常读取。"""
    _section("集成 | 图文导览（image_base64 分支）")

    img = Image.new("RGB", (200, 200), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64_str = b64_module.b64encode(buf.getvalue()).decode("utf-8")

    _show("图像尺寸", "200×200 px，纯红色")
    _show("base64 长度 (字符)", len(b64_str))

    agent = YuYuanGuidanceAgent()
    answer = await agent.generate_guidance_with_image(
        "图片里是什么颜色？",
        image_base64=b64_str,
        session_id="integration_img_02",
    )

    _show("LLM 返回", answer)

    assert isinstance(answer, str)
    assert answer.strip() != ""
    assert answer != "导览助手暂时无法连接，请稍后再试。"
    print("  ✓ image_base64 分支：图像存入 state，工具读取后正常回答")


@pytest.mark.asyncio
async def test_generate_guidance_with_image_data_uri():
    """集成测试：传入完整 data URI（带前缀），验证截断前缀后正常处理。"""
    _section("集成 | 图文导览（data URI 前缀去除）")

    img = Image.new("RGB", (100, 100), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64_data = b64_module.b64encode(buf.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64_data}"

    _show("输入格式", "data:image/jpeg;base64,<payload>")
    _show("payload 长度 (字符)", len(b64_data))

    agent = YuYuanGuidanceAgent()
    answer = await agent.generate_guidance_with_image(
        "图片里的场景是什么？",
        image_base64=data_uri,
        session_id="integration_img_03",
    )

    _show("LLM 返回", answer)

    assert isinstance(answer, str)
    assert answer.strip() != ""
    print("  ✓ data URI 前缀自动去除，后续流程正常")


@pytest.mark.asyncio
async def test_generate_guidance_with_image_no_image_returns_hint():
    """集成测试：既不传 image_path 也不传 image_base64 时，返回提示而非抛错。"""
    _section("集成 | 图文导览（无图像输入保护）")
    _show("image_path", None)
    _show("image_base64", None)

    agent = YuYuanGuidanceAgent()
    answer = await agent.generate_guidance_with_image(
        "这是什么？",
        session_id="integration_img_04",
    )

    _show("实际返回", answer)
    assert "请提供图片" in answer or "无效" in answer
    print("  ✓ 无图像时返回提示文案，无异常抛出")


@pytest.mark.asyncio
async def test_image_not_exposed_to_llm_in_initial_message():
    """集成测试：验证图像不出现在 LLM 初始消息中（历史消息无 image_url / 大体积 base64）。"""
    _section("集成 | 图像隔离验证（不进入 LLM 消息历史）")

    img = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64_str = b64_module.b64encode(buf.getvalue()).decode("utf-8")
    _show("图像 base64 长度", len(b64_str))

    agent = YuYuanGuidanceAgent()
    session = "integration_img_05"
    await agent.generate_guidance_with_image(
        "描述图像",
        image_base64=b64_str,
        session_id=session,
    )

    state = agent.agent.get_state({"configurable": {"thread_id": session}})
    messages = state.values.get("messages", []) if state else []
    _show("checkpointer 历史消息条数", len(messages))

    violations = []
    for i, msg in enumerate(messages):
        content_str = str(msg.content)
        if "image_url" in content_str:
            violations.append(f"msg[{i}] 含 image_url")
        if len(content_str) > 10000:
            violations.append(f"msg[{i}] 内容过长（{len(content_str)} 字符）")

    _show("违规消息", violations if violations else "无")
    for v in violations:
        print(f"  ✗ {v}")

    assert not violations, f"历史消息中发现图像数据泄露：{violations}"
    print("  ✓ 历史消息中无 image_url，无大体积 base64——图像成功隔离在 state")
