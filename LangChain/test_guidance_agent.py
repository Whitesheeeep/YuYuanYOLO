"""
单元测试：YuYuanGuidanceAgent.
================================
策略：绕过真实 __init__（避免初始化 LLM/RAG），
通过 object.__new__ + 手动赋属性来测试各个方法。
需要真实 agent 执行的测试使用轻量 DummyAgent mock。
"""

import base64
import io

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image

from LangChain.guidance_agent import YuYuanAgentState, YuYuanGuidanceAgent, _apply_nms

# ============================================================================
# 审查输出辅助
# ============================================================================


def _section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _show(label: str, value):
    val_str = str(value)
    if len(val_str) > 120:
        val_str = val_str[:120] + "..."
    print(f"  {label:<28} {val_str}")


# ============================================================================
# 公共 Dummy 组件
# ============================================================================


class DummyRAG:
    def retrieve_with_score(self, query, k=3, rerank=False):
        if not query:
            return []
        if query == "empty":
            return []
        return [("九曲桥始建于明代", 0.95), ("桥体蜿蜒以延缓水流", 0.88)]


class DummyAgent:
    """模拟 create_agent 返回的 agent 对象，用于测试 generate_guidance* 接口。."""

    def __init__(self, response="导览建议：从九曲桥入口开始参观。", raise_error=False):
        self._response = response
        self._raise_error = raise_error
        self._last_input = None

    async def ainvoke(self, payload, config=None):
        if self._raise_error:
            raise RuntimeError("mock llm failure")
        self._last_input = payload
        return {"messages": [AIMessage(content=self._response)]}

    def get_state(self, config):
        return None


class DummyCheckpointer:
    def __init__(self):
        self.storage = {"session_a": "data_a", "other": "data_b"}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def agent():
    """创建最小化 agent 实例，跳过真实 __init__。."""
    a = object.__new__(YuYuanGuidanceAgent)
    a.rag = DummyRAG()
    a._yolo_model = None
    a.agent_history_limit = 10
    a.agent = DummyAgent()
    a._checkpointer = DummyCheckpointer()
    return a


# ============================================================================
# _apply_nms 测试
# ============================================================================


def test_apply_nms_empty():
    _section("NMS | 空输入")
    result = _apply_nms([], [])
    _show("输入 boxes/scores", "[] / []")
    _show("输出 keep 列表", result)
    assert result == []
    print("  ✓ 空输入返回空列表")


def test_apply_nms_single():
    _section("NMS | 单框")
    boxes = [[0, 0, 10, 10]]
    scores = [0.9]
    result = _apply_nms(boxes, scores)
    _show("输入 boxes", boxes)
    _show("输入 scores", scores)
    _show("输出 keep 列表", result)
    assert result == [0]
    print("  ✓ 单框直接保留，索引为 0")


def test_apply_nms_suppresses_overlap():
    _section("NMS | 高重叠框抑制")
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11]]
    scores = [0.9, 0.8]
    keep = _apply_nms(boxes, scores, iou_threshold=0.5)
    _show("输入 boxes", boxes)
    _show("输入 scores", scores)
    _show("iou_threshold", 0.5)
    _show("输出 keep 列表", keep)
    assert 0 in keep
    assert 1 not in keep
    print("  ✓ 高 IoU 时低分框（索引 1）被抑制，高分框（索引 0）保留")


def test_apply_nms_keeps_non_overlap():
    _section("NMS | 无重叠框全部保留")
    boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
    scores = [0.9, 0.8]
    keep = _apply_nms(boxes, scores, iou_threshold=0.5)
    _show("输入 boxes", boxes)
    _show("输入 scores", scores)
    _show("输出 keep 列表", keep)
    assert set(keep) == {0, 1}
    print("  ✓ 两框不重叠，全部保留")


# ============================================================================
# 图像辅助方法测试
# ============================================================================


def test_image_file_to_base64_compresses_large_image(tmp_path):
    _section("图像编码 | 大图压缩 + base64 编码")
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (2000, 2000), color="red").save(img_path)
    a = object.__new__(YuYuanGuidanceAgent)

    encoded = a._image_file_to_base64(str(img_path))
    decoded = base64.b64decode(encoded)

    _show("原始尺寸", "2000×2000 px")
    _show("max_side 限制", 1024)
    _show("编码后长度 (字符)", len(encoded))
    _show("解码头 2 字节 (hex)", decoded[:2].hex())
    _show("JPEG magic bytes", "ffd8")

    assert isinstance(encoded, str) and len(encoded) > 0
    assert decoded[:2] == b"\xff\xd8"
    print("  ✓ 输出为合法 JPEG base64 字符串")


def test_image_file_to_base64_not_found():
    _section("图像编码 | 路径不存在抛出异常")
    path = "/nonexistent/path.jpg"
    _show("输入路径", path)
    a = object.__new__(YuYuanGuidanceAgent)
    with pytest.raises(FileNotFoundError) as exc_info:
        a._image_file_to_base64(path)
    _show("捕获异常类型", type(exc_info.value).__name__)
    _show("异常消息", str(exc_info.value))
    print("  ✓ 正确抛出 FileNotFoundError")


def test_normalize_base64_pure():
    _section("Base64 标准化 | 纯 base64 原样返回")
    b64 = "SGVsbG8gV29ybGQ="
    a = object.__new__(YuYuanGuidanceAgent)
    result = a._normalize_base64(b64)
    _show("输入", b64)
    _show("输出", result)
    assert result == b64
    print("  ✓ 纯 base64 不被修改")


def test_normalize_base64_data_uri():
    _section("Base64 标准化 | data URI 前缀去除")
    data_uri = "data:image/jpeg;base64,SGVsbG8gV29ybGQ="
    a = object.__new__(YuYuanGuidanceAgent)
    result = a._normalize_base64(data_uri)
    _show("输入", data_uri)
    _show("输出 (去掉前缀后)", result)
    assert result == "SGVsbG8gV29ybGQ="
    print("  ✓ data: 前缀成功去除，只保留 payload")


def test_normalize_base64_empty():
    _section("Base64 标准化 | 空/None 输入返回空字符串")
    a = object.__new__(YuYuanGuidanceAgent)
    r1 = a._normalize_base64("")
    r2 = a._normalize_base64(None)
    _show("输入 ''  → 输出", repr(r1))
    _show("输入 None → 输出", repr(r2))
    assert r1 == ""
    assert r2 == ""
    print("  ✓ 空值均返回 ''")


# ============================================================================
# 会话管理测试
# ============================================================================


def test_clear_session_history_removes_matching_keys(agent):
    _section("会话管理 | clear_session_history 按 thread_id 清除")
    storage_before = {
        ("session_a", "ns", "ckpt1"): "x",
        ("session_a", "ns", "ckpt2"): "y",
        ("other_session", "ns", "ckpt1"): "z",
    }
    agent._checkpointer.storage = dict(storage_before)
    _show("清除前 key 数量", len(agent._checkpointer.storage))
    _show("目标 thread_id", "session_a")

    agent.clear_session_history("session_a")

    remaining = list(agent._checkpointer.storage.keys())
    _show("清除后 key 数量", len(remaining))
    _show("剩余 key", remaining)

    assert ("other_session", "ns", "ckpt1") in remaining
    assert not any(k[0] == "session_a" for k in remaining)
    print("  ✓ session_a 的 2 条记录已删除，other_session 不受影响")


def test_delete_session_history_delegates_to_clear(agent, monkeypatch):
    _section("会话管理 | delete_session_history 委托给 clear")
    called = []
    monkeypatch.setattr(agent, "clear_session_history", lambda sid: called.append(sid))
    agent.delete_session_history("sess_x")
    _show("delete 调用 session_id", "sess_x")
    _show("clear 收到的 session_id 列表", called)
    assert called == ["sess_x"]
    print("  ✓ delete_session_history 正确转发给 clear_session_history")


# ============================================================================
# generate_guidance 接口测试
# ============================================================================


@pytest.mark.asyncio
async def test_generate_guidance_success(agent):
    _section("generate_guidance | 正常返回导览建议")
    query = "九曲桥为什么是弯的？"
    session = "tourist_A"
    _show("输入 query", query)
    _show("session_id", session)

    ans = await agent.generate_guidance(query, session_id=session)

    _show("Agent 返回", ans)
    assert "导览建议" in ans
    print("  ✓ 返回包含'导览建议'关键词")


@pytest.mark.asyncio
async def test_generate_guidance_passes_text_only(agent):
    _section("generate_guidance | 消息仅含纯文字，无图像字段")
    query = "三穗堂介绍"
    _show("输入 query", query)

    await agent.generate_guidance(query, session_id="tourist_B")
    last_input = agent.agent._last_input

    msg = last_input["messages"][0]
    _show("payload keys", list(last_input.keys()))
    _show("消息类型", type(msg).__name__)
    _show("消息 content 类型", type(msg.content).__name__)
    _show("content 值 (前 80 字)", str(msg.content)[:80])
    _show("包含 'current_image_base64'", "current_image_base64" in last_input)
    _show("包含 'image_url'", "image_url" in str(msg.content))

    assert "messages" in last_input
    assert "current_image_base64" not in last_input
    assert isinstance(msg, HumanMessage)
    assert isinstance(msg.content, str)
    assert "image_url" not in str(msg.content)
    print("  ✓ 纯文本查询：消息内容为 str，payload 不含图像字段")


@pytest.mark.asyncio
async def test_generate_guidance_fail_returns_fallback(agent):
    _section("generate_guidance | LLM 异常时返回兜底文案")
    agent.agent = DummyAgent(raise_error=True)
    _show("mock 行为", "ainvoke 抛出 RuntimeError")

    ans = await agent.generate_guidance("九曲桥", session_id="tourist_A")

    _show("实际返回", ans)
    assert ans == "导览助手暂时无法连接，请稍后再试。"
    print("  ✓ 异常被捕获，返回兜底文案")


# ============================================================================
# generate_guidance_with_image 接口测试
# ============================================================================


@pytest.mark.asyncio
async def test_generate_guidance_with_image_no_image_provided(agent):
    _section("generate_guidance_with_image | 无图像时返回提示")
    _show("image_path", None)
    _show("image_base64", None)

    ans = await agent.generate_guidance_with_image("这是什么？")

    _show("实际返回", ans)
    assert "请提供图片" in ans
    print("  ✓ 无图像输入时返回提示文案")


@pytest.mark.asyncio
async def test_generate_guidance_with_image_invalid_base64_returns_hint(agent):
    _section("generate_guidance_with_image | 空 base64 返回提示")
    _show("image_base64", repr(""))

    ans = await agent.generate_guidance_with_image("这是什么？", image_base64="")

    _show("实际返回", ans)
    assert "请提供图片" in ans or "无效" in ans
    print("  ✓ 空 base64 返回提示文案（非抛错）")


@pytest.mark.asyncio
async def test_generate_guidance_with_image_passes_image_in_state(agent, tmp_path):
    _section("generate_guidance_with_image | 图像存入 state，消息为纯文字")
    img_path = tmp_path / "scene.jpg"
    Image.new("RGB", (800, 600), color="blue").save(img_path)
    _show("图像文件", str(img_path))
    _show("图像尺寸", "800×600 px")

    await agent.generate_guidance_with_image(
        "描述这张图",
        image_path=str(img_path),
        session_id="img_test_01",
    )

    last_input = agent.agent._last_input
    msg = last_input["messages"][0]
    b64_in_state = last_input.get("current_image_base64", "")

    _show("payload keys", list(last_input.keys()))
    _show("current_image_base64 长度", len(b64_in_state))
    _show("消息 content 类型", type(msg.content).__name__)
    _show("消息内容 (前 60 字)", str(msg.content)[:60])
    _show("消息含 image_url", "image_url" in str(msg.content))

    assert "current_image_base64" in last_input
    assert b64_in_state != ""
    assert isinstance(msg, HumanMessage)
    assert isinstance(msg.content, str)
    assert "image_url" not in str(msg.content)
    print("  ✓ 图像存入 state，LLM 消息为纯文字——工具可按需读取图像")


@pytest.mark.asyncio
async def test_generate_guidance_with_image_base64_direct(agent):
    _section("generate_guidance_with_image | 直传 base64 存入 state")
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    _show("base64 长度 (字符)", len(b64))

    await agent.generate_guidance_with_image(
        "这是什么颜色？",
        image_base64=b64,
        session_id="img_test_02",
    )

    last_input = agent.agent._last_input
    _show("state 中 base64 长度", len(last_input.get("current_image_base64", "")))
    _show("state 与输入一致", last_input["current_image_base64"] == b64)
    _show("消息 content 类型", type(last_input["messages"][0].content).__name__)

    assert last_input["current_image_base64"] == b64
    assert isinstance(last_input["messages"][0].content, str)
    print("  ✓ base64 原样存入 state，消息为纯文字")


@pytest.mark.asyncio
async def test_generate_guidance_with_image_data_uri_stripped(agent):
    _section("generate_guidance_with_image | data URI 前缀在存入 state 前去除")
    raw_b64 = base64.b64encode(b"fake_jpeg_data").decode()
    data_uri = f"data:image/jpeg;base64,{raw_b64}"
    _show("输入 (data URI 前 60 字)", data_uri[:60])
    _show("期望存入 state 的纯 base64", raw_b64)

    await agent.generate_guidance_with_image(
        "描述图像",
        image_base64=data_uri,
        session_id="img_test_03",
    )

    stored = agent.agent._last_input["current_image_base64"]
    _show("实际存入 state", stored)
    _show("以 'data:' 开头", stored.startswith("data:"))

    assert stored == raw_b64
    assert not stored.startswith("data:")
    print("  ✓ data: 前缀去除，state 中只存纯 payload")


@pytest.mark.asyncio
async def test_generate_guidance_with_image_llm_error_fallback(agent):
    _section("generate_guidance_with_image | Agent 异常时返回兜底文案")
    agent.agent = DummyAgent(raise_error=True)
    _show("mock 行为", "ainvoke 抛出 RuntimeError")

    ans = await agent.generate_guidance_with_image(
        "描述这张图",
        image_base64=base64.b64encode(b"fake").decode(),
        session_id="img_test_err",
    )

    _show("实际返回", ans)
    assert ans == "导览助手暂时无法连接，请稍后再试。"
    print("  ✓ 异常被捕获，返回兜底文案")


# ============================================================================
# YuYuanAgentState 结构测试
# ============================================================================


def test_agent_state_has_image_field():
    _section("YuYuanAgentState | current_image_base64 字段存在且默认为空")
    state = YuYuanAgentState(messages=[], current_image_base64="")
    _show("state['current_image_base64']", repr(state["current_image_base64"]))
    assert state["current_image_base64"] == ""
    print("  ✓ 字段存在，默认值为 ''")


def test_agent_state_image_field_stores_value():
    _section("YuYuanAgentState | current_image_base64 正确存储赋值")
    b64 = "SGVsbG8="
    state = YuYuanAgentState(messages=[], current_image_base64=b64)
    _show("赋值", b64)
    _show("读取", state["current_image_base64"])
    assert state["current_image_base64"] == b64
    print("  ✓ 赋值与读取一致")
