import pytest
import os
import io
import base64
from PIL import Image
from langchain_core.chat_history import InMemoryChatMessageHistory

from LangChain.guidance_agent import YuYuanGuidanceAgent


# ==================== Dummy 组件 ====================

class DummyRAG:
    def retrieve(self, query, k=3, rerank=False):
        if query == "empty":
            return []
        return ["九曲桥始建于明代", "桥体蜿蜒以延缓水流"]


class DummyChainSuccess:
    async def ainvoke(self, payload, config=None):
        assert "context" in payload
        assert "input" in payload
        assert "configurable" in config
        return "导览建议：从九曲桥入口开始参观。"


class DummyChainFail:
    async def ainvoke(self, payload, config=None):
        raise RuntimeError("mock llm failure")


@pytest.fixture
def agent(monkeypatch):
    # 避免触发真实 __init__（会初始化 ChatOpenAI）
    a = object.__new__(YuYuanGuidanceAgent)
    a.rag = DummyRAG()
    a.rerank = False
    a.rerank_top_k = 2
    a.session_store = {}
    return a


# ==================== 原有逻辑测试 ====================

def test_rag_retrieve_wrapper_with_results(agent):
    out = agent._rag_retrieve_wrapper("九曲桥")
    assert "- 九曲桥始建于明代" in out
    assert "- 桥体蜿蜒以延缓水流" in out


def test_rag_retrieve_wrapper_empty_query(agent):
    assert agent._rag_retrieve_wrapper("") == "（未收到有效输入）"


def test_rag_retrieve_wrapper_no_results(agent):
    assert agent._rag_retrieve_wrapper("empty") == "暂无相关景点历史记录。"


def test_get_session_history_create_and_trim(agent):
    """测试会话历史创建和裁剪到最近10条消息"""
    sid = "s1"
    h = agent._get_session_history(sid)
    assert isinstance(h, InMemoryChatMessageHistory)
    assert sid in agent.session_store

    # 添加12条消息，应该只保留最后10条
    for i in range(12):
        h.add_user_message(f"u{i}")
    h2 = agent._get_session_history(sid)
    assert len(h2.messages) == 10, "应该只保留最近10条消息"
    assert h2.messages[0].content == "u2", "第一条应该是u2（u0和u1被裁剪）"
    assert h2.messages[-1].content == "u11", "最后一条应该是u11"


def test_get_session_history_trim_boundary(agent):
    """测试边界情况：刚好10条时不裁剪，超过10条时才裁剪"""
    sid = "s_boundary"
    h = agent._get_session_history(sid)

    # 刚好10条，不应该裁剪
    for i in range(10):
        h.add_user_message(f"msg{i}")
    h2 = agent._get_session_history(sid)
    assert len(h2.messages) == 10, "刚好10条时不应裁剪"
    assert h2.messages[0].content == "msg0"

    # 添加第11条，应该触发裁剪，保留msg1-msg11（删除msg0）
    h.add_user_message("msg11")
    h3 = agent._get_session_history(sid)
    assert len(h3.messages) == 10, "超过10条后应裁剪到10条"
    assert h3.messages[0].content == "msg1", "msg0应被裁剪"
    assert h3.messages[-1].content == "msg11"


def test_clear_and_delete_session_history(agent):
    sid = "s2"
    h = agent._get_session_history(sid)
    h.add_user_message("hello")
    assert len(agent.session_store[sid].messages) == 1

    agent.clear_session_history(sid)
    assert len(agent.session_store[sid].messages) == 0

    agent.delete_session_history(sid)
    assert sid not in agent.session_store


@pytest.mark.asyncio
async def test_generate_guidance_success(agent):
    agent.chain_with_history = DummyChainSuccess()
    ans = await agent.generate_guidance("九曲桥为什么是弯的？", session_id="tourist_A")
    assert "导览建议" in ans


@pytest.mark.asyncio
async def test_generate_guidance_fail_returns_fallback(agent):
    agent.chain_with_history = DummyChainFail()
    ans = await agent.generate_guidance("九曲桥为什么是弯的？", session_id="tourist_A")
    assert ans == "导览助手暂时无法连接，请稍后再试。"


# ==================== 图像历史记录清理测试 ====================

def test_summarize_image_in_history_replaces_image_with_text(agent):
    """测试带图消息被替换为纯文本描述"""
    from langchain_core.messages import HumanMessage, AIMessage

    session_id = "img_session_01"
    history = agent._get_session_history(session_id)

    # 构造带图的多模态消息
    image_content = [
        {"type": "text", "text": "这张图片是什么景点？"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="}}
    ]

    history.add_user_message(HumanMessage(content=image_content))
    history.add_ai_message(AIMessage(content="这是九曲桥的著名景点。"))

    # 调用图像摘要方法
    agent._summarize_image_in_history(session_id, "这张图片是什么景点？")

    # 验证历史记录中的图像消息已被替换为纯文本
    user_msg = history.messages[0]
    assert isinstance(user_msg, HumanMessage)
    assert isinstance(user_msg.content, str), "内容应已变为字符串，不再是列表"
    assert "[已处理图像请求]" in user_msg.content, "应包含图像处理标记"
    assert "这张图片是什么景点？" in user_msg.content
    # 验证不再包含 image_url 数据
    assert "image_url" not in str(user_msg.content)


def test_summarize_image_in_history_multiple_messages(agent):
    """测试历史记录中有多条消息时，只替换最后一条带图的用户消息"""
    from langchain_core.messages import HumanMessage, AIMessage

    session_id = "img_session_02"
    history = agent._get_session_history(session_id)

    # 添加多条消息
    history.add_user_message("九曲桥在哪里？")
    history.add_ai_message(AIMessage(content="九曲桥位于豫园中心。"))

    # 添加带图消息
    image_content = [
        {"type": "text", "text": "描述这张图"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,fake_base64"}}
    ]
    history.add_user_message(HumanMessage(content=image_content))
    history.add_ai_message(AIMessage(content="图片显示的是古建筑。"))

    # 调用图像摘要
    agent._summarize_image_in_history(session_id, "描述这张图")

    # 验证：第一条用户消息不受影响
    assert history.messages[0].content == "九曲桥在哪里？"
    # 验证：第二条用户消息（带图）已被替换
    assert "[已处理图像请求]" in history.messages[2].content
    assert isinstance(history.messages[2].content, str)


def test_summarize_image_in_history_no_image_message(agent):
    """测试历史记录中没有带图消息时，方法不应报错"""
    session_id = "no_img_session"
    history = agent._get_session_history(session_id)

    # 只添加纯文本消息
    history.add_user_message("九曲桥的历史是什么？")
    history.add_ai_message("九曲桥始建于明代。")

    # 调用方法，不应抛出异常
    agent._summarize_image_in_history(session_id, "九曲桥的历史是什么？")

    # 验证消息未被修改
    assert history.messages[0].content == "九曲桥的历史是什么？"


def test_summarize_image_in_history_mixed_content(agent):
    """测试混合内容（文本+图像）被正确替换"""
    from langchain_core.messages import HumanMessage

    session_id = "mixed_session"
    history = agent._get_session_history(session_id)

    # 构造复杂的混合内容
    mixed_content = [
        {"type": "text", "text": "请分析这个建筑"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc123"}},
        {"type": "text", "text": "并给出导览建议"}
    ]

    history.add_user_message(HumanMessage(content=mixed_content))
    agent._summarize_image_in_history(session_id, "请分析这个建筑并给出导览建议")

    # 验证整个多模态内容被替换为单个文本字符串
    user_msg = history.messages[0]
    assert isinstance(user_msg.content, str)
    assert "[已处理图像请求]" in user_msg.content
    assert "data:image" not in user_msg.content


def test_session_history_keeps_only_10_messages_after_image_requests(agent):
    """测试图文请求后，历史记录仍然只保留最近10条"""
    from langchain_core.messages import HumanMessage, AIMessage

    session_id = "img_trim_session"
    history = agent._get_session_history(session_id)

    # 添加多条图文请求
    for i in range(6):
        image_content = [
            {"type": "text", "text": f"图片问题{i}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,img{i}"}}
        ]
        history.add_user_message(HumanMessage(content=image_content))
        history.add_ai_message(AIMessage(content=f"回答{i}"))
    agent.print_history(session_id)
    # 触发图像摘要
    for i in range(6):
        agent._summarize_image_in_history(session_id, f"图片问题{i}")
        # agent.print_history(session_id)

    # 添加更多消息使其超过10条
    for i in range(6, 10):
        history.add_user_message(f"文本问题{i}")
        history.add_ai_message(f"文本回答{i}")


    # 再次获取历史，验证只保留最近10条
    trimmed_history = agent._get_session_history(session_id)
    agent.print_history(session_id)
    assert len(trimmed_history.messages) == 10, "历史记录应裁剪到10条"

    # 验证所有图像数据都已被清理
    for msg in trimmed_history.messages:
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            assert "data:image" not in msg.content, "历史记录中不应包含base64图像数据"


# ==================== 图像处理工具函数测试 ====================

def test_image_file_to_base64(tmp_path):
    """测试图片路径 -> base64 压缩编码"""
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (2000, 2000), color="red")
    img.save(img_path)

    # 直接实例化，避免触发__init__中的LLM初始化
    agent_instance = object.__new__(YuYuanGuidanceAgent)
    encoded = agent_instance._image_file_to_base64(str(img_path))
    assert isinstance(encoded, str)
    assert len(encoded) > 0
    # 验证可解码
    decoded = base64.b64decode(encoded)
    assert decoded[:2] == b"\xff\xd8"


def test_image_file_to_base64_not_found():
    """路径不存在应抛出 FileNotFoundError"""
    agent_instance = object.__new__(YuYuanGuidanceAgent)
    with pytest.raises(FileNotFoundError):
        agent_instance._image_file_to_base64("/nonexistent/path.jpg")


def test_normalize_base64_pure():
    """纯 base64 应原样返回"""
    b64 = "SGVsbG8gV29ybGQ="
    agent_instance = object.__new__(YuYuanGuidanceAgent)
    result = agent_instance._normalize_base64(b64)
    assert result == b64


def test_normalize_base64_data_uri():
    """data:image/jpeg;base64,... 应去掉前缀，只留 payload"""
    data_uri = "data:image/jpeg;base64,SGVsbG8gV29ybGQ="
    agent_instance = object.__new__(YuYuanGuidanceAgent)
    result = agent_instance._normalize_base64(data_uri)
    assert result == "SGVsbG8gV29ybGQ="


def test_normalize_base64_empty():
    agent_instance = object.__new__(YuYuanGuidanceAgent)
    assert agent_instance._normalize_base64("") == ""
    assert agent_instance._normalize_base64(None) == ""


# ==================== 图文接口测试（mock LLM） ====================

class DummyLLMVision:
    """模拟支持 vision 的 LLM"""
    async def ainvoke(self, messages):
        # 验证收到的是多模态消息格式
        last_msg = messages[-1]
        assert hasattr(last_msg, "content")
        content = last_msg.content
        assert isinstance(content, list)
        has_text = any(c.get("type") == "text" for c in content)
        has_image = any(c.get("type") == "image_url" for c in content)
        assert has_text, "消息应包含文本部分"
        assert has_image, "消息应包含图像部分"
        return type("AIMessage", (), {"content": "图中有古建筑，红柱黄瓦，疑似亭台。"})()


@pytest.mark.asyncio
async def test_generate_guidance_with_image_no_image_provided(monkeypatch):
    """无图时返回提示"""
    # 避免触发真实 __init__（会初始化 ChatOpenAI）
    agent = object.__new__(YuYuanGuidanceAgent)
    agent.session_store = {}
    ans = await agent.generate_guidance_with_image("这是什么？")
    assert "请提供图片" in ans


@pytest.mark.asyncio
async def test_generate_guidance_with_image_base64_path_success(agent, tmp_path):
    """image_path 分支：验证路径读取、压缩、编码全链路"""
    img_path = tmp_path / "scene.jpg"
    Image.new("RGB", (800, 600), color="blue").save(img_path)

    class DummyChain:
        async def ainvoke(self, payload, config=None):
            # 验证 payload 包含正确的键
            assert "context" in payload
            assert "input" in payload
            # 验证 input 是多模态格式
            human_content = payload["input"]
            assert isinstance(human_content, list)
            img_part = next(c for c in human_content if c["type"] == "image_url")
            assert img_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
            return "蓝色场景。"

    # 初始化必要的属性
    agent.base_system_content = "你是豫园的专业导游。"
    agent.chain_with_history = DummyChain()

    ans = await agent.generate_guidance_with_image(
        "描述这张图",
        image_path=str(img_path),
        session_id="img_test_01",
    )
    assert ans == "蓝色场景。"


@pytest.mark.asyncio
async def test_generate_guidance_with_image_base64_direct_success(agent):
    """image_base64 分支：直接传 base64，验证 normalize 后送入 LLM"""
    class DummyChain:
        async def ainvoke(self, payload, config=None):
            human_content = payload["input"]
            img_part = next(c for c in human_content if c["type"] == "image_url")
            # data URI 中应含 /9j/ (JPEG magic bytes base64)
            assert "/9j/" in img_part["image_url"]["url"] or "SGVsb" in img_part["image_url"]["url"]
            return "红色场景。"

    # 初始化必要的属性
    agent.base_system_content = "你是豫园的专业导游。"
    agent.chain_with_history = DummyChain()

    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    ans = await agent.generate_guidance_with_image(
        "这是什么颜色？",
        image_base64=b64,
        session_id="img_test_02",
    )
    assert ans == "红色场景。"


@pytest.mark.asyncio
async def test_generate_guidance_with_image_llm_error_fallback(agent):
    """LLM 异常时返回兜底文案"""
    class BadLLM:
        async def ainvoke(self, messages):
            raise RuntimeError("network error")

    agent.llm = BadLLM()
    ans = await agent.generate_guidance_with_image(
        "描述这张图",
        image_base64=base64.b64encode(b"fake").decode(),
        session_id="img_test_03",
    )
    assert ans == "导览助手暂时无法连接，请稍后再试。"
