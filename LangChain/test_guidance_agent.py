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
    sid = "s1"
    h = agent._get_session_history(sid)
    assert isinstance(h, InMemoryChatMessageHistory)
    assert sid in agent.session_store

    for i in range(12):
        h.add_user_message(f"u{i}")
    h2 = agent._get_session_history(sid)
    assert len(h2.messages) == 10
    assert h2.messages[0].content == "u2"


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


# ==================== 图像处理工具函数测试 ====================

def test_image_file_to_base64(tmp_path):
    """测试图片路径 -> base64 压缩编码"""
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (2000, 2000), color="red")
    img.save(img_path)

    encoded = YuYuanGuidanceAgent()._image_file_to_base64(str(img_path))
    assert isinstance(encoded, str)
    assert len(encoded) > 0
    # 验证可解码
    decoded = base64.b64decode(encoded)
    assert decoded[:2] == b"\xff\xd8"


def test_image_file_to_base64_not_found():
    """路径不存在应抛出 FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        YuYuanGuidanceAgent()._image_file_to_base64("/nonexistent/path.jpg")


def test_normalize_base64_pure():
    """纯 base64 应原样返回"""
    b64 = "SGVsbG8gV29ybGQ="
    result = YuYuanGuidanceAgent()._normalize_base64(b64)
    assert result == b64


def test_normalize_base64_data_uri():
    """data:image/jpeg;base64,... 应去掉前缀，只留 payload"""
    data_uri = "data:image/jpeg;base64,SGVsbG8gV29ybGQ="
    result = YuYuanGuidanceAgent()._normalize_base64(data_uri)
    assert result == "SGVsbG8gV29ybGQ="


def test_normalize_base64_empty():
    assert YuYuanGuidanceAgent()._normalize_base64("") == ""
    assert YuYuanGuidanceAgent()._normalize_base64(None) == ""


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
async def test_generate_guidance_with_image_no_image_provided():
    """无图时返回提示"""
    agent = YuYuanGuidanceAgent()
    ans = await agent.generate_guidance_with_image("这是什么？")
    assert "请提供图片" in ans


@pytest.mark.asyncio
async def test_generate_guidance_with_image_base64_path_success(agent, tmp_path):
    """image_path 分支：验证路径读取、压缩、编码全链路"""
    img_path = tmp_path / "scene.jpg"
    Image.new("RGB", (800, 600), color="blue").save(img_path)

    class DummyLLM:
        async def ainvoke(self, messages):
            last = messages[-1]
            content = last.content
            assert isinstance(content, list)
            img_part = next(c for c in content if c["type"] == "image_url")
            assert img_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
            return type("AIMessage", (), {"content": "蓝色场景。"})()

    agent.llm = DummyLLM()
    ans = await agent.generate_guidance_with_image(
        "描述这张图",
        image_path=str(img_path),
        session_id="img_test_01",
    )
    assert ans == "蓝色场景。"


@pytest.mark.asyncio
async def test_generate_guidance_with_image_base64_direct_success(agent):
    """image_base64 分支：直接传 base64，验证 normalize 后送入 LLM"""
    class DummyLLM:
        async def ainvoke(self, messages):
            last = messages[-1]
            content = last.content
            img_part = next(c for c in content if c["type"] == "image_url")
            # data URI 中应含 /9j/ (JPEG magic bytes base64)
            assert "/9j/" in img_part["image_url"]["url"] or "SGVsb" in img_part["image_url"]["url"]
            return type("AIMessage", (), {"content": "红色场景。"})()

    agent.llm = DummyLLM()
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
