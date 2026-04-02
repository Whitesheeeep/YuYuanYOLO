import pytest
import os
import io
import base64 as b64_module
from PIL import Image
import tempfile

from LangChain.guidance_agent import YuYuanGuidanceAgent
from dotenv import load_dotenv
load_dotenv()

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_generate_guidance_with_real_rag_and_llm():
    """集成测试：使用真实 RAG + 真实 LLM，验证主流程可用。"""
    agent = YuYuanGuidanceAgent()

    answer = await agent.generate_guidance(
        "九曲桥为什么是弯的？请用两句话简洁回答。",
        session_id="integration_user_01",
    )
    print(answer)

    assert isinstance(answer, str)
    assert answer.strip() != ""
    assert answer != "导览助手暂时无法连接，请稍后再试。"


@pytest.mark.asyncio
async def test_generate_guidance_with_memory_real_stack():
    """集成测试：连续提问，验证真实链路下的会话记忆不报错。"""
    agent = YuYuanGuidanceAgent()
    session_id = "integration_user_02"

    first = await agent.generate_guidance("我现在在九曲桥。", session_id=session_id)
    second = await agent.generate_guidance("结合我刚才的位置，下一步推荐看什么？", session_id=session_id)
    print(first)
    print(second)
    assert isinstance(first, str) and first.strip() != ""
    assert isinstance(second, str) and second.strip() != ""
    assert second != "导览助手暂时无法连接，请稍后再试。"


# ==================== 图像接口集成测试 ====================

@pytest.mark.asyncio
async def test_generate_guidance_with_image_path():
    """集成测试：传入图片路径，自动压缩+base64 上传，验证完整 vision 链路。"""
    agent = YuYuanGuidanceAgent()

    # 创建临时大图（触发压缩）
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img_path = f.name

    try:
        img = Image.new("RGB", (1600, 1200), color=(50, 150, 200))
        img.save(img_path)

        answer = await agent.generate_guidance_with_image(
            "这张图片里有什么, 15 字以内，快速回答",
            image_path=r"D:\Yuyuan_2\YuyuanYOLO\test.png",
            session_id="integration_img_01",
        )
        print(answer)
        assert isinstance(answer, str)
        assert answer.strip() != ""
        assert answer != "导览助手暂时无法连接，请稍后再试。"
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


@pytest.mark.asyncio
async def test_generate_guidance_with_image_base64():
    """集成测试：传入纯 base64 字符串，验证 base64 直传通路。"""
    agent = YuYuanGuidanceAgent()

    img = Image.new("RGB", (200, 200), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    b64_str = b64_module.b64encode(buffer.getvalue()).decode("utf-8")

    answer = await agent.generate_guidance_with_image(
        "图片里是什么颜色？",
        image_base64=b64_str,
        session_id="integration_img_02",
    )
    print(answer)
    assert isinstance(answer, str)
    assert answer.strip() != ""
    assert answer != "导览助手暂时无法连接，请稍后再试。"


@pytest.mark.asyncio
async def test_generate_guidance_with_image_data_uri():
    """集成测试：传入完整 data URI（带前缀），验证截断前缀后正常处理。"""
    agent = YuYuanGuidanceAgent()

    # 生成一个真实 JPEG 的 data URI
    img = Image.new("RGB", (100, 100), color="green")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    b64_data = b64_module.b64encode(buffer.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64_data}"

    answer = await agent.generate_guidance_with_image(
        "图片里的场景是什么？",
        image_base64=data_uri,
        session_id="integration_img_03",
    )
    print(answer)
    assert isinstance(answer, str)
    assert answer.strip() != ""


@pytest.mark.asyncio
async def test_generate_guidance_with_image_no_image_returns_hint():
    """集成测试：既不传 image_path 也不传 image_base64 时，返回提示而非抛错。"""
    agent = YuYuanGuidanceAgent()

    answer = await agent.generate_guidance_with_image(
        "这是什么？",
        session_id="integration_img_04",
    )
    print(answer)
    assert "请提供图片" in answer or "请重新上传" in answer
