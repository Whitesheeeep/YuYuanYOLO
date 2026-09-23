import asyncio

from dotenv import load_dotenv
from tenacity import sleep

from LangChain.guidance_agent import YuYuanGuidanceAgent

load_dotenv()  # 这会自动把 .env 里的内容注入到 os.environ


async def generate_guidance_with_image_base64():
    """集成测试：传入图片路径，自动压缩+base64 上传，验证完整 vision 链路。."""
    agent = YuYuanGuidanceAgent()

    answer = await agent.generate_guidance_with_image(
        "这张图片里有什么, 15 字以内，快速回答",
        image_path=r"D:\Yuyuan_2\YuyuanYOLO\test2.jpg",
        session_id="integration_img_01",
    )
    print(answer)


if __name__ == "__main__":
    asyncio.run(generate_guidance_with_image_base64())
    while True:
        sleep(1)
