import base64
import io
import os

from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

import config

llm = ChatOpenAI(
    model=config.DOUBao_MODEL,
    base_url=config.DOUBao_BASE_URL,
    api_key=config.DOUBao_API_KEY,
    streaming=False
)
prompt_template = ChatPromptTemplate.from_messages([
    # ("system", self.base_system_content + "\n\n【实时背景资料】：\n{context}"),
    ("system", "{context}"),
    # MessagesPlaceholder(variable_name="input"),
    MessagesPlaceholder(variable_name="input"), # 这里接收原始消息列表
])
chain = prompt_template | llm
image_path = r"D:\Yuyuan_2\YuyuanYOLO\test2.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"图片不存在: {image_path}")

with Image.open(image_path) as img:
    rgb_img = img.convert("RGB")
    rgb_img.thumbnail((640, 640))
    buf = io.BytesIO()
    rgb_img.save(buf, format="JPEG", quality=80, optimize=True)
    rgb_img.show()
base = base64.b64encode(buf.getvalue()).decode("utf-8")
text =[
    {"type": "text", "text": "请根据图片内容，15 字以内，快速回答这张图片里有什么？"},
    {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base}"}
    }
]
print(llm.invoke([HumanMessage(content=text)]))
print(chain.invoke({"context": "", "input": [HumanMessage(content=text)]}))
# print(llm.invoke([HumanMessage(content=text)]))
