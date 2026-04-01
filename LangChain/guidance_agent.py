import os
import io
import base64
import time
from typing import Optional
from pydantic import SecretStr
from PIL import Image

# 核心 LangChain 导入
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 假设你的项目结构中包含这些
from . import config
from .rag_engine import YuYuanRAG


class YuYuanGuidanceAgent:
    """豫园 XR 听障辅助导游 Agent (2026 LCEL 高性能版)"""

    def __init__(
            self,
            rag_engine: Optional[YuYuanRAG] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            model: Optional[str] = None,
    ):
        # 1. RAG 引擎初始化
        self.rag = rag_engine or YuYuanRAG(config.FAISS_INDEX_PATH)
        self.rerank = config.RERANK
        self.rerank_top_k = config.RERANK_TOP_K

        # 2. LLM 初始化
        secure_api_key = SecretStr(api_key or config.DOUBao_API_KEY)
        self.llm = ChatOpenAI(
            model=model or config.DOUBao_MODEL,
            api_key=secure_api_key,
            base_url=base_url or config.DOUBao_BASE_URL,
            temperature=0,
            streaming=False,  # 建议开启流式，提升 XR 交互体验
        )

        # 3. 加载 System Prompt 文本
        self.base_system_content = self._load_system_prompt()
        print(f"[Agent] System Prompt 加载成功: {self.base_system_content[:60]}...")  # 只打印前 60 字，避免泄露敏感信息)

        # 4. 构建 Prompt 模板 (包含记忆占位符)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.base_system_content + "\n\n【实时背景资料】：\n{context}"),
            MessagesPlaceholder(variable_name="history"),  # 记忆将注入到这里
            # MessagesPlaceholder(variable_name="input"),
            ("human", "{input}"),
        ])

        # 5. 内存记忆存储 (Session 字典)
        self.session_store = {}

        # 6. 构建带记忆的完整链条
        # 链路：数据输入 -> Prompt -> LLM -> 字符串解析
        base_chain = self.prompt_template | self.llm | StrOutputParser()

        self.chain_with_history = RunnableWithMessageHistory(
            base_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _load_system_prompt(self) -> str:
        if os.path.exists(config.SYSTEM_PROMPT_PATH):
            with open(config.SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
                return f.read()
        return "你是豫园的专业导游，请为听障人士提供简洁、具象的视觉导览。优先参考背景资料。"

    #======================对话历史管理============================
    def _get_session_history(self, session_id: str):
        """获取或创建会话历史，并限制记忆长度"""
        if session_id not in self.session_store:
            self.session_store[session_id] = InMemoryChatMessageHistory()

        # 限制记忆：只保留最近 10 条消息（约 5 轮对话），防止 Token 溢出
        h = self.session_store[session_id]
        if len(h.messages) > 10:
            h.messages = h.messages[-10:]
        return h

    def clear_session_history(self, session_id: str):
        if session_id not in self.session_store:
            return
        self.session_store[session_id].clear()
        print(f"[Session] 已清除会话历史: {session_id}")

    def delete_session_history(self, session_id: str):
        if session_id not in self.session_store:
            return
        del self.session_store[session_id]
        print(f"[Session] 已销毁 {session_id} 所有数据")

    def _rag_retrieve_wrapper(self, query: str) -> str:
        """执行 RAG 检索并格式化"""
        if not query:
            return "（未收到有效输入）"
        results = self.rag.retrieve(query, k=self.rerank_top_k, rerank=self.rerank)
        if not results:
            return "暂无相关景点历史记录。"
        return "\n".join([f"- {r}" for r in results])

    def _summarize_image_in_history(self, session_id: str, query: str):
        """将历史记录中最后一条带图的消息替换为纯文本，节省 Token"""
        history = self.session_store.get(session_id)
        if history and len(history.messages) > 0:
            # 倒数第二条通常是刚发出的 User 消息（最后一条是 AI 的回答）
            # 如果你连续调用，建议检查一下内容
            for i in range(len(history.messages) - 1, -1, -1):
                msg = history.messages[i]
                if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
                    # 检查 content 里面是否有 image_url 类型
                    has_image = any(isinstance(item, dict) and item.get("type") == "image_url" for item in msg.content)
                    if has_image:
                        # 替换为纯文本，彻底切断下一轮的 Base64 传输
                        history.messages[i].content = f"[已处理图像请求]: {query}"
                        break

    def print_history(self, session_id: str):
        history = self.session_store.get(session_id)
        if not history:
            print(f"[Session] 无历史记录: {session_id}")
            return
        print(f"[Session] 历史记录 ({session_id}):")
        # print("=" * 20)
        # print(history.messages)
        # print("=" * 20)
        # print(history)
        for msg in history.messages:
            # prefix = "AI" if msg.is_ai else "User"
            prefix = "AI" if isinstance(msg, AIMessage) else "User"
            # print(msg)
            content_preview = str(msg.content)[:60].replace("\n", " ") + ("..." if len(str(msg.content)) > 60 else "")
            print(f"  [{prefix}] {content_preview}")

    # 无图像识别的文本导览接口，适合纯文本查询
    async def generate_guidance(
            self,
            user_query: str,
            session_id: str = "unity_user_01"  # 由 Unity 端传入唯一 ID
    ) -> str:
        """生成导览建议（文本模式，支持异步调用）"""
        # A. 检索背景知识
        context = self._rag_retrieve_wrapper(user_query)

        # B. 调用带记忆的链条
        # 注意：configurable 参数是告诉 LangChain 应该用哪个 session_id
        try:
            answer = await self.chain_with_history.ainvoke(
                {"context": context, "input": user_query},
                config={"configurable": {"session_id": session_id}}
            )
            return answer
        except Exception as e:
            print(f"[Agent Error] {e}")
            return "导览助手暂时无法连接，请稍后再试。"

    # ============================================
    # =========图像相关的辅助函数和接口===========
    # ============================================
    def _image_file_to_base64(self, image_path: str, max_side: int = 1024, jpeg_quality: int = 85) -> str:
        """读取图片并压缩为 JPEG，再转为 base64 字符串。"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")

        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            rgb_img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _normalize_base64(self, image_base64: str) -> str:
        """兼容 data URL 与纯 base64。"""
        if not image_base64:
            return ""
        raw = image_base64.strip()
        if "," in raw and raw.startswith("data:"):
            return raw.split(",", 1)[1]
        return raw

    async def generate_guidance_with_image(
            self,
            user_query: str,
            session_id: str = "unity_user_01",
            image_path: Optional[str] = None,
            image_base64: Optional[str] = None,
    ) -> str:
        """生成导览建议（图文模式）。支持直接传图路径或 base64。"""
        try:
            if not image_path and not image_base64:
                return "请提供图片路径或 base64 图片数据。"

            if image_base64:
                encoded = self._normalize_base64(image_base64)
            else:
                encoded = self._image_file_to_base64(image_path)

            if not encoded:
                return "图片数据无效，请重新上传。"

            start_time = time.perf_counter()
            context = self._rag_retrieve_wrapper(user_query)
            end_time = time.perf_counter()
            print("[RAG] 检索耗时: {:.2f} 秒".format(end_time - start_time))

            # history = self._get_session_history(session_id)
            system_content = self.base_system_content + "\n\n【实时背景资料】：\n" + context
            effective_query = user_query or "请结合这张图片给出导览说明。"

            human_content = [
                {"type": "text", "text": effective_query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
                }
            ]

            # messages = [SystemMessage(content=system_content), *history.messages, HumanMessage(content=human_content)]
            # llm_output = await self.llm.ainvoke(messages)

            start_time = time.perf_counter()
            output = await self.chain_with_history.ainvoke({"context": context, "input": human_content},
                                            config={"configurable": {"session_id": session_id}})
            end_time = time.perf_counter()
            print("[LLM] 响应耗时: {:.2f} 秒".format(end_time - start_time))
            # if self.output_total_tokens:
            #     print(f"[LLM] 输出 Tokens: {output.total_tokens}, 输出内容 Tokens: {output.output_tokens}")
            if isinstance(output, str):
                answer = output
            else:
                answer = str(output)

            # 去除对话历史中的图像存储
            self._summarize_image_in_history(session_id, effective_query)

            # 配合 llm 使用的手动调整 history
            # history.add_user_message(f"[图像问题] {effective_query}")
            # history.add_ai_message(answer)
            return answer
        except Exception as e:
            print(f"[Agent Error] {e}")
            return "导览助手暂时无法连接，请稍后再试。"

# --- 使用示例 ---
# agent = YuYuanGuidanceAgent()
# response = await agent.generate_guidance("九曲桥为什么是弯的？", session_id="tourist_A")
# response = await agent.generate_guidance_with_image("这是什么景点？", image_path="E:/Master/ultralytics-main/TestImgs/img.png")
