import os
from typing import Optional, List, Dict
from pydantic import SecretStr

# 核心 LangChain 导入
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import InMemoryChatMessageHistory

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

        # 4. 构建 Prompt 模板 (包含记忆占位符)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.base_system_content + "\n\n【实时背景资料】：\n{context}"),
            MessagesPlaceholder(variable_name="history"),  # 记忆将注入到这里
            ("human", "{input}")
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

    def _get_session_history(self, session_id: str):
        """获取或创建会话历史，并限制记忆长度"""
        if session_id not in self.session_store:
            self.session_store[session_id] = InMemoryChatMessageHistory()

        # 限制记忆：只保留最近 10 条消息（约 5 轮对话），防止 Token 溢出
        h = self.session_store[session_id]
        if len(h.messages) > 10:
            h.messages = h.messages[-10:]
        return h

    def _rag_retrieve_wrapper(self, query: str) -> str:
        """执行 RAG 检索并格式化"""
        if not query: return "（未收到有效输入）"
        results = self.rag.retrieve(query, k=self.rerank_top_k, rerank=self.rerank)
        if not results: return "暂无相关景点历史记录。"
        return "\n".join([f"- {r}" for r in results])

    async def generate_guidance(
            self,
            user_query: str,
            session_id: str = "unity_user_01"  # 由 Unity 端传入唯一 ID
    ) -> str:
        """生成导览建议（支持异步调用）"""
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

# --- 使用示例 ---
# agent = YuYuanGuidanceAgent()
# response = await agent.generate_guidance("九曲桥为什么是弯的？", session_id="tourist_A")