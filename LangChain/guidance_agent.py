import json
import re
from os.path import exists
from typing import Optional, List, Dict

# 1. 核心 Agent 逻辑 - 2026 标准路径
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
# 修正消息类型和 Prompt 导入
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import SecretStr

from . import config

# 假设你的 rag_engine 和 prompt_templates 已按前文建议准备好
from .rag_engine import YuYuanRAG


class YuYuanGuidanceAgent:
    """豫园 XR 听障辅助导游 Agent (ReAct 模式)"""

    def __init__(
            self,
            rag_engine: Optional[YuYuanRAG] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            model: Optional[str] = None,
            rerank: Optional[bool] = True,
            rerank_top_k: Optional[int] = 3,
            system_prompt: Optional[str] = None,
            prompt_template: Optional[PromptTemplate] = None,
    ):
        # --- RAG 引擎初始化 (逻辑保持) ---
        if rag_engine is not None:
            self.rag = rag_engine
        else:
            self.rag = YuYuanRAG(config.FAISS_INDEX_PATH)
            # ... (构建逻辑略)

        # 重排配置
        self.rerank = config.RERANK
        self.rerank_top_k = config.RERANK_TOP_K

        # --- LLM 初始化 (确保 streaming 开启，方便 WS 以后扩展) ---
        secure_api_key = SecretStr(api_key or config.DOUBao_API_KEY)
        self.llm = ChatOpenAI(
            model=model or config.DOUBao_MODEL,
            api_key=secure_api_key,  # 注意新版参数名是 api_key
            base_url=base_url or config.DOUBao_BASE_URL,
            temperature=0,  # 导游知识输出，建议温控为 0 以保证稳定性
            streaming=False,  # 目前先不开启流式，后续 WebSocket 版本再改为 True
        )

        # --- Tool 定义 ---
        self.tools = [
            # Tool(
            #     name="YuYuanKnowledge",
            #     func=self._rag_retrieve_wrapper,
            #     description="检索豫园文物和景点的历史知识。输入应为具体的景点名称。"
            # )
        ]

        #template
        self.prompt_template = prompt_template or ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="{context}"),
                HumanMessage(content="{input}")
            ]
        )

        # Agent 配置
        self.rerank = rerank or config.RERANK
        self.rerank_top_k = rerank_top_k or config.RERANK_TOP_K
        if not exists(config.SYSTEM_PROMPT_PATH):
            print(f"[警告] 系统提示文件不存在: {config.SYSTEM_PROMPT_PATH}")
            system_prompt = system_prompt or "你是豫园的导游，专门为听障人士提供视觉导览。请根据视觉输入和用户状态生成简洁的导览建议。"
        else:
            with open(config.SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        # --- Agent 构建 (核心修正点) ---
        # 如果 REACT_AGENT_PROMPT 是从 hub.pull 拿到的，直接用；
        # 如果是自定义的，确保它包含：input, tools, tool_names, agent_scratchpad
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
        )

    def _rag_retrieve_wrapper(self, query: str) -> str:
        """RAG 检索包装"""
        # 针对听障优化：检索结果也要简洁
        results = self.rag.retrieve(query, k=self.rerank_top_k, rerank= self.rerank)
        if not results: return "未找到该景点的历史记录。"
        return "\n".join([f"资料片段: {r}" for r in results])

    async def generate_guidance(  # 建议改为异步，适配 WebSocket
            self,
            user_query: Optional[str] = None,
    ) -> str:
        context = self._rag_retrieve_wrapper(user_query)
        prompt = self.prompt_template.invoke({
            "context": context,
            "input": user_query
        })
        response = self.agent.invoke(prompt)
        print(response.c)
        return response["messages"][-1]["content"]


