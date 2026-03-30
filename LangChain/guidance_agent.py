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
from langchain_core.prompts import PromptTemplate
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
            Tool(
                name="YuYuanKnowledge",
                func=self._rag_retrieve_wrapper,
                description="检索豫园文物和景点的历史知识。输入应为具体的景点名称。"
            )
        ]

        # 短期记忆

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
            detections: List[Dict],
            user_query: Optional[str] = None,
            ui_status: str = "idle"
    ) -> Optional[Dict]:
        """核心生成方法"""
        if not detections: return None

        major = max(detections, key=lambda x: x.get("confidence", 0))
        if major.get("confidence", 0) < config.CONFIDENCE_THRESHOLD:
            return None

        class_name = major.get("class", "未知物体")

        # --- 针对听障群体的 Prompt 注入 ---
        # 重点：要求 Agent 必须输出 JSON，且强调视觉引导
        agent_input = f"""
【视觉识别】当前看到：{class_name}
【视觉列表】{self._format_detections(detections)}
【用户状态】{ui_status}
{"【用户追问】" + user_query if user_query else ""}

任务：作为豫园导游，为听障人士提供视觉导览。
1. 如果不确定 {class_name} 的背景，请务必使用 YuYuanKnowledge 工具。
2. 最终回复必须是一个 JSON 对象，包含 display_text (简洁短句), action_cmd, vibrate。
3. 听障优化：避免长难句，多用视觉方位词（如“您的左前方”）。
"""

        try:
            # 执行 Agent 推理
            # 注意：invoke 里的参数名要和 Prompt 模板里的变量名对应
            response = await self.agent_executor.ainvoke({
                "input": agent_input
            })

            return self._parse_response(response.get("output", ""), major)

        except Exception as e:
            print(f"[Agent Error] {e}")
            return self._fallback_guidance(major)

    def _parse_response(self, text: str, major: dict) -> dict:
        """强化版 JSON 解析"""
        try:
            # 优先寻找 Markdown JSON 块
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return data
        except:
            pass

        # 彻底解析失败后的结构化补全
        return self._build_standard_response(text, major)

    def _build_standard_response(self, text: str, major: dict) -> dict:
        """统一输出格式"""
        # 过滤掉 LLM 可能带出的 Thought/Action 杂质文字
        clean_text = text.split("Final Answer:")[-1].strip()
        return {
            "type": "ai_guidance",
            "display_text": clean_text[:100],
            "action_cmd": {
                "type": "AR_HIGHLIGHT",
                "target": major.get("class", "object"),
                "effect": "pulse_glow"
            },
            "vibrate": "short"
        }