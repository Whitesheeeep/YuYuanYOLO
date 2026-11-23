from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

import cv2
import numpy as np
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.tools import tool

# 核心 LangChain 导入
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from PIL import Image
from pydantic import SecretStr

from . import config
from .rag_engine import YuYuanRAG

# ============================================================================
# 自定义 Agent State（在标准 AgentState 基础上增加图像字段）
# ============================================================================


class YuYuanAgentState(AgentState):
    """扩展 AgentState，新增当前帧图像字段。 图像不放入 messages，避免 LLM 在初始轮直接分析； 工具按需从 state 中读取。.
    """

    current_image_base64: str = ""


# ============================================================================
# 模块级工具函数
# ============================================================================


def _apply_nms(boxes: list, scores: list, iou_threshold: float = 0.5) -> list:
    """非极大值抑制（NMS）。."""
    if not boxes:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


# ============================================================================
# Agent 主类
# ============================================================================


class YuYuanGuidanceAgent:
    """豫园 XR 听障辅助导游 Agent 架构：create_agent + YuYuanAgentState + checkpointer + middleware 图像单独存入 state，工具通过 ToolRuntime
    按需读取，LLM 初始轮不直接分析图像。.
    """

    def __init__(
        self,
        rag_engine: YuYuanRAG | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        yolo_model=None,
        agent_history_limit: int = 10,
    ):
        # 1. 外部依赖注入
        self.rag = rag_engine or YuYuanRAG(config.FAISS_INDEX_PATH)
        self._yolo_model = yolo_model  # 由 detection_websocket.py 传入已加载的 YOLO 实例
        self.agent_history_limit = agent_history_limit

        # 2. LLM 初始化
        self.llm = ChatOpenAI(
            model=model or config.DOUBao_MODEL,
            api_key=SecretStr(api_key or config.DOUBao_API_KEY),
            base_url=base_url or config.DOUBao_BASE_URL,
            temperature=0,
            streaming=False,
        )

        # 3. System Prompt
        self.base_system_content = self._load_system_prompt()
        print(f"[Agent] System Prompt 加载成功: {self.base_system_content[:60]}...")

        # 4. Checkpointer（短期记忆后端）
        self._checkpointer = InMemorySaver()

        # 5. 构建 Agent（自定义 state + tools + middleware + checkpointer）
        self.agent = create_agent(
            model=self.llm,
            tools=self._build_tools(),
            system_prompt=self.base_system_content,
            state_schema=YuYuanAgentState,
            middleware=self._build_middleware(),
            checkpointer=self._checkpointer,
        )

    # =========================================================================
    # Tool 构建（闭包捕获 self.rag / self.llm / self._yolo_model）
    # =========================================================================

    def _build_tools(self) -> list:
        rag = self.rag
        llm = self.llm
        yolo_model = self._yolo_model

        @tool
        def rag_search(query: str, runtime: ToolRuntime) -> str:
            """根据关键词检索豫园景点的历史文化知识。 适合回答"是什么""有什么历史""典故""形状"等知识性问题。 query: 用户问题或景点名称关键词。.
            """
            results = rag.retrieve_with_score(query, k=config.RERANK_TOP_K, rerank=config.RERANK)
            if not results:
                return "未找到相关景点资料。"
            return "\n".join([f"- {text}" for text, _ in results])

        @tool
        def yolo_detect(runtime: ToolRuntime) -> str:
            """对当前图像进行 YOLO 目标检测，快速识别图中豫园景点的类别和置信度。 返回文字描述，如"检测到：三穗堂(0.92)、荷花池(0.87)"。 无需传参，自动读取当前会话图像。.
            """
            if yolo_model is None:
                return "YOLO 检测服务不可用（未注入模型）。"
            image_base64 = runtime.state.get("current_image_base64", "")
            if not image_base64:
                return "当前会话没有可供检测的图像。"
            try:
                img_bytes = base64.b64decode(image_base64)
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    return "图像解码失败。"
                results = yolo_model(img, verbose=False)
                raw = []
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = yolo_model.names[cls]
                        coords = box.xyxy[0].cpu().numpy().tolist()
                        raw.append((name, conf, coords))
                if not raw:
                    return "未检测到已知景点目标。"
                keep = _apply_nms([d[2] for d in raw], [d[1] for d in raw])
                kept = [raw[i] for i in keep]
                return "检测到：" + "、".join([f"{name}({conf:.2f})" for name, conf, _ in kept])
            except Exception as e:
                return f"YOLO 检测出错: {e}"

        @tool
        def image_understand(question: str, runtime: ToolRuntime) -> str:
            """调用视觉模型理解当前图像，回答关于图像场景、建筑细节、风格的问题。 适合需要详细描述的情况。 question: 关于图像的具体问题。无需传入图像，自动读取当前会话图像。.
            """
            image_base64 = runtime.state.get("current_image_base64", "")
            if not image_base64:
                return "当前会话没有可供理解的图像。"
            try:
                human_content = [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ]
                response = llm.invoke([HumanMessage(content=human_content)])
                return response.content
            except Exception as e:
                return f"图像理解出错: {e}"

        return [rag_search, yolo_detect, image_understand]

    # =========================================================================
    # Middleware 构建（闭包捕获 self.agent_history_limit）
    # =========================================================================

    def _build_middleware(self) -> list:
        limit = self.agent_history_limit

        @before_model
        def trim_history(state: YuYuanAgentState, runtime: Runtime) -> dict[str, Any] | None:
            """模型调用前，限制历史消息数量，防止 Token 溢出。."""
            messages = state["messages"]
            if len(messages) <= limit:
                return None
            recent = messages[-limit:]
            return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent]}

        return [trim_history]

    # =========================================================================
    # System Prompt
    # =========================================================================

    def _load_system_prompt(self) -> str:
        if os.path.exists(config.SYSTEM_PROMPT_PATH):
            with open(config.SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
                return f.read()
        return "你是豫园的专业导游，请为听障人士提供简洁、具象的视觉导览。优先参考背景资料。"

    # =========================================================================
    # 会话管理
    # =========================================================================

    def clear_session_history(self, session_id: str):
        """清除指定会话的 checkpointer 记录。."""
        try:
            storage = self._checkpointer.storage
            keys_to_delete = [k for k in storage if k[0] == session_id]
            for k in keys_to_delete:
                del storage[k]
            print(f"[Session] 已清除会话历史: {session_id}（共 {len(keys_to_delete)} 条记录）")
        except Exception as e:
            print(f"[Session] 清除失败: {e}")

    def delete_session_history(self, session_id: str):
        self.clear_session_history(session_id)

    def print_history(self, session_id: str):
        try:
            state = self.agent.get_state({"configurable": {"thread_id": session_id}})
            messages = state.values.get("messages", []) if state else []
        except Exception:
            messages = []
        if not messages:
            print(f"[Session] 无历史记录: {session_id}")
            return
        print(f"[Session] 历史记录 ({session_id}):")
        for msg in messages:
            prefix = "AI" if isinstance(msg, AIMessage) else "User"
            preview = str(msg.content)[:60].replace("\n", " ")
            if len(str(msg.content)) > 60:
                preview += "..."
            print(f"  [{prefix}] {preview}")

    # =========================================================================
    # 图像辅助
    # =========================================================================

    def _image_file_to_base64(self, image_path: str, max_side: int = 1024, jpeg_quality: int = 85) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            rgb_img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _normalize_base64(self, image_base64: str) -> str:
        if not image_base64:
            return ""
        raw = image_base64.strip()
        if "," in raw and raw.startswith("data:"):
            return raw.split(",", 1)[1]
        return raw

    # =========================================================================
    # 对外接口
    # =========================================================================

    @staticmethod
    def _log_trace(messages: list):
        """打印本次调用的工具调用轨迹。."""
        print("[Trace] ┌" + "─" * 50)
        for msg in messages:
            if isinstance(msg, HumanMessage):
                preview = str(msg.content)[:80].replace("\n", " ")
                print(f"[Trace] │ 👤 User    : {preview}")
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        args = str(tc.get("args", {}))[:60]
                        print(f"[Trace] │ 🤖 调用工具: {tc['name']}({args})")
                else:
                    preview = str(msg.content)[:80].replace("\n", " ")
                    print(f"[Trace] │ 🤖 最终回答: {preview}")
            elif isinstance(msg, ToolMessage):
                preview = str(msg.content)[:80].replace("\n", " ")
                print(f"[Trace] │ 🔧 工具结果: [{msg.name}] {preview}")
        print("[Trace] └" + "─" * 50)

    async def generate_guidance(
        self,
        user_query: str,
        session_id: str = "unity_user_01",
    ) -> str:
        """文本导览接口（无图像）。."""
        try:
            prev = self.agent.get_state({"configurable": {"thread_id": session_id}})
            prev_count = len(prev.values.get("messages", [])) if prev and prev.values else 0

            result = await self.agent.ainvoke(
                {"messages": [HumanMessage(content=user_query)]},
                config={"configurable": {"thread_id": session_id}},
            )
            self._log_trace(result["messages"][prev_count:])
            return result["messages"][-1].content
        except Exception as e:
            print(f"[Agent Error] {e}")
            return "导览助手暂时无法连接，请稍后再试。"

    async def generate_guidance_with_image(
        self,
        user_query: str,
        session_id: str = "unity_user_01",
        image_path: str | None = None,
        image_base64: str | None = None,
    ) -> str:
        """图文导览接口。 图像存入 state（current_image_base64），LLM 初始轮仅收到文字， Agent 自行决定是否调用 yolo_detect 或 image_understand
        工具来处理图像。.
        """
        try:
            if not image_path and not image_base64:
                return "请提供图片路径或 base64 图片数据。"

            encoded = self._normalize_base64(image_base64) if image_base64 else self._image_file_to_base64(image_path)
            if not encoded:
                return "图片数据无效，请重新上传。"

            effective_query = user_query or "请结合当前图像给出导览说明。"
            print("[Agent] 导览请求 | query:", effective_query)
            prev = self.agent.get_state({"configurable": {"thread_id": session_id}})
            prev_count = len(prev.values.get("messages", [])) if prev and prev.values else 0

            start_time = time.perf_counter()
            result = await self.agent.ainvoke(
                {
                    "messages": [HumanMessage(content=effective_query)],
                    "current_image_base64": encoded,
                },
                config={"configurable": {"thread_id": session_id}},
            )
            print(f"[Agent] 导览完成，耗时 {time.perf_counter() - start_time:.2f} 秒")
            self._log_trace(result["messages"][prev_count:])
            return result["messages"][-1].content
        except Exception as e:
            print(f"[Agent Error] {e}")
            return "导览助手暂时无法连接，请稍后再试。"
