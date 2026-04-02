# ============================================================
# 豫园 XR 听障辅助系统 - 配置文件
# ============================================================

# --- DouBao API (火山引擎) 配置 ---
# 请替换为你的实际 API Key
DOUBao_API_KEY = "98af0e67-fd36-4973-86f4-69a85425ee06"
DOUBao_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
# DOUBao_MODEL = "doubao-seed-2-0-mini-260215"
# 备用模型：
DOUBao_MODEL = "doubao-seed-2-0-lite-260215"
# DOUBao_MODEL = "doubao-seed-2-0-pro-260215"

#Agent 配置
RERANK = True #默认需要重排
RERANK_TOP_K = 3 #重排时保留的 top k 个文档
SYSTEM_PROMPT_PATH = r"D:\Yuyuan_2\YuyuanYOLO\LangChain\data\system_prompt.txt"

# --- RAG 配置 ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cpu"  # 开发环境用 cpu，生产环境可改为 "cuda"
FAISS_INDEX_PATH = r"D:\Yuyuan_2\YuyuanYOLO\LangChain\data\faiss_index"
KNOWLEDGE_BASE_PATH = r"D:\Yuyuan_2\YuyuanYOLO\LangChain\data\knowledge_base.txt"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
RAG_SCORE_THRESHOLD = 0.7

# --- 听障优化配置 ---
CONFIDENCE_THRESHOLD = 0.85   # 自动推送讲解的最低置信度阈值
MAX_TEXT_LENGTH = 50         # display_text 单句最大字数（字符）
AUTO_HINT_ENABLED = True     # 是否开启自动视觉注释

# --- LLM 生成参数 ---
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 300
LLM_TIMEOUT = 30              # LLM 调用超时（秒）