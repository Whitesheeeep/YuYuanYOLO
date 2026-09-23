from __future__ import annotations

import logging
import os
import time

from dotenv import load_dotenv
from flashrank import Ranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.vectorstores import FAISS

# 核心组件
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings  # 2026 推荐路径
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sympy import true

load_dotenv()

# 假设你的项目结构中有 config.py
from . import config

# try:
#     import config
# except ImportError:
#     print("config 加载出现问题")
#     # 兜底配置，防止独立运行时报错
#     class MockConfig:
#         FAISS_INDEX_PATH = "./data/faiss_index"
#         KNOWLEDGE_BASE_PATH = "./data/yuyuan_kb.txt"
#         EMBEDDING_MODEL = "BAAI/bge-m3"
#         EMBEDDING_DEVICE = "cuda"  # 或 "cpu"
#         CHUNK_SIZE = 300
#         CHUNK_OVERLAP = 50
#
#
#     config = MockConfig()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("YuYuanRAG")


class YuYuanRAG:
    """豫园知识库 RAG 检索引擎 - 2026 生产级实现."""

    def __init__(self, index_path: str | None = None):
        self.index_path = index_path or config.FAISS_INDEX_PATH
        self.vectorstore: FAISS | None = None
        self.embeddings: HuggingFaceEmbeddings | None = None

        # 1. 预初始化 Embedding (BGE-M3 加载较慢，建议只加载一次)
        self._init_embeddings()

        # 2. 尝试加载现有索引
        self._try_load_index()

        # 3. 初始化 Reranker
        cache_path = "./cache/ranker"
        os.makedirs(cache_path, exist_ok=True)
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=cache_path)
        self.base_compressor = FlashrankRerank(client=self.ranker)

        if self.vectorstore:
            self.base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
            self._init_compression_retriver()
        else:
            self.base_retriever = None
            self.compression_retriver = None
            logger.warning("[RAG] 未找到有效索引，请确保后续调用 build_from_text()")

    def _init_embeddings(self):
        """初始化嵌入模型，增加异常捕获."""
        if self.embeddings is None:
            try:
                logger.info(f"[RAG] 正在加载 Embedding 模型: {config.EMBEDDING_MODEL}...")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=config.EMBEDDING_MODEL,
                    model_kwargs={"device": config.EMBEDDING_DEVICE},
                    encode_kwargs={"normalize_embeddings": True},
                )
                logger.info("[RAG] Embedding 模型加载成功")
            except Exception as e:
                logger.error(f"[RAG] Embedding 加载失败: {e}")
                raise RuntimeError("无法启动 RAG 引擎：Embedding 模型缺失")

    def _init_compression_retriver(self):
        self.compression_retriver = ContextualCompressionRetriever(
            base_compressor=self.base_compressor,
            base_retriever=self.base_retriever,
        )

    def _try_load_index(self) -> bool:
        """安全加载 FAISS 索引."""
        # 检查关键文件 index.faiss 是否存在
        faiss_file = os.path.join(self.index_path, "index.faiss")
        if os.path.exists(faiss_file):
            try:
                # 2026年强制要求 allow_dangerous_deserialization
                self.vectorstore = FAISS.load_local(
                    self.index_path, self.embeddings, allow_dangerous_deserialization=True
                )
                logger.info(f"[RAG] 成功载入本地索引: {self.index_path}")
                return True
            except Exception as e:
                logger.error(f"[RAG] 索引文件损坏或版本不匹配: {e}")
        return False

    # 构建 向量库
    def build_from_text(self, text_file: str):
        """从纯文本构建知识库（含语义切分逻辑）."""
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"知识库源文件不存在: {text_file}")

        logger.info(f"[RAG] 正在从 {text_file} 构建索引...")

        with open(text_file, encoding="utf-8") as f:
            raw_text = f.read()

        # 语义切分
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )

        # 封装为 Document 对象并切分
        initial_doc = Document(page_content=raw_text, metadata={"source": os.path.basename(text_file)})
        documents = splitter.split_documents([initial_doc])

        # 构建向量库
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

        # 初始化 retriever
        self.base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        self._init_compression_retriver()

        # 持久化
        os.makedirs(self.index_path, exist_ok=True)
        self.vectorstore.save_local(self.index_path)
        logger.info(f"[RAG] 索引构建完成并保存至: {self.index_path} (共 {len(documents)} 块)")

    def retrieve(self, query: str, k: int = 3, rerank: bool = True) -> list[str]:
        print(f"[RAG] 收到查询: '{query}'，参数 - k: {k}, rerank: {rerank}")
        if not rerank:
            """执行相似度检索"""
            if not self.is_ready():
                return ["知识库未就绪。"]

            try:
                # 2026 推荐使用 similarity_search 保持简单语义对齐
                # 使用的是 from_Documents，默认为 IndexFlatL2 索引（欧氏距离），距离越小越相似
                docs = self.vectorstore.similarity_search(query, k=k)
                print(f"[RAG] 基础检索返回 {len(docs)} 条结果: {docs[0].page_content}")
                return [doc.page_content for doc in docs]
            except Exception as e:
                logger.error(f"[RAG] 检索出错: {e}")
                return []
        else:
            try:
                compressed_docs = self.compression_retriver.invoke(query)
                print(f"[RAG] 重排序后返回 {len(compressed_docs)} 条结果: {compressed_docs[0].page_content}")
                return [doc.page_content for doc in compressed_docs[:k]]

            except Exception as e:
                logger.error(f"[RAG] 检索或重排序出错: {e}")
                # 降级处理：如果 Reranker 出错，退回到基础的 FAISS 检索
                docs = self.vectorstore.similarity_search(query, k=k)
                return [doc.page_content for doc in docs]

    def retrieve_with_score(
        self, query: str, k: int = 3, threshold: float | None = None, rerank: bool = true
    ) -> list[tuple[str, float]]:
        if not self.is_ready():
            return []

        start_time = time.perf_counter()
        threshold = threshold if threshold is not None else config.RAG_SCORE_THRESHOLD
        if not rerank:
            """带分数过滤的精细检索"""
            try:
                # 返回的是 (Document, score)
                # FAISS 默认 L2 距离，score 越小越相似
                docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)

                results = []
                for doc, score in docs_and_scores:
                    # 距离转相似度的经验公式
                    similarity = 1 / (1 + score)
                    print(score, similarity)
                    if similarity >= threshold:
                        results.append((doc.page_content, float(similarity)))
                return results
            except Exception as e:
                logger.error(f"[RAG] 分数检索出错: {e}")
                return []
        else:
            try:
                compressed_docs = self.compression_retriver.invoke(query)
                print(compressed_docs)
                print(
                    f"[RAG] 重排序后返回 {len(compressed_docs)} 条结果，第 0 条: {compressed_docs[0].page_content[:100]}..."
                )
                # 筛选分数过低的
                results = []
                for doc in compressed_docs:
                    score = doc.metadata.get("relevance_score", 0.0)
                    print(score)
                    if score >= threshold:
                        results.append((doc.page_content, float(score)))
                return results[:k]  # 2026年版本调整：重排序结果默认相似度为 1.0，实际应用中可以根据需要调整为其他值
            except Exception as e:
                logger.error(f"[RAG] 检索或重排序出错: {e}")
                # 降级处理：如果 Reranker 出错，退回到基础的 FAISS 检索
                docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
                results = []
                for doc, score in docs_and_scores:
                    similarity = 1 / (1 + score)
                    if similarity >= threshold:
                        results.append((doc.page_content, float(similarity)))

                print(f"[RAG] 检索耗时：{time.perf_counter() - start_time:.2f} 秒，返回 {len(results)} 条结果")
                return results

    def is_ready(self) -> bool:
        """检查引擎是否可用."""
        return self.vectorstore is not None and self.embeddings is not None
