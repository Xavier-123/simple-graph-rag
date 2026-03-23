"""
Graph-RAG 最小可行性原型（MVP）

依赖库安装命令：
    pip install networkx lancedb sentence-transformers numpy pyarrow pandas

运行方式：
    python graph_rag.py

Embedding 策略说明：
    - 默认使用无需下载的字符 n-gram TF-IDF 向量，可在本地离线运行。
    - 若希望获得更高质量的语义检索，可将 USE_SENTENCE_TRANSFORMERS 改为 True，
      并确保网络可以访问 HuggingFace（首次运行会自动下载模型）：
          USE_SENTENCE_TRANSFORMERS = True
          SENTENCE_TRANSFORMER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
"""

import warnings
warnings.filterwarnings("ignore")

import os
import math
import shutil
import numpy as np
import networkx as nx
import lancedb
import pyarrow as pa
from collections import Counter
from typing import Callable

# ── 是否使用 sentence-transformers（需联网下载模型）──────────────────────────
USE_SENTENCE_TRANSFORMERS = False
SENTENCE_TRANSFORMER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ─────────────────────────────────────────────
# 1. 知识三元组定义
#    格式：(主体, 关系, 客体)
# ─────────────────────────────────────────────
TRIPLES = [
    ("乔布斯",   "创立",   "苹果公司"),
    ("乔布斯",   "创立",   "皮克斯"),
    ("乔布斯",   "毕业于", "里德学院"),
    ("苹果公司", "总部位于", "库比蒂诺"),
    ("苹果公司", "生产",   "iPhone"),
    ("苹果公司", "生产",   "Mac"),
    ("比尔盖茨", "创立",   "微软"),
    ("比尔盖茨", "就读于", "哈佛大学"),
    ("微软",     "总部位于", "雷德蒙德"),
    ("微软",     "开发",   "Windows"),
    ("微软",     "开发",   "Office"),
    ("马斯克",   "创立",   "特斯拉"),
    ("马斯克",   "创立",   "SpaceX"),
    ("特斯拉",   "生产",   "电动汽车"),
    ("SpaceX",   "开发",   "猎鹰火箭"),
    ("皮克斯",   "制作",   "玩具总动员"),
    ("iPhone",   "运行",   "iOS"),
    ("Windows",  "开发者", "微软"),
]

# ─────────────────────────────────────────────
# 2. 构建 NetworkX 有向图
# ─────────────────────────────────────────────
def build_graph(triples: list[tuple[str, str, str]]) -> nx.DiGraph:
    """根据知识三元组构建有向图，边的 label 属性存储关系。"""
    G = nx.DiGraph()
    for subj, rel, obj in triples:
        G.add_node(subj)
        G.add_node(obj)
        G.add_edge(subj, obj, label=rel)
    return G


# ─────────────────────────────────────────────
# 3-A. 字符 n-gram TF-IDF 嵌入（离线，无需下载）
#      适合中文短文本的轻量级语义检索
# ─────────────────────────────────────────────

def _ngrams(text: str, n: int = 2) -> list[str]:
    """从文本中提取所有 n-gram（包含单字和双字，兼顾中英文）。"""
    chars = list(text)
    grams: list[str] = list(chars)  # 单字（unigram）
    for i in range(len(chars) - n + 1):
        grams.append("".join(chars[i: i + n]))
    return grams


class NgramTfidfEmbedder:
    """
    基于字符 n-gram 的 TF-IDF 向量化器。
    先用语料库（节点名称）拟合 IDF，再对任意文本编码为稠密向量。
    """

    def __init__(self, n: int = 2):
        self.n = n
        self.vocab: dict[str, int] = {}   # gram -> 维度索引
        self.idf: np.ndarray | None = None

    def fit(self, corpus: list[str]) -> "NgramTfidfEmbedder":
        """用语料库构建词汇表并计算 IDF 权重。"""
        # 统计每个 gram 出现在多少文档中
        doc_freq: Counter = Counter()
        tokenized = [_ngrams(text, self.n) for text in corpus]
        for grams in tokenized:
            doc_freq.update(set(grams))

        # 按频率降序构建词汇表，保留至少出现 1 次的 gram
        for gram in sorted(doc_freq, key=lambda g: -doc_freq[g]):
            self.vocab[gram] = len(self.vocab)

        num_docs = len(corpus)
        idf_values = np.zeros(len(self.vocab), dtype=np.float32)
        for gram, idx in self.vocab.items():
            idf_values[idx] = math.log((num_docs + 1) / (doc_freq[gram] + 1)) + 1.0
        self.idf = idf_values
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        """将文本列表编码为 L2 归一化的 TF-IDF 向量矩阵。"""
        if self.idf is None:
            raise RuntimeError("请先调用 fit() 拟合语料库。")
        dim = len(self.vocab)
        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            grams = _ngrams(text, self.n)
            tf: Counter = Counter(grams)
            for gram, count in tf.items():
                if gram in self.vocab:
                    idx = self.vocab[gram]
                    matrix[i, idx] = count * self.idf[idx]
            # L2 归一化，使余弦相似度等价于点积
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm
        return matrix


# ─────────────────────────────────────────────
# 3-B. sentence-transformers 嵌入（需联网/已缓存模型）
# ─────────────────────────────────────────────

def _load_sentence_transformer(model_name: str) -> "SentenceTransformer":
    """加载 sentence-transformers 多语言嵌入模型（CPU 友好）。"""
    from sentence_transformers import SentenceTransformer  # type: ignore
    print(f"[信息] 加载 sentence-transformers 模型：{model_name} …")
    return SentenceTransformer(model_name, device="cpu")


# ─────────────────────────────────────────────
# 3-C. 统一 Embedder 工厂
# ─────────────────────────────────────────────

def build_embedder(
    corpus: list[str],
) -> tuple[Callable[[list[str]], np.ndarray], int]:
    """
    根据 USE_SENTENCE_TRANSFORMERS 标志选择嵌入方案。
    返回 (encode_fn, embedding_dim) 元组：
        - encode_fn(texts: list[str]) -> np.ndarray  shape=(N, dim)
        - embedding_dim: int
    """
    if USE_SENTENCE_TRANSFORMERS:
        st_model = _load_sentence_transformer(SENTENCE_TRANSFORMER_MODEL)
        sample = st_model.encode(["test"], convert_to_numpy=True, show_progress_bar=False)
        dim = sample.shape[1]

        def encode_fn(texts: list[str]) -> np.ndarray:
            vecs = st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            # L2 归一化
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return (vecs / norms).astype(np.float32)

        print(f"[信息] 使用 sentence-transformers 嵌入，维度={dim}")
    else:
        # 用语料库拟合 TF-IDF，再用全部文本确定维度
        embedder = NgramTfidfEmbedder(n=2)
        embedder.fit(corpus)
        dim = len(embedder.vocab)

        def encode_fn(texts: list[str]) -> np.ndarray:
            return embedder.encode(texts)

        print(f"[信息] 使用字符 n-gram TF-IDF 嵌入（离线），维度={dim}")

    return encode_fn, dim


# ─────────────────────────────────────────────
# 4. 初始化 LanceDB 并写入节点向量
# ─────────────────────────────────────────────
def build_lancedb_index(
    G: nx.DiGraph,
    encode_fn: Callable[[list[str]], np.ndarray],
    dim: int,
    db_path: str = "/tmp/graph_rag_lancedb",
    table_name: str = "nodes",
) -> lancedb.table.Table:
    """
    将图中所有节点名称编码为向量，存入 LanceDB。
    若数据库已存在则先删除，保证每次运行结果一致。
    """
    # 清理旧数据，确保索引与当前图保持一致
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    db = lancedb.connect(db_path)
    nodes = list(G.nodes())

    print(f"[信息] 为 {len(nodes)} 个节点生成嵌入向量 …")
    embeddings = encode_fn(nodes)  # shape: (N, dim)

    # 构造 PyArrow 表：node_name（文本）+ vector（浮点数组）
    schema = pa.schema([
        pa.field("node_name", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])

    records = [
        {"node_name": node, "vector": emb.tolist()}
        for node, emb in zip(nodes, embeddings)
    ]

    table = db.create_table(table_name, data=records, schema=schema)
    print(f"[信息] LanceDB 索引构建完成，表名：{table_name}，维度={dim}")
    return table


# ─────────────────────────────────────────────
# 5. 向量检索：找到语义最接近的节点
# ─────────────────────────────────────────────
def vector_search(
    query: str,
    encode_fn: Callable[[list[str]], np.ndarray],
    table: lancedb.table.Table,
    top_k: int = 3,
) -> list[str]:
    """
    将问题编码为向量，在 LanceDB 中做近似最近邻检索，
    返回 top_k 个最相关节点的名称列表。
    """
    query_vec = encode_fn([query])[0].tolist()
    results = table.search(query_vec).limit(top_k).to_pandas()
    matched_nodes = results["node_name"].tolist()
    return matched_nodes


# ─────────────────────────────────────────────
# 6. 图增强：利用 G.neighbors() 提取关联信息
# ─────────────────────────────────────────────
def graph_context(G: nx.DiGraph, nodes: list[str]) -> list[str]:
    """
    对每个检索到的节点，提取其所有出边邻居，
    组装成"主体 -> 关系 -> 客体"格式的字符串列表。
    同时检查入边，以便覆盖"客体"角色的节点。
    """
    facts: list[str] = []
    visited_edges: set[tuple[str, str]] = set()

    for node in nodes:
        # 出边：node 作为主体
        for neighbor in G.neighbors(node):
            edge_key = (node, neighbor)
            if edge_key not in visited_edges:
                rel = G[node][neighbor].get("label", "关联")
                facts.append(f"{node} --[{rel}]--> {neighbor}")
                visited_edges.add(edge_key)

        # 入边：node 作为客体（查找指向 node 的边）
        for predecessor in G.predecessors(node):
            edge_key = (predecessor, node)
            if edge_key not in visited_edges:
                rel = G[predecessor][node].get("label", "关联")
                facts.append(f"{predecessor} --[{rel}]--> {node}")
                visited_edges.add(edge_key)

    return facts


# ─────────────────────────────────────────────
# 7. 合成答案：拼装 RAG 风格的 Prompt 并输出
# ─────────────────────────────────────────────
def synthesize_answer(query: str, matched_nodes: list[str], facts: list[str]) -> str:
    """
    将向量检索结果和图谱关联信息合并，
    按 RAG 的 Prompt 格式输出（此处直接打印，无需调用外部 LLM）。
    """
    nodes_str = "、".join(matched_nodes) if matched_nodes else "（无）"
    facts_str = "\n  ".join(facts) if facts else "  （未检索到相关三元组）"

    prompt = f"""
╔══════════════════════════════════════════════════════╗
║              Graph-RAG  问答  Prompt                 ║
╚══════════════════════════════════════════════════════╝

【用户问题】
  {query}

【向量检索结果】（语义最近节点）
  {nodes_str}

【图谱关联事实】（来自 NetworkX 邻居扩展）
  {facts_str}

【参考答案（基于以上上下文）】
  根据知识图谱，与"{query}"最相关的节点为：{nodes_str}。
  相关联的知识如下：
  {facts_str}
  （在真实 RAG 系统中，以上内容将作为上下文传给大语言模型生成自然语言回答。）
"""
    return prompt.strip()


# ─────────────────────────────────────────────
# 8. 主流程
# ─────────────────────────────────────────────
def main():
    # 步骤 1：构建知识图谱
    print("\n[步骤 1] 构建 NetworkX 知识图谱 …")
    G = build_graph(TRIPLES)
    print(f"  节点数：{G.number_of_nodes()}，边数：{G.number_of_edges()}")

    # 步骤 2：构建嵌入器（优先使用语料库节点做词汇表拟合）
    print("\n[步骤 2] 初始化嵌入器 …")
    nodes = list(G.nodes())
    encode_fn, dim = build_embedder(corpus=nodes)

    # 步骤 3：构建 LanceDB 向量索引
    print("\n[步骤 3] 构建 LanceDB 向量索引 …")
    table = build_lancedb_index(G, encode_fn, dim)

    # 步骤 4：示例查询演示
    print("\n[步骤 4] 示例查询演示\n")
    sample_queries = [
        "乔布斯做过什么",
        "苹果公司有哪些产品",
        "比尔盖茨创建了什么公司",
        "马斯克的企业",
    ]

    for query in sample_queries:
        print(f"{'─' * 56}")
        matched_nodes = vector_search(query, encode_fn, table, top_k=3)
        facts = graph_context(G, matched_nodes)
        answer = synthesize_answer(query, matched_nodes, facts)
        print(answer)
        print()

    # 步骤 5：交互式输入
    print(f"{'─' * 56}")
    while True:
        try:
            user_input = input("\n请输入您的问题（输入 'exit' 退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[信息] 已退出。")
            break
        if not user_input or user_input.lower() == "exit":
            print("[信息] 已退出。")
            break
        matched_nodes = vector_search(user_input, encode_fn, table, top_k=3)
        facts = graph_context(G, matched_nodes)
        answer = synthesize_answer(user_input, matched_nodes, facts)
        print(answer)


if __name__ == "__main__":
    main()
