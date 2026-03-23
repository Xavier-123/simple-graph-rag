"""
方案一：元数据关联（Metadata Mapping）

核心思路：
    在构建图谱阶段，将文档切片的唯一标识符（ChunkID）作为属性存入图节点和关系中。
    检索时通过图谱命中节点/关系上的 source_chunks，回扫原始文档切片，
    最终将"图结构化描述"与"原始切片文本"合并后送入 LLM。

检索流程：
    1. 图检索：根据查询提取实体，在图中搜索相关三元组。
    2. ID 提取：获取命中节点/关系上的 source_chunks ID 列表。
    3. 切片回扫：根据 ID 从切片存储中取回原始文本。
    4. 上下文合并：将图结构描述与原始切片文本一起喂给 LLM。

运行方式：
    python metadata_mapping.py
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

# ── 配置 ────────────────────────────────────────────────
USE_SENTENCE_TRANSFORMERS = False
SENTENCE_TRANSFORMER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ─────────────────────────────────────────────
# 1. 模拟文档切片（每个切片有唯一 ID 和文本内容）
# ─────────────────────────────────────────────
CHUNKS = {
    "chunk_001": "乔布斯（Steve Jobs）于 1976 年与沃兹尼亚克共同创立了苹果公司，后来又创立了皮克斯动画工作室。",
    "chunk_002": "乔布斯年轻时曾就读于里德学院，但仅旁听了一个学期便退学。",
    "chunk_003": "苹果公司总部位于加州库比蒂诺，是全球最具价值的科技企业之一。",
    "chunk_004": "苹果公司的主要产品包括 iPhone 智能手机和 Mac 系列电脑，iPhone 运行 iOS 操作系统。",
    "chunk_005": "比尔·盖茨于 1975 年创立了微软公司，他曾就读于哈佛大学但中途辍学。",
    "chunk_006": "微软总部位于华盛顿州雷德蒙德，主要产品有 Windows 操作系统和 Office 办公套件。",
    "chunk_007": "埃隆·马斯克创立了特斯拉和 SpaceX 两家公司。特斯拉主要生产电动汽车。",
    "chunk_008": "SpaceX 开发了猎鹰系列运载火箭，致力于降低太空运输成本。",
    "chunk_009": "皮克斯动画工作室制作了《玩具总动员》等经典动画电影。",
}

# ─────────────────────────────────────────────
# 2. 带元数据的知识三元组
#    每个三元组附带 source_chunks 列表，记录该事实来源
# ─────────────────────────────────────────────
TRIPLES_WITH_META = [
    # (主体, 关系, 客体, [来源切片ID列表])
    ("乔布斯",   "创立",   "苹果公司",   ["chunk_001"]),
    ("乔布斯",   "创立",   "皮克斯",     ["chunk_001"]),
    ("乔布斯",   "毕业于", "里德学院",   ["chunk_002"]),
    ("苹果公司", "总部位于", "库比蒂诺", ["chunk_003"]),
    ("苹果公司", "生产",   "iPhone",     ["chunk_004"]),
    ("苹果公司", "生产",   "Mac",        ["chunk_004"]),
    ("iPhone",   "运行",   "iOS",        ["chunk_004"]),
    ("比尔盖茨", "创立",   "微软",       ["chunk_005"]),
    ("比尔盖茨", "就读于", "哈佛大学",   ["chunk_005"]),
    ("微软",     "总部位于", "雷德蒙德", ["chunk_006"]),
    ("微软",     "开发",   "Windows",    ["chunk_006"]),
    ("微软",     "开发",   "Office",     ["chunk_006"]),
    ("马斯克",   "创立",   "特斯拉",     ["chunk_007"]),
    ("马斯克",   "创立",   "SpaceX",     ["chunk_007"]),
    ("特斯拉",   "生产",   "电动汽车",   ["chunk_007"]),
    ("SpaceX",   "开发",   "猎鹰火箭",   ["chunk_008"]),
    ("皮克斯",   "制作",   "玩具总动员", ["chunk_009"]),
]

# ─────────────────────────────────────────────
# 3. 构建带元数据的知识图谱
# ─────────────────────────────────────────────
def build_graph_with_metadata(
    triples: list[tuple[str, str, str, list[str]]],
) -> nx.DiGraph:
    """
    构建有向图，节点和边均携带 source_chunks 元数据。
    - 节点属性 source_chunks：该实体出现在哪些切片中。
    - 边属性 source_chunks：该关系三元组来源于哪些切片。
    """
    G = nx.DiGraph()
    for subj, rel, obj, chunk_ids in triples:
        # 更新主体节点的 source_chunks
        if subj not in G:
            G.add_node(subj, source_chunks=[])
        G.nodes[subj]["source_chunks"] = list(
            set(G.nodes[subj]["source_chunks"]) | set(chunk_ids)
        )

        # 更新客体节点的 source_chunks
        if obj not in G:
            G.add_node(obj, source_chunks=[])
        G.nodes[obj]["source_chunks"] = list(
            set(G.nodes[obj]["source_chunks"]) | set(chunk_ids)
        )

        # 添加边，附带关系标签和来源切片
        G.add_edge(subj, obj, label=rel, source_chunks=chunk_ids)

    return G


# ─────────────────────────────────────────────
# 4. 嵌入器（与 graph_rag.py 相同的 n-gram TF-IDF）
# ─────────────────────────────────────────────
def _ngrams(text: str, n: int = 2) -> list[str]:
    chars = list(text)
    grams: list[str] = list(chars)
    for i in range(len(chars) - n + 1):
        grams.append("".join(chars[i: i + n]))
    return grams


class NgramTfidfEmbedder:
    def __init__(self, n: int = 2):
        self.n = n
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, corpus: list[str]) -> "NgramTfidfEmbedder":
        doc_freq: Counter = Counter()
        tokenized = [_ngrams(text, self.n) for text in corpus]
        for grams in tokenized:
            doc_freq.update(set(grams))
        for gram in sorted(doc_freq, key=lambda g: -doc_freq[g]):
            self.vocab[gram] = len(self.vocab)
        num_docs = len(corpus)
        idf_values = np.zeros(len(self.vocab), dtype=np.float32)
        for gram, idx in self.vocab.items():
            idf_values[idx] = math.log((num_docs + 1) / (doc_freq[gram] + 1)) + 1.0
        self.idf = idf_values
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
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
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm
        return matrix


def build_embedder(
    corpus: list[str],
) -> tuple[Callable[[list[str]], np.ndarray], int]:
    if USE_SENTENCE_TRANSFORMERS:
        from sentence_transformers import SentenceTransformer
        print(f"[信息] 加载 sentence-transformers 模型：{SENTENCE_TRANSFORMER_MODEL} …")
        st_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device="cpu")
        sample = st_model.encode(["test"], convert_to_numpy=True, show_progress_bar=False)
        dim = sample.shape[1]

        def encode_fn(texts: list[str]) -> np.ndarray:
            vecs = st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return (vecs / norms).astype(np.float32)

        print(f"[信息] 使用 sentence-transformers 嵌入，维度={dim}")
    else:
        embedder = NgramTfidfEmbedder(n=2)
        embedder.fit(corpus)
        dim = len(embedder.vocab)

        def encode_fn(texts: list[str]) -> np.ndarray:
            return embedder.encode(texts)

        print(f"[信息] 使用字符 n-gram TF-IDF 嵌入（离线），维度={dim}")

    return encode_fn, dim


# ─────────────────────────────────────────────
# 5. 构建 LanceDB 向量索引（以节点名称为索引对象）
# ─────────────────────────────────────────────
def build_lancedb_index(
    G: nx.DiGraph,
    encode_fn: Callable[[list[str]], np.ndarray],
    dim: int,
    db_path: str = "/tmp/metadata_mapping_lancedb",
    table_name: str = "nodes",
) -> lancedb.table.Table:
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    db = lancedb.connect(db_path)
    nodes = list(G.nodes())

    print(f"[信息] 为 {len(nodes)} 个节点生成嵌入向量 …")
    embeddings = encode_fn(nodes)

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
# 6. 向量检索
# ─────────────────────────────────────────────
def vector_search(
    query: str,
    encode_fn: Callable[[list[str]], np.ndarray],
    table: lancedb.table.Table,
    top_k: int = 3,
) -> list[str]:
    query_vec = encode_fn([query])[0].tolist()
    results = table.search(query_vec).limit(top_k).to_pandas()
    return results["node_name"].tolist()


# ─────────────────────────────────────────────
# 7. 图检索 + 元数据 ID 提取
# ─────────────────────────────────────────────
def graph_search_with_metadata(
    G: nx.DiGraph, nodes: list[str],
) -> tuple[list[str], list[str]]:
    """
    对命中节点进行图遍历，提取关联事实和来源切片 ID。

    返回:
        facts: 结构化的三元组描述列表
        chunk_ids: 去重后的关联切片 ID 列表
    """
    facts: list[str] = []
    chunk_id_set: set[str] = set()
    visited_edges: set[tuple[str, str]] = set()

    for node in nodes:
        # 收集节点自身的 source_chunks
        node_chunks = G.nodes[node].get("source_chunks", [])
        chunk_id_set.update(node_chunks)

        # 出边
        for neighbor in G.neighbors(node):
            edge_key = (node, neighbor)
            if edge_key not in visited_edges:
                edge_data = G[node][neighbor]
                rel = edge_data.get("label", "关联")
                facts.append(f"{node} --[{rel}]--> {neighbor}")
                visited_edges.add(edge_key)
                # 收集边的 source_chunks
                chunk_id_set.update(edge_data.get("source_chunks", []))

        # 入边
        for predecessor in G.predecessors(node):
            edge_key = (predecessor, node)
            if edge_key not in visited_edges:
                edge_data = G[predecessor][node]
                rel = edge_data.get("label", "关联")
                facts.append(f"{predecessor} --[{rel}]--> {node}")
                visited_edges.add(edge_key)
                chunk_id_set.update(edge_data.get("source_chunks", []))

    return facts, sorted(chunk_id_set)


# ─────────────────────────────────────────────
# 8. 切片回扫：根据 ID 取回原始文本
# ─────────────────────────────────────────────
def retrieve_chunks(chunk_ids: list[str], chunk_store: dict[str, str]) -> list[str]:
    """根据切片 ID 列表，从切片存储中取回原始文本。"""
    texts = []
    for cid in chunk_ids:
        if cid in chunk_store:
            texts.append(f"[{cid}] {chunk_store[cid]}")
    return texts


# ─────────────────────────────────────────────
# 9. 上下文合并与答案合成
# ─────────────────────────────────────────────
def synthesize_answer(
    query: str,
    matched_nodes: list[str],
    facts: list[str],
    chunk_texts: list[str],
) -> str:
    """
    将图结构描述与原始切片文本合并为 RAG Prompt。
    """
    nodes_str = "、".join(matched_nodes) if matched_nodes else "（无）"
    facts_str = "\n  ".join(facts) if facts else "（未检索到相关三元组）"
    chunks_str = "\n  ".join(chunk_texts) if chunk_texts else "（未找到关联切片）"

    prompt = f"""
╔══════════════════════════════════════════════════════════╗
║        元数据关联（Metadata Mapping）Graph-RAG           ║
╚══════════════════════════════════════════════════════════╝

【用户问题】
  {query}

【向量检索命中节点】
  {nodes_str}

【图谱关联事实】（结构化三元组）
  {facts_str}

【原始切片回扫】（通过 source_chunks ID 取回）
  {chunks_str}

【参考答案】
  根据知识图谱，与"{query}"最相关的节点为：{nodes_str}。
  图谱结构化知识：
  {facts_str}
  关联的原始文档内容：
  {chunks_str}
  （在真实系统中，以上结构化知识与原始切片将一并作为上下文送入 LLM 生成回答。）
"""
    return prompt.strip()


# ─────────────────────────────────────────────
# 10. 主流程
# ─────────────────────────────────────────────
def main():
    # 步骤 1：构建带元数据的知识图谱
    print("\n[步骤 1] 构建带元数据的知识图谱 …")
    G = build_graph_with_metadata(TRIPLES_WITH_META)
    print(f"  节点数：{G.number_of_nodes()}，边数：{G.number_of_edges()}")

    # 展示节点的元数据示例
    print("\n  节点元数据示例：")
    for node in list(G.nodes())[:3]:
        chunks = G.nodes[node].get("source_chunks", [])
        print(f"    {node}: source_chunks = {chunks}")

    # 步骤 2：构建嵌入器
    print("\n[步骤 2] 初始化嵌入器 …")
    nodes = list(G.nodes())
    encode_fn, dim = build_embedder(corpus=nodes)

    # 步骤 3：构建 LanceDB 向量索引
    print("\n[步骤 3] 构建 LanceDB 向量索引 …")
    table = build_lancedb_index(G, encode_fn, dim)

    # 步骤 4：示例查询
    print("\n[步骤 4] 示例查询演示\n")
    sample_queries = [
        "乔布斯做过什么",
        "苹果公司有哪些产品",
        "比尔盖茨创建了什么公司",
        "马斯克的企业",
    ]

    for query in sample_queries:
        print(f"{'─' * 60}")
        # 6. 向量检索命中节点
        matched_nodes = vector_search(query, encode_fn, table, top_k=3)
        # 7. 图检索 + 元数据 ID 提取
        facts, chunk_ids = graph_search_with_metadata(G, matched_nodes)
        # 8. 切片回扫
        chunk_texts = retrieve_chunks(chunk_ids, CHUNKS)
        # 9. 合成答案
        answer = synthesize_answer(query, matched_nodes, facts, chunk_texts)
        print(answer)
        print()

    # 步骤 5：交互式输入
    print(f"{'─' * 60}")
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
        facts, chunk_ids = graph_search_with_metadata(G, matched_nodes)
        chunk_texts = retrieve_chunks(chunk_ids, CHUNKS)
        answer = synthesize_answer(user_input, matched_nodes, facts, chunk_texts)
        print(answer)


if __name__ == "__main__":
    main()
