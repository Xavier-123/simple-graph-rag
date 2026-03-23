"""
方案三：社区摘要（Community Summaries）

核心思路：
    不直接对应单个切片，而是对应"切片簇"。
    1. 聚类：图构建完成后，利用图算法对节点进行聚类，形成不同的"社区"。
    2. 生成摘要：为每个社区撰写摘要，并记录该社区涵盖了哪些原始切片。
    3. 检索：当用户提问时，系统检索相关的社区摘要。
    4. 溯源：摘要中附带来源引用，直接指向底层文档切片。

    适合处理跨文档的复杂全局问题，解决"碎片化"问题。

运行方式：
    python community_summaries.py
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
from collections import Counter, defaultdict
from typing import Callable

# ── 配置 ────────────────────────────────────────────────
USE_SENTENCE_TRANSFORMERS = False
SENTENCE_TRANSFORMER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ─────────────────────────────────────────────
# 1. 文档切片
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
# 2. 知识三元组（带来源切片标注）
# ─────────────────────────────────────────────
TRIPLES_WITH_META = [
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
# 3. 构建知识图谱
# ─────────────────────────────────────────────
def build_graph(
    triples: list[tuple[str, str, str, list[str]]],
) -> nx.DiGraph:
    G = nx.DiGraph()
    for subj, rel, obj, chunk_ids in triples:
        if subj not in G:
            G.add_node(subj, source_chunks=[])
        G.nodes[subj]["source_chunks"] = list(
            set(G.nodes[subj]["source_chunks"]) | set(chunk_ids)
        )
        if obj not in G:
            G.add_node(obj, source_chunks=[])
        G.nodes[obj]["source_chunks"] = list(
            set(G.nodes[obj]["source_chunks"]) | set(chunk_ids)
        )
        G.add_edge(subj, obj, label=rel, source_chunks=chunk_ids)
    return G


# ─────────────────────────────────────────────
# 4. 社区检测（使用 Louvain 算法）
#    NetworkX 内置支持，无需额外依赖
# ─────────────────────────────────────────────
def detect_communities(G: nx.DiGraph) -> dict[int, list[str]]:
    """
    使用 Louvain 社区检测算法对图节点进行聚类。
    由于 Louvain 算法作用于无向图，先将有向图转为无向图。

    返回:
        communities: {社区ID: [节点名称列表]}
    """
    G_undirected = G.to_undirected()
    # networkx.community.louvain_communities 返回节点集合的列表
    community_sets = nx.community.louvain_communities(
        G_undirected, resolution=1.0, seed=42
    )

    communities: dict[int, list[str]] = {}
    for idx, node_set in enumerate(community_sets):
        communities[idx] = sorted(node_set)

    return communities


# ─────────────────────────────────────────────
# 5. 为每个社区生成摘要
# ─────────────────────────────────────────────
def generate_community_summaries(
    G: nx.DiGraph,
    communities: dict[int, list[str]],
    chunk_store: dict[str, str],
) -> list[dict]:
    """
    为每个社区生成摘要，包含：
    - 社区 ID
    - 包含的实体列表
    - 社区内的三元组事实
    - 关联的原始切片 ID（Source Citations）
    - 摘要文本

    在真实系统中，摘要文本应由 LLM 生成；此处采用模板自动生成。
    """
    summaries: list[dict] = []

    for comm_id, nodes in communities.items():
        # 收集社区内的三元组事实
        facts: list[str] = []
        source_chunk_ids: set[str] = set()

        for node in nodes:
            # 收集节点的 source_chunks
            node_chunks = G.nodes[node].get("source_chunks", [])
            source_chunk_ids.update(node_chunks)

            # 收集出边（仅社区内部 + 社区到外部的边）
            for neighbor in G.neighbors(node):
                edge_data = G[node][neighbor]
                rel = edge_data.get("label", "关联")
                facts.append(f"{node} --[{rel}]--> {neighbor}")
                source_chunk_ids.update(edge_data.get("source_chunks", []))

            # 收集入边
            for predecessor in G.predecessors(node):
                edge_data = G[predecessor][node]
                rel = edge_data.get("label", "关联")
                fact = f"{predecessor} --[{rel}]--> {node}"
                if fact not in facts:
                    facts.append(fact)
                source_chunk_ids.update(edge_data.get("source_chunks", []))

        # 去重
        facts = list(dict.fromkeys(facts))
        sorted_chunk_ids = sorted(source_chunk_ids)

        # 生成摘要文本（模板方式；真实系统中应由 LLM 生成）
        entities_str = "、".join(nodes)
        facts_str = "；".join(facts) if facts else "无事实"

        # 收集关联的原始切片文本用于摘要
        source_texts = []
        for cid in sorted_chunk_ids:
            if cid in chunk_store:
                source_texts.append(chunk_store[cid])
        source_context = " ".join(source_texts)

        summary_text = (
            f"本社区主要涉及以下实体：{entities_str}。"
            f"核心事实包括：{facts_str}。"
            f"相关原始文档内容摘要：{source_context[:200]}"
        )

        summaries.append({
            "community_id": comm_id,
            "entities": nodes,
            "facts": facts,
            "source_chunk_ids": sorted_chunk_ids,
            "summary": summary_text,
        })

    return summaries


# ─────────────────────────────────────────────
# 6. 嵌入器
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
# 7. 构建社区摘要的向量索引
# ─────────────────────────────────────────────
def build_summary_vector_index(
    summaries: list[dict],
    encode_fn: Callable[[list[str]], np.ndarray],
    dim: int,
    db_path: str = "/tmp/community_summaries_lancedb",
    table_name: str = "summaries",
) -> lancedb.table.Table:
    """将每个社区摘要编码为向量，存入 LanceDB，用于语义检索。"""
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    db = lancedb.connect(db_path)
    texts = [s["summary"] for s in summaries]

    print(f"[信息] 为 {len(texts)} 个社区摘要生成嵌入向量 …")
    embeddings = encode_fn(texts)

    schema = pa.schema([
        pa.field("community_id", pa.int32()),
        pa.field("summary", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])

    records = [
        {
            "community_id": s["community_id"],
            "summary": s["summary"],
            "vector": emb.tolist(),
        }
        for s, emb in zip(summaries, embeddings)
    ]

    table = db.create_table(table_name, data=records, schema=schema)
    print(f"[信息] 社区摘要向量索引构建完成，维度={dim}")
    return table


# ─────────────────────────────────────────────
# 8. 社区摘要检索
# ─────────────────────────────────────────────
def search_community_summaries(
    query: str,
    encode_fn: Callable[[list[str]], np.ndarray],
    table: lancedb.table.Table,
    summaries: list[dict],
    top_k: int = 2,
) -> list[dict]:
    """
    检索与查询最相关的社区摘要。
    返回包含摘要文本、来源切片引用等信息的列表。
    """
    query_vec = encode_fn([query])[0].tolist()
    results = table.search(query_vec).limit(top_k).to_pandas()

    matched_summaries = []
    for _, row in results.iterrows():
        comm_id = int(row["community_id"])
        # 找到对应的完整摘要数据
        full_summary = next(
            (s for s in summaries if s["community_id"] == comm_id), None
        )
        if full_summary:
            matched_summaries.append(full_summary)

    return matched_summaries


# ─────────────────────────────────────────────
# 9. 溯源：从摘要追溯到原始切片
# ─────────────────────────────────────────────
def trace_to_source_chunks(
    matched_summaries: list[dict],
    chunk_store: dict[str, str],
) -> list[str]:
    """
    根据匹配的社区摘要，收集所有关联的原始切片文本。
    附带来源引用（Source Citations）。
    """
    source_texts: list[str] = []
    seen_ids: set[str] = set()

    for summary in matched_summaries:
        for cid in summary.get("source_chunk_ids", []):
            if cid not in seen_ids and cid in chunk_store:
                source_texts.append(f"[{cid}] {chunk_store[cid]}")
                seen_ids.add(cid)

    return source_texts


# ─────────────────────────────────────────────
# 10. 答案合成
# ─────────────────────────────────────────────
def synthesize_answer(
    query: str,
    matched_summaries: list[dict],
    source_chunk_texts: list[str],
) -> str:
    """将社区摘要和溯源切片合并为 RAG Prompt。"""

    summaries_parts: list[str] = []
    for s in matched_summaries:
        entities_str = "、".join(s["entities"])
        facts_str = "\n    ".join(s["facts"]) if s["facts"] else "无"
        citations = "、".join(s["source_chunk_ids"])
        summaries_parts.append(
            f"  [社区 {s['community_id']}]\n"
            f"    实体：{entities_str}\n"
            f"    事实：\n    {facts_str}\n"
            f"    来源切片：{citations}\n"
            f"    摘要：{s['summary'][:150]}…"
        )
    summaries_str = "\n\n".join(summaries_parts) if summaries_parts else "（未匹配到相关社区）"

    source_str = "\n  ".join(source_chunk_texts) if source_chunk_texts else "（无关联切片）"

    prompt = f"""
╔══════════════════════════════════════════════════════════╗
║        社区摘要（Community Summaries）Graph-RAG           ║
╚══════════════════════════════════════════════════════════╝

【用户问题】
  {query}

【匹配的社区摘要】
{summaries_str}

【溯源 - 原始切片引用（Source Citations）】
  {source_str}

【参考答案】
  根据社区摘要检索，与"{query}"相关的社区如上所示。
  社区摘要提供了跨文档的全局视角，避免了单一切片的碎片化问题。
  溯源切片提供了原始文档上下文，确保答案可追溯。
  （在真实系统中，以上社区摘要与溯源切片将一并作为上下文送入 LLM 生成回答。）
"""
    return prompt.strip()


# ─────────────────────────────────────────────
# 11. 主流程
# ─────────────────────────────────────────────
def main():
    # 步骤 1：构建知识图谱
    print("\n[步骤 1] 构建知识图谱 …")
    G = build_graph(TRIPLES_WITH_META)
    print(f"  节点数：{G.number_of_nodes()}，边数：{G.number_of_edges()}")

    # 步骤 2：社区检测
    print("\n[步骤 2] 社区检测（Louvain 算法）…")
    communities = detect_communities(G)
    print(f"  检测到 {len(communities)} 个社区：")
    for comm_id, nodes in communities.items():
        print(f"    社区 {comm_id}：{', '.join(nodes)}")

    # 步骤 3：生成社区摘要
    print("\n[步骤 3] 生成社区摘要 …")
    summaries = generate_community_summaries(G, communities, CHUNKS)
    for s in summaries:
        print(f"\n  社区 {s['community_id']}（{len(s['entities'])} 个实体）：")
        print(f"    摘要：{s['summary'][:100]}…")
        print(f"    来源切片：{', '.join(s['source_chunk_ids'])}")

    # 步骤 4：构建嵌入器
    print("\n\n[步骤 4] 初始化嵌入器 …")
    corpus = [s["summary"] for s in summaries] + list(G.nodes())
    encode_fn, dim = build_embedder(corpus=corpus)

    # 步骤 5：构建社区摘要向量索引
    print("\n[步骤 5] 构建社区摘要向量索引 …")
    summary_table = build_summary_vector_index(summaries, encode_fn, dim)

    # 步骤 6：示例查询
    print("\n[步骤 6] 示例查询演示\n")
    sample_queries = [
        "乔布斯做过什么",
        "苹果公司有哪些产品",
        "比尔盖茨创建了什么公司",
        "马斯克的企业",
    ]

    for query in sample_queries:
        print(f"{'─' * 60}")
        matched = search_community_summaries(query, encode_fn, summary_table, summaries, top_k=2)
        source_texts = trace_to_source_chunks(matched, CHUNKS)
        answer = synthesize_answer(query, matched, source_texts)
        print(answer)
        print()

    # 步骤 7：交互式输入
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
        matched = search_community_summaries(user_input, encode_fn, summary_table, summaries, top_k=2)
        source_texts = trace_to_source_chunks(matched, CHUNKS)
        answer = synthesize_answer(user_input, matched, source_texts)
        print(answer)


if __name__ == "__main__":
    main()
