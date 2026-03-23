"""
方案二：双向索引（Hybrid Indexing）

核心思路：
    建立图索引和向量索引的联动，支持两种检索路径：
    - Top-down（从切片到图）：先向量检索找到最相关切片，再根据切片中的实体
      去图中查询二度/三度邻居，扩展背景知识。
    - Bottom-up（从图到切片）：先通过图检索找到核心事实，再根据事实关联的
      ChunkID 找到切片，并取前后相邻切片（Window Retrieval）补充上下文。

运行方式：
    python hybrid_indexing.py
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
# 1. 文档切片定义（有序列表，支持窗口检索）
# ─────────────────────────────────────────────
CHUNKS = [
    {"id": "chunk_001", "text": "乔布斯（Steve Jobs）于 1976 年与沃兹尼亚克共同创立了苹果公司，后来又创立了皮克斯动画工作室。",
     "entities": ["乔布斯", "苹果公司", "皮克斯"]},
    {"id": "chunk_002", "text": "乔布斯年轻时曾就读于里德学院，但仅旁听了一个学期便退学。",
     "entities": ["乔布斯", "里德学院"]},
    {"id": "chunk_003", "text": "苹果公司总部位于加州库比蒂诺，是全球最具价值的科技企业之一。",
     "entities": ["苹果公司", "库比蒂诺"]},
    {"id": "chunk_004", "text": "苹果公司的主要产品包括 iPhone 智能手机和 Mac 系列电脑，iPhone 运行 iOS 操作系统。",
     "entities": ["苹果公司", "iPhone", "Mac", "iOS"]},
    {"id": "chunk_005", "text": "比尔·盖茨于 1975 年创立了微软公司，他曾就读于哈佛大学但中途辍学。",
     "entities": ["比尔盖茨", "微软", "哈佛大学"]},
    {"id": "chunk_006", "text": "微软总部位于华盛顿州雷德蒙德，主要产品有 Windows 操作系统和 Office 办公套件。",
     "entities": ["微软", "雷德蒙德", "Windows", "Office"]},
    {"id": "chunk_007", "text": "埃隆·马斯克创立了特斯拉和 SpaceX 两家公司。特斯拉主要生产电动汽车。",
     "entities": ["马斯克", "特斯拉", "SpaceX", "电动汽车"]},
    {"id": "chunk_008", "text": "SpaceX 开发了猎鹰系列运载火箭，致力于降低太空运输成本。",
     "entities": ["SpaceX", "猎鹰火箭"]},
    {"id": "chunk_009", "text": "皮克斯动画工作室制作了《玩具总动员》等经典动画电影。",
     "entities": ["皮克斯", "玩具总动员"]},
]

# 构建辅助索引：chunk_id → 列表位置（用于窗口检索）
CHUNK_INDEX = {chunk["id"]: i for i, chunk in enumerate(CHUNKS)}
CHUNK_BY_ID = {chunk["id"]: chunk for chunk in CHUNKS}

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
# 3. 构建知识图谱（节点/边均携带 source_chunks）
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
# 4. 嵌入器
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
# 5. 构建双向索引：向量索引（切片级）+ 图索引
# ─────────────────────────────────────────────
def build_chunk_vector_index(
    chunks: list[dict],
    encode_fn: Callable[[list[str]], np.ndarray],
    dim: int,
    db_path: str = "/tmp/hybrid_indexing_lancedb",
    table_name: str = "chunks",
) -> lancedb.table.Table:
    """构建切片级向量索引（向量数据库存储切片文本的嵌入）。"""
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    db = lancedb.connect(db_path)
    texts = [c["text"] for c in chunks]

    print(f"[信息] 为 {len(texts)} 个切片生成嵌入向量 …")
    embeddings = encode_fn(texts)

    schema = pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("chunk_text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])

    records = [
        {"chunk_id": c["id"], "chunk_text": c["text"], "vector": emb.tolist()}
        for c, emb in zip(chunks, embeddings)
    ]

    table = db.create_table(table_name, data=records, schema=schema)
    print(f"[信息] 切片向量索引构建完成，维度={dim}")
    return table


def vector_search_chunks(
    query: str,
    encode_fn: Callable[[list[str]], np.ndarray],
    table: lancedb.table.Table,
    top_k: int = 3,
) -> list[dict]:
    """向量检索：返回 top_k 个最相关切片的 {chunk_id, chunk_text}。"""
    query_vec = encode_fn([query])[0].tolist()
    results = table.search(query_vec).limit(top_k).to_pandas()
    return [
        {"chunk_id": row["chunk_id"], "chunk_text": row["chunk_text"]}
        for _, row in results.iterrows()
    ]


# ─────────────────────────────────────────────
# 6. Top-down：从切片到图（向量检索 → 图扩展）
# ─────────────────────────────────────────────
def get_n_hop_neighbors(G: nx.DiGraph, entity: str, max_hops: int = 2) -> list[str]:
    """获取实体的 N 度邻居（包含入边和出边方向）。"""
    visited: set[str] = set()
    current_level: set[str] = {entity}

    for _ in range(max_hops):
        next_level: set[str] = set()
        for node in current_level:
            if node in G:
                next_level.update(G.neighbors(node))
                next_level.update(G.predecessors(node))
        visited.update(current_level)
        current_level = next_level - visited

    visited.update(current_level)
    visited.discard(entity)
    return list(visited)


def extract_facts(G: nx.DiGraph, entities: list[str]) -> list[str]:
    """提取一组实体间所有关联的三元组事实。"""
    facts: list[str] = []
    visited_edges: set[tuple[str, str]] = set()

    for node in entities:
        if node not in G:
            continue
        for neighbor in G.neighbors(node):
            edge_key = (node, neighbor)
            if edge_key not in visited_edges:
                rel = G[node][neighbor].get("label", "关联")
                facts.append(f"{node} --[{rel}]--> {neighbor}")
                visited_edges.add(edge_key)
        for predecessor in G.predecessors(node):
            edge_key = (predecessor, node)
            if edge_key not in visited_edges:
                rel = G[predecessor][node].get("label", "关联")
                facts.append(f"{predecessor} --[{rel}]--> {node}")
                visited_edges.add(edge_key)

    return facts


def top_down_retrieval(
    query: str,
    encode_fn: Callable[[list[str]], np.ndarray],
    chunk_table: lancedb.table.Table,
    G: nx.DiGraph,
    top_k: int = 3,
    max_hops: int = 2,
) -> dict:
    """
    Top-down 检索路径：
    1. 向量检索找到最相关的 Top-K 个切片
    2. 提取切片中的实体
    3. 去图中查询这些实体的 N 度邻居，扩展背景知识
    """
    # 步骤 1：向量检索最相关切片
    matched_chunks = vector_search_chunks(query, encode_fn, chunk_table, top_k)

    # 步骤 2：从命中切片中提取实体
    entities_in_chunks: set[str] = set()
    for mc in matched_chunks:
        chunk_data = CHUNK_BY_ID.get(mc["chunk_id"])
        if chunk_data:
            entities_in_chunks.update(chunk_data.get("entities", []))

    # 步骤 3：图扩展 — 获取 N 度邻居
    expanded_entities: set[str] = set(entities_in_chunks)
    for entity in entities_in_chunks:
        neighbors = get_n_hop_neighbors(G, entity, max_hops=max_hops)
        expanded_entities.update(neighbors)

    # 步骤 4：提取扩展后所有实体的关联事实
    facts = extract_facts(G, list(expanded_entities))

    return {
        "matched_chunks": matched_chunks,
        "entities_in_chunks": sorted(entities_in_chunks),
        "expanded_entities": sorted(expanded_entities),
        "facts": facts,
    }


# ─────────────────────────────────────────────
# 7. Bottom-up：从图到切片（图检索 → 窗口检索）
# ─────────────────────────────────────────────
def window_retrieval(chunk_id: str, window_size: int = 1) -> list[dict]:
    """
    窗口检索：取指定切片及其前后相邻切片，补充上下文连贯性。
    window_size=1 表示取前后各 1 个切片。
    """
    if chunk_id not in CHUNK_INDEX:
        return []

    center_idx = CHUNK_INDEX[chunk_id]
    start = max(0, center_idx - window_size)
    end = min(len(CHUNKS), center_idx + window_size + 1)

    return [
        {"chunk_id": CHUNKS[i]["id"], "chunk_text": CHUNKS[i]["text"],
         "is_center": i == center_idx}
        for i in range(start, end)
    ]


def bottom_up_retrieval(
    query: str,
    encode_fn: Callable[[list[str]], np.ndarray],
    G: nx.DiGraph,
    top_k: int = 3,
    window_size: int = 1,
) -> dict:
    """
    Bottom-up 检索路径：
    1. 根据查询在图中找到核心实体（简化：用 n-gram 匹配图节点）
    2. 提取核心事实和关联的 ChunkID
    3. 对每个 ChunkID 做窗口检索（取前后相邻切片）
    """
    # 步骤 1：从查询中提取可能的实体（简化：匹配图中节点名称）
    query_entities: list[str] = []
    for node in G.nodes():
        if node in query or query in node:
            query_entities.append(node)

    # 若精确匹配未命中，使用向量相似度在节点名上做模糊匹配
    if not query_entities:
        node_list = list(G.nodes())
        query_vec = encode_fn([query])[0]
        node_vecs = encode_fn(node_list)
        similarities = node_vecs @ query_vec
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        query_entities = [node_list[i] for i in top_indices]

    # 步骤 2：图检索 — 提取核心事实和关联 ChunkID
    facts: list[str] = []
    chunk_id_set: set[str] = set()
    visited_edges: set[tuple[str, str]] = set()

    for node in query_entities:
        if node not in G:
            continue
        node_chunks = G.nodes[node].get("source_chunks", [])
        chunk_id_set.update(node_chunks)

        for neighbor in G.neighbors(node):
            edge_key = (node, neighbor)
            if edge_key not in visited_edges:
                edge_data = G[node][neighbor]
                rel = edge_data.get("label", "关联")
                facts.append(f"{node} --[{rel}]--> {neighbor}")
                visited_edges.add(edge_key)
                chunk_id_set.update(edge_data.get("source_chunks", []))

        for predecessor in G.predecessors(node):
            edge_key = (predecessor, node)
            if edge_key not in visited_edges:
                edge_data = G[predecessor][node]
                rel = edge_data.get("label", "关联")
                facts.append(f"{predecessor} --[{rel}]--> {node}")
                visited_edges.add(edge_key)
                chunk_id_set.update(edge_data.get("source_chunks", []))

    # 步骤 3：窗口检索 — 取回切片及其相邻切片
    windowed_chunks: list[dict] = []
    seen_chunk_ids: set[str] = set()
    for cid in sorted(chunk_id_set):
        window = window_retrieval(cid, window_size=window_size)
        for wc in window:
            if wc["chunk_id"] not in seen_chunk_ids:
                windowed_chunks.append(wc)
                seen_chunk_ids.add(wc["chunk_id"])

    return {
        "query_entities": query_entities,
        "facts": facts,
        "core_chunk_ids": sorted(chunk_id_set),
        "windowed_chunks": windowed_chunks,
    }


# ─────────────────────────────────────────────
# 8. 答案合成
# ─────────────────────────────────────────────
def synthesize_answer(
    query: str,
    td_result: dict,
    bu_result: dict,
) -> str:
    """将 Top-down 和 Bottom-up 的检索结果合并为 RAG Prompt。"""

    # Top-down 部分
    td_chunks = "\n  ".join(
        [f"[{c['chunk_id']}] {c['chunk_text']}" for c in td_result["matched_chunks"]]
    ) if td_result["matched_chunks"] else "（无）"
    td_entities = "、".join(td_result["entities_in_chunks"]) if td_result["entities_in_chunks"] else "（无）"
    td_expanded = "、".join(td_result["expanded_entities"]) if td_result["expanded_entities"] else "（无）"
    td_facts = "\n  ".join(td_result["facts"]) if td_result["facts"] else "（无）"

    # Bottom-up 部分
    bu_entities = "、".join(bu_result["query_entities"]) if bu_result["query_entities"] else "（无）"
    bu_facts = "\n  ".join(bu_result["facts"]) if bu_result["facts"] else "（无）"
    bu_chunks_parts: list[str] = []
    for wc in bu_result["windowed_chunks"]:
        marker = " ★" if wc.get("is_center") else ""
        bu_chunks_parts.append(f"[{wc['chunk_id']}]{marker} {wc['chunk_text']}")
    bu_chunks = "\n  ".join(bu_chunks_parts) if bu_chunks_parts else "（无）"

    prompt = f"""
╔══════════════════════════════════════════════════════════╗
║          双向索引（Hybrid Indexing）Graph-RAG             ║
╚══════════════════════════════════════════════════════════╝

【用户问题】
  {query}

═══ Top-down 路径（从切片到图）═══

  [向量检索命中切片]
  {td_chunks}

  [切片中提取的实体]
  {td_entities}

  [图扩展后的实体集合（{len(td_result['expanded_entities'])} 个）]
  {td_expanded}

  [图扩展关联事实]
  {td_facts}

═══ Bottom-up 路径（从图到切片）═══

  [查询命中实体]
  {bu_entities}

  [图谱核心事实]
  {bu_facts}

  [窗口检索结果]（★ 标记为核心切片，其余为相邻上下文）
  {bu_chunks}

【综合参考答案】
  Top-down 路径找到了相关切片并通过图扩展获得了更丰富的背景知识。
  Bottom-up 路径从图谱出发定位核心事实，并通过窗口检索补充了上下文连贯性。
  （在真实系统中，以上双向检索结果将合并后作为上下文送入 LLM 生成回答。）
"""
    return prompt.strip()


# ─────────────────────────────────────────────
# 9. 主流程
# ─────────────────────────────────────────────
def main():
    # 步骤 1：构建知识图谱
    print("\n[步骤 1] 构建知识图谱 …")
    G = build_graph(TRIPLES_WITH_META)
    print(f"  节点数：{G.number_of_nodes()}，边数：{G.number_of_edges()}")

    # 步骤 2：构建嵌入器（用切片文本 + 节点名称作为语料库）
    print("\n[步骤 2] 初始化嵌入器 …")
    corpus = [c["text"] for c in CHUNKS] + list(G.nodes())
    encode_fn, dim = build_embedder(corpus=corpus)

    # 步骤 3：构建切片级向量索引
    print("\n[步骤 3] 构建切片向量索引 …")
    chunk_table = build_chunk_vector_index(CHUNKS, encode_fn, dim)

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
        td_result = top_down_retrieval(query, encode_fn, chunk_table, G, top_k=3, max_hops=2)
        bu_result = bottom_up_retrieval(query, encode_fn, G, top_k=3, window_size=1)
        answer = synthesize_answer(query, td_result, bu_result)
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
        td_result = top_down_retrieval(user_input, encode_fn, chunk_table, G, top_k=3, max_hops=2)
        bu_result = bottom_up_retrieval(user_input, encode_fn, G, top_k=3, window_size=1)
        answer = synthesize_answer(user_input, td_result, bu_result)
        print(answer)


if __name__ == "__main__":
    main()
