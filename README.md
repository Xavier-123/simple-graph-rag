# simple-graph-rag

一个简易的 Graph-RAG（图增强检索生成）最小可行性原型（MVP）。

## 功能概述

| 模块 | 说明 |
|------|------|
| 知识库 | 手动定义知识三元组（主体 → 关系 → 客体） |
| 图谱 | 使用 [NetworkX](https://networkx.org/) 将三元组构建为有向图 |
| 向量检索 | 使用 [LanceDB](https://lancedb.github.io/lancedb/) 存储节点向量，语义检索最近节点 |
| 图增强 | 通过 `G.neighbors()` 提取相关三元组，扩展上下文 |
| 答案合成 | 将检索结果与图谱事实合并，以 RAG Prompt 格式输出 |

## 快速开始

### 1. 安装依赖

```bash
pip install networkx lancedb sentence-transformers numpy pyarrow pandas
```

### 2. 运行

```bash
python graph_rag.py
```

脚本会先演示 4 个示例查询，然后进入交互式问答模式（输入 `exit` 退出）。

## Embedding 策略

脚本默认使用**字符 n-gram TF-IDF**向量，无需联网即可运行。

若需要更高质量的语义检索，可在 `graph_rag.py` 顶部将：

```python
USE_SENTENCE_TRANSFORMERS = False
```

改为：

```python
USE_SENTENCE_TRANSFORMERS = True
```

首次运行时会自动从 HuggingFace 下载 `paraphrase-multilingual-MiniLM-L12-v2` 模型（约 470 MB，CPU 友好）。

## 示例输出

```
【用户问题】
  乔布斯做过什么

【向量检索结果】（语义最近节点）
  乔布斯、马斯克、皮克斯

【图谱关联事实】（来自 NetworkX 邻居扩展）
  乔布斯 --[创立]--> 苹果公司
  乔布斯 --[创立]--> 皮克斯
  乔布斯 --[毕业于]--> 里德学院
  ...
```
