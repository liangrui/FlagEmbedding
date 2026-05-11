# FlagEmbedding 评估模块详解

## 目录
- [1. 整体架构](#1-整体架构)
- [2. 抽象基类分析](#2-抽象基类分析)
- [3. 各评估基准实现](#3-各评估基准实现)
  - [3.1 MTEB / C-MTEB](#31-mteb--c-mteb)
  - [3.2 BEIR](#32-beir)
  - [3.3 MSMARCO](#33-msmarco)
  - [3.4 MIRACL](#34-miracl)
  - [3.5 MKQA](#35-mkqa)
  - [3.6 MLDR](#36-mldr)
  - [3.7 AIR-Bench](#37-air-bench)
  - [3.8 BRIGHT](#38-bright)
- [4. 评估流程与指标](#4-评估流程与指标)

---

## 1. 整体架构

FlagEmbedding 的评估模块采用了**抽象基类 + 具体实现**的设计模式。评估流程主要包括：

```
数据加载 → 检索/重排序 → 评估指标计算
```

目录结构如下：
```
FlagEmbedding/
├── abc/evaluation/          # 抽象基类定义
└── evaluation/
    ├── air_bench/           # AIR Bench 评估
    ├── beir/                # BEIR 评估
    ├── bright/              # BRIGHT 评估
    ├── custom/              # 自定义评估
    ├── miracl/              # MIRACL 评估
    ├── mkqa/                # MKQA 评估
    ├── mldr/                # MLDR 评估
    ├── msmarco/             # MSMARCO 评估
    └── mteb/                # MTEB 评估
```

---

## 2. 抽象基类分析

### 2.1 AbsEvalRunner [runner.py](file:///workspace/FlagEmbedding/abc/evaluation/runner.py)

评估运行器的抽象基类，负责协调整个评估流程：

```python
class AbsEvalRunner:
    def __init__(self, eval_args, model_args):
        # 加载检索器、重排序器、数据加载器和评估器
        self.retriever, self.reranker = self.load_retriever_and_reranker()
        self.data_loader = self.load_data_loader()
        self.evaluator = self.load_evaluator()
```

**核心方法**：
- `get_models()`: 加载嵌入模型和重排序模型
- `load_retriever_and_reranker()`: 创建检索器和重排序器
- `load_data_loader()`: 加载数据加载器
- `load_evaluator()`: 加载评估器
- `evaluate_metrics()`: 计算评估指标
- `run()`: 运行完整评估流程

### 2.2 AbsEvaluator [evaluator.py](file:///workspace/FlagEmbedding/abc/evaluation/evaluator.py)

评估器抽象基类，负责执行检索和评估：

```python
class AbsEvaluator:
    def __init__(self, eval_name, data_loader, overwrite=False):
        self.eval_name = eval_name
        self.data_loader = data_loader
        self.overwrite = overwrite
```

**核心流程**：
1. 加载语料库和查询
2. 执行检索（`retriever()`）
3. 保存检索结果
4. 如果有重排序器，执行重排序
5. 计算评估指标

### 2.3 EvalDenseRetriever & EvalReranker [searcher.py](file:///workspace/FlagEmbedding/abc/evaluation/searcher.py)

**EvalDenseRetriever**：
- 使用嵌入模型将语料库转换为向量
- 使用 Faiss 索引进行快速检索
- 支持缓存语料库向量

**EvalReranker**：
- 对检索结果进行重排序
- 截断到指定数量的 top-k 结果

### 2.4 AbsEvalDataLoader [data_loader.py](file:///workspace/FlagEmbedding/abc/evaluation/data_loader.py)

数据加载器抽象基类，负责加载评估数据：

```python
class AbsEvalDataLoader(ABC):
    def __init__(self, eval_name, dataset_dir=None, cache_dir=None, ...):
        # 初始化参数
```

**核心方法**：
- `load_corpus()`: 加载语料库
- `load_queries()`: 加载查询
- `load_qrels()`: 加载相关性标注
- `_load_remote_*()`: 从远程下载数据
- `_load_local_*()`: 从本地加载数据

### 2.5 AbsEvalArgs & AbsEvalModelArgs [arguments.py](file:///workspace/FlagEmbedding/abc/evaluation/arguments.py)

评估和模型参数的数据类定义。

---

## 3. 各评估基准实现

### 3.1 MTEB / C-MTEB

**实现文件**：[evaluation/mteb/runner.py](file:///workspace/FlagEmbedding/evaluation/mteb/runner.py)

**特点**：
- 基于第三方 `mteb` 库
- 支持多种任务类型（分类、聚类、检索等）
- 支持多语言
- 提供任务特定的指令

**核心类**：`MTEBEvalRunner`

**主要流程**：
1. 使用 `mteb.get_tasks()` 获取任务
2. 遍历任务，为每个任务设置适当的指令
3. 运行评估
4. 读取并整理结果

**示例结果格式**：
```json
{
  "task_type": {
    "task_name": score,
    "Avg": average_score
  },
  "Avg": overall_average
}
```

### 3.2 BEIR

**实现文件**：
- [evaluation/beir/runner.py](file:///workspace/FlagEmbedding/evaluation/beir/runner.py)
- [evaluation/beir/data_loader.py](file:///workspace/FlagEmbedding/evaluation/beir/data_loader.py)

**特点**：
- 异构检索基准
- 包含 15 个数据集
- 支持多种任务（问答、新闻检索、科学文献检索等）

**支持的数据集**：
- arguana, climate-fever, cqadupstack, dbpedia-entity
- fever, fiqa, hotpotqa, msmarco, nfcorpus, nq
- quora, scidocs, scifact, trec-covid, webis-touche2020

**核心类**：
- `BEIREvalRunner`: BEIR 评估运行器
- `BEIREvalDataLoader`: BEIR 数据加载器

**特殊功能**：
- 为不同数据集提供特定的查询指令（[prompts.py](file:///workspace/FlagEmbedding/evaluation/beir/prompts.py)）
- 支持 cqadupstack 的子数据集

### 3.3 MSMARCO

**实现文件**：
- [evaluation/msmarco/runner.py](file:///workspace/FlagEmbedding/evaluation/msmarco/runner.py)
- [evaluation/msmarco/data_loader.py](file:///workspace/FlagEmbedding/evaluation/msmarco/data_loader.py)

**特点**：
- 微软大规模机器阅读理解数据集
- 支持 passage 和 document 两种模式
- 支持 dev, dl19, dl20 三个分割

**核心类**：
- `MSMARCOEvalRunner`: MSMARCO 评估运行器
- `MSMARCOEvalDataLoader`: MSMARCO 数据加载器

**数据来源**：
- passage 模式：`Tevatron/msmarco-passage-corpus`
- document 模式：`irds/msmarco-document`

### 3.4 MIRACL

**实现文件**：
- [evaluation/miracl/runner.py](file:///workspace/FlagEmbedding/evaluation/miracl/runner.py)
- [evaluation/miracl/data_loader.py](file:///workspace/FlagEmbedding/evaluation/miracl/data_loader.py)

**特点**：
- 多语言信息检索基准
- 支持 18 种语言
- 包含 train 和 dev 分割

**支持的语言**：
- ar, bn, en, es, fa, fi, fr, hi, id, ja, ko, ru, sw, te, th, zh, de, yo

**核心类**：
- `MIRACLEvalRunner`: MIRACL 评估运行器
- `MIRACLEvalDataLoader`: MIRACL 数据加载器

**数据来源**：`miracl/miracl-corpus` (HuggingFace)

### 3.5 MKQA

**实现文件**：
- [evaluation/mkqa/runner.py](file:///workspace/FlagEmbedding/evaluation/mkqa/runner.py)
- [evaluation/mkqa/data_loader.py](file:///workspace/FlagEmbedding/evaluation/mkqa/data_loader.py)

**特点**：
- 多语言知识问答数据集
- 支持 26 种语言/方言
- 评估答案在检索结果中的存在性

**支持的语言**：
- en, ar, fi, ja, ko, ru, es, sv, he, th, da, de, fr, it, nl, pt, hu, vi, ms, km, no, tr, zh_cn, zh_hk, zh_tw

**核心类**：
- `MKQAEvalRunner`: MKQA 评估运行器
- `MKQAEvalDataLoader`: MKQA 数据加载器

**特殊功能**：
- 所有语言共享同一个语料库（来自 BeIR 的 nq 数据集）
- 文本标准化处理（[utils/normalize_text.py](file:///workspace/FlagEmbedding/evaluation/mkqa/utils/normalize_text.py)）

**数据来源**：`Shitao/bge-m3-data` (HuggingFace)

### 3.6 MLDR

**实现文件**：
- [evaluation/mldr/runner.py](file:///workspace/FlagEmbedding/evaluation/mldr/runner.py)
- [evaluation/mldr/data_loader.py](file:///workspace/FlagEmbedding/evaluation/mldr/data_loader.py)

**特点**：
- 多语言长文档检索基准
- 支持 13 种语言
- 包含 train, dev, test 三个分割

**支持的语言**：
- ar, de, en, es, fr, hi, it, ja, ko, pt, ru, th, zh

**核心类**：
- `MLDREvalRunner`: MLDR 评估运行器
- `MLDREvalDataLoader`: MLDR 数据加载器

**数据来源**：`Shitao/MLDR` (HuggingFace)

### 3.7 AIR-Bench

**实现文件**：[evaluation/air_bench/runner.py](file:///workspace/FlagEmbedding/evaluation/air_bench/runner.py)

**特点**：
- 基于第三方 `air-bench` 库
- 评估检索系统在实际场景中的性能
- 支持多种任务类型、领域、语言

**核心类**：`AIRBenchEvalRunner`

**主要流程**：
1. 创建 `AIRBench` 实例
2. 传入任务类型、领域、语言等参数
3. 运行评估

### 3.8 BRIGHT

**实现文件**：[evaluation/bright/runner.py](file:///workspace/FlagEmbedding/evaluation/bright/runner.py)

**特点**：
- 支持两种任务类型：short 和 long
- 为不同任务提供特定的查询指令
- 在检索时排除已知相关文档

**核心类**：
- `BrightEvalRunner`: BRIGHT 评估运行器
- `BrightEvalDenseRetriever`: BRIGHT 检索器
- `BrightShortEvalDataLoader` / `BrightLongEvalDataLoader`: 数据加载器

**特殊功能**：
- 支持任务特定的指令（[prompts.py](file:///workspace/FlagEmbedding/evaluation/bright/prompts.py)）
- 可在检索时通过 `retriever_qrels` 排除已知相关文档

---

## 4. 评估流程与指标

### 4.1 完整评估流程

1. **初始化**：加载模型、数据加载器、评估器
2. **数据准备**：加载语料库、查询、相关性标注
3. **检索阶段**：
   - 编码语料库（支持缓存）
   - 编码查询
   - 使用 Faiss 进行检索
   - 保存检索结果
4. **重排序阶段**（可选）：
   - 对检索结果进行重排序
   - 保存重排序结果
5. **指标计算**：
   - 加载检索结果和 qrels
   - 计算各项指标
   - 输出结果（JSON 或 Markdown）

### 4.2 主要评估指标

所有评估基准共同使用的指标包括：

- **NDCG@k**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision
- **Recall@k**: 召回率
- **Precision@k**: 精确率
- **MRR**: Mean Reciprocal Rank
- **Recall Cap**: 有召回上限的召回率

这些指标通过 [utils.py](file:///workspace/FlagEmbedding/abc/evaluation/utils.py) 中的函数计算。

### 4.3 结果输出格式

**JSON 格式**：
```json
{
  "model_name": {
    "reranker_name": {
      "split": {
        "ndcg_at_10": 0.5,
        "recall_at_10": 0.6,
        ...
      }
    }
  }
}
```

**Markdown 格式**：
生成表格，高亮每个指标的最佳结果。

---

## 5. 总结

FlagEmbedding 的评估模块设计精巧，具有以下优点：

1. **统一的抽象接口**：通过抽象基类统一各评估基准的接口
2. **模块化设计**：各评估基准独立实现，易于扩展
3. **灵活的配置**：支持多种参数配置（top-k、指标、输出格式等）
4. **缓存机制**：支持缓存语料库向量和检索结果，提高效率
5. **多基准支持**：覆盖主流的检索评估基准
