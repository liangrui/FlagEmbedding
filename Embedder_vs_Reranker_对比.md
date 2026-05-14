
# Embedder vs Reranker 详细对比

本文档对比 `BaseEmbedder` 和 `BaseReranker` 两个类的差异。

---

## 一、概览

| 特性 | [BaseEmbedder](file:///workspace/FlagEmbedding/inference/embedder/encoder_only/base.py) | [BaseReranker](file:///workspace/FlagEmbedding/inference/reranker/encoder_only/base.py) |
|-----|-----|-----|
| **继承自** | `AbsEmbedder` | `AbsReranker` |
| **模型类型** | `AutoModel` | `AutoModelForSequenceClassification` |
| **输出** | 嵌入向量 | 相关性分数 |
| **默认 batch_size** | 256 | 128 |
| **默认 FP16** | True | False |

---

## 二、核心差异详解

### 2.1 模型加载差异

#### Embedder [L78-L88](file:///workspace/FlagEmbedding/inference/embedder/encoder_only/base.py#L78-L88)
```python
self.tokenizer = AutoTokenizer.from_pretrained(...)
self.model = AutoModel.from_pretrained(
    model_name_or_path,
    trust_remote_code=trust_remote_code,
    cache_dir=cache_dir,
    dtype=self.get_model_torch_dtype(),
)
```

#### Reranker [L66-L75](file:///workspace/FlagEmbedding/inference/reranker/encoder_only/base.py#L66-L75)
```python
self.tokenizer = AutoTokenizer.from_pretrained(...)
self.model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, 
    trust_remote_code=trust_remote_code, 
    cache_dir=cache_dir
)
```

**差异点：**
- Embedder 使用 `AutoModel`（通用编码器）
- Reranker 使用 `AutoModelForSequenceClassification`（序列分类模型）
- Embedder 会自动处理 dtype（FP16/BF16）
- Reranker 没有 dtype 参数，在推理时才切换精度

---

### 2.2 推理方法差异

#### Embedder - [encode_single_device](file:///workspace/FlagEmbedding/inference/embedder/encoder_only/base.py#L175-L282)

**输入：** 文本列表
**输出：** 嵌入向量列表
**核心流程：**
```python
# 1. 单独 tokenize 每个句子
inputs_batch = self.tokenizer(sentences_batch, ...)

# 2. 模型前向传播
last_hidden_state = self.model(**inputs_batch, return_dict=True).last_hidden_state

# 3. Pooling 得到嵌入向量
embeddings = self.pooling(last_hidden_state, inputs_batch['attention_mask'])

# 4. 可选截断和归一化
embeddings = self._truncate_embeddings(embeddings)
if self.normalize_embeddings:
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
```

#### Reranker - [compute_score_single_gpu](file:///workspace/FlagEmbedding/inference/reranker/encoder_only/base.py#L78-L195)

**输入：** (查询, 段落) 对列表
**输出：** 相关性分数列表
**核心流程：**
```python
# 1. 分别 tokenize 查询和段落
queries_inputs_batch = self.tokenizer(queries, add_special_tokens=False, ...)
passages_inputs_batch = self.tokenizer(passages, add_special_tokens=False, ...)

# 2. 拼接查询和段落
for q_inp, d_inp in zip(queries_inputs_batch, passages_inputs_batch):
    item = self.tokenizer.prepare_for_model(
        q_inp,
        d_inp,
        truncation='only_second',  # 只截断段落
        max_length=max_length,
        padding=False,
    )
    all_inputs.append(item)

# 3. 模型前向传播，直接得到分数
scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()

# 4. 可选 Sigmoid 归一化
if normalize:
    all_scores = [sigmoid(score) for score in all_scores]
```

---

### 2.3 输入处理差异

#### Embedder 输入处理

简单直接，每个文本独立处理：
```python
inputs_batch = self.tokenizer(
    sentences_batch,
    truncation=True,
    max_length=max_length,
    **kwargs
)
```

#### Reranker 输入处理 - [L124-L154](file:///workspace/FlagEmbedding/inference/reranker/encoder_only/base.py#L124-L154)

更复杂，需要成对处理：
```python
# 1. 分别处理查询和段落
queries_inputs_batch = self.tokenizer(
    queries,
    return_tensors=None,
    add_special_tokens=False,  # 不添加特殊 token
    max_length=query_max_length,
    truncation=True,
    **kwargs
)['input_ids']

passages_inputs_batch = self.tokenizer(
    passages,
    return_tensors=None,
    add_special_tokens=False,  # 不添加特殊 token
    max_length=max_length,
    truncation=True,
    **kwargs
)['input_ids']

# 2. 使用 prepare_for_model 拼接
for q_inp, d_inp in zip(queries_inputs_batch, passages_inputs_batch):
    item = self.tokenizer.prepare_for_model(
        q_inp,
        d_inp,
        truncation='only_second',  # 关键：只截断段落（第二个序列）
        max_length=max_length,
        padding=False,
    )
    all_inputs.append(item)
```

**关键点：**
- Reranker 使用 `add_special_tokens=False`，先不添加特殊 token
- 使用 `tokenizer.prepare_for_model()` 拼接两个序列
- 使用 `truncation='only_second'` 确保优先保留查询，只截断段落

---

### 2.4 Pooling 差异

#### Embedder - [pooling 方法](file:///workspace/FlagEmbedding/inference/embedder/encoder_only/base.py#L284-L308)

有完整的 pooling 实现：
```python
def pooling(self, last_hidden_state, attention_mask):
    if self.pooling_method == 'cls':
        return last_hidden_state[:, 0]
    elif self.pooling_method == 'mean':
        s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        return s / d
```

**支持两种方式：** CLS 和 Mean

#### Reranker

**没有 pooling 方法！**

因为 Reranker 使用的是序列分类模型，模型直接输出 logits（分数），不需要 pooling。

```python
# 直接取 logits
scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
```

---

### 2.5 归一化差异

#### Embedder - [L263-L264](file:///workspace/FlagEmbedding/inference/embedder/encoder_only/base.py#L263-L264)
```python
if self.normalize_embeddings:
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
```
使用 **L2 归一化**，使向量长度为 1，便于计算余弦相似度。

#### Reranker - [L192-L193](file:///workspace/FlagEmbedding/inference/reranker/encoder_only/base.py#L192-L193)
```python
if normalize:
    all_scores = [sigmoid(score) for score in all_scores]
```
使用 **Sigmoid 归一化**，将分数映射到 (0, 1) 区间：
```python
def sigmoid(x):
    return float(1 / (1 + np.exp(-x)))
```

---

### 2.6 FP16 处理差异

#### Embedder - [L87](file:///workspace/FlagEmbedding/inference/embedder/encoder_only/base.py#L87)
在初始化时就设置 dtype：
```python
self.model = AutoModel.from_pretrained(
    ...,
    dtype=self.get_model_torch_dtype(),
)
```

#### Reranker - [L113-L114](file:///workspace/FlagEmbedding/inference/reranker/encoder_only/base.py#L113-L114)
在推理时才切换：
```python
if device == "cpu": self.use_fp16 = False
if self.use_fp16: self.model.half()
```

---

## 三、详细对比表格

| 方面 | BaseEmbedder | BaseReranker |
|-----|-----|-----|
| **继承类** | `AbsEmbedder` | `AbsReranker` |
| **主要方法** | `encode_single_device` | `compute_score_single_gpu` |
| **输入** | 单个文本 | (查询, 段落) 对 |
| **输出** | 嵌入向量 | 相关性分数 |
| **模型类** | `AutoModel` | `AutoModelForSequenceClassification` |
| **Pooling** | ✅ CLS/Mean | ❌ 无（直接用 logits） |
| **归一化** | L2 归一化 | Sigmoid 归一化 |
| **输入拼接** | ❌ 不需要 | ✅ `prepare_for_model` 拼接查询和段落 |
| **截断策略** | 普通截断 | `only_second` 只截断段落 |
| **默认 batch_size** | 256 | 128 |
| **默认 FP16** | True | False |
| **特殊 token** | 正常添加 | 先 `add_special_tokens=False`，后拼接 |
| **返回类型** | `np.ndarray` 或 `torch.Tensor` | `List[float]` |

---

## 四、使用场景对比

### Embedder 的使用场景
1. **向量检索** - 将文档编码为向量存入向量数据库
2. **语义相似度计算** - 计算两个句子的相似度
3. **聚类** - 对文本进行聚类分析
4. **分类任务的特征提取** - 作为其他模型的输入特征

### Reranker 的使用场景
1. **检索结果重排序** - 对初检结果进行精细排序
2. **问答系统** - 找出最相关的段落
3. **信息提取** - 找出与查询最匹配的文本

---

## 五、两阶段检索流程

通常这两个模型会配合使用：

```
阶段 1: 粗检索（Embedder）
┌─────────────────────────────────────┐
│  1. 对所有文档编码，存入向量数据库  │
│  2. 对查询编码                       │
│  3. 快速检索 Top-K 个最相关文档       │
└─────────────────────────────────────┘
           ↓ (得到 K 个候选文档)
           
阶段 2: 精排序（Reranker）
┌─────────────────────────────────────┐
│  1. 将查询与每个候选文档配对         │
│  2. 使用 Reranker 计算相关性分数     │
│  3. 根据分数重新排序                 │
└─────────────────────────────────────┘
           ↓ (得到最终排序结果)
```

**为什么这样设计？**
- Embedder 速度快，但精度相对低（适合从大量文档中快速筛选）
- Reranker 精度高，但速度慢（适合对少量候选进行精细排序）

---

## 六、代码示例对比

### Embedder 使用示例
```python
from FlagEmbedding import FlagAutoModel

model = FlagAutoModel.from_finetuned('BAAI/bge-small-en-v1.5')

# 编码文档
docs = ["文档 1", "文档 2", "文档 3"]
doc_embeddings = model.encode(docs)

# 编码查询
query = "用户查询"
query_embedding = model.encode_queries([query])

# 计算相似度
similarities = query_embedding @ doc_embeddings.T
```

### Reranker 使用示例
```python
from FlagEmbedding import FlagAutoReranker

reranker = FlagAutoReranker.from_finetuned('BAAI/bge-reranker-base')

# 准备 (查询, 文档) 对
query = "用户查询"
docs = ["文档 1", "文档 2", "文档 3"]
pairs = [(query, doc) for doc in docs]

# 计算分数
scores = reranker.compute_score(pairs)
# 输出: [0.8, 0.3, 0.6] 表示各文档的相关性
```

---

## 七、总结

### 核心差异总结

| 维度 | Embedder | Reranker |
|-----|-----|-----|
| **目标** | 生成语义表示 | 判断相关性 |
| **建模方式** | 特征提取 | 序列分类 |
| **计算复杂度** | 低（单向编码） | 高（双向交互） |
| **适用数据规模** | 大规模（百万级） | 小规模（百级） |
| **与其他模型配合** | 可独立使用 | 通常配合 Embedder 使用 |

### 最佳实践
1. **第一阶段**：用 Embedder 快速检索 Top-100 文档
2. **第二阶段**：用 Reranker 对这 100 个文档精细排序
3. **结果**：兼顾速度和精度

---

## 代码参考

- [BaseEmbedder](file:///workspace/FlagEmbedding/inference/embedder/encoder_only/base.py)
- [BaseReranker](file:///workspace/FlagEmbedding/inference/reranker/encoder_only/base.py)
