# FlagEmbedding 微调模块详细分析

## 目录
- [整体架构概述](#整体架构概述)
- [抽象基类层](#抽象基类层)
- [Embedder 微调实现](#embedder-微调实现)
- [Reranker 微调实现](#reranker-微调实现)
- [核心训练机制](#核心训练机制)
- [数据流](#数据流)
- [关键数据结构](#关键数据结构)

---

## 整体架构概述

FlagEmbedding 的微调模块采用了**分层抽象架构**，通过抽象基类定义统一接口，再由具体实现类继承并实现具体功能。整个模块分为两个主要部分：**Embedder 微调**和**Reranker 微调**。

```
finetune/
├── embedder/
│   ├── encoder_only/
│   │   ├── base/      # 基础 Encoder-Only 实现
│   │   └── m3/        # BGE-M3 特殊实现
│   └── decoder_only/
│       ├── base/      # 基础 Decoder-Only 实现
│       └── icl/       # In-Context Learning 实现
└── reranker/
    ├── encoder_only/
    │   └── base/      # 基础 Encoder-Only Reranker
    └── decoder_only/
        ├── base/      # 基础 Decoder-Only Reranker
        └── layerwise/ # 分层训练实现
```

---

## 抽象基类层

### AbsEmbedderModel
**文件路径**: [/workspace/FlagEmbedding/abc/finetune/embedder/AbsModeling.py](file:///workspace/FlagEmbedding/abc/finetune/embedder/AbsModeling.py)

`AbsEmbedderModel` 是 Embedder 微调模型的核心抽象基类，定义了所有 Embedder 模型必须实现的接口和核心训练逻辑。

#### 核心方法
1. **`encode(features)`** - 抽象方法，子类实现具体的编码逻辑
2. **`compute_loss(scores, target)`** - 抽象方法，计算损失
3. **`compute_score(q_reps, p_reps)`** - 抽象方法，计算 Query-Passage 相似度分数
4. **`forward(queries, passages, teacher_scores, no_in_batch_neg_flag)`** - 核心前向传播函数

#### 核心训练逻辑
```python
# 1. 编码 queries 和 passages
q_reps = self.encode(queries)
p_reps = self.encode(passages)

# 2. 根据配置选择损失计算策略
if no_in_batch_neg_flag:
    compute_loss_func = self._compute_no_in_batch_neg_loss
else:
    if self.negatives_cross_device:
        compute_loss_func = self._compute_cross_device_neg_loss
    else:
        compute_loss_func = self._compute_in_batch_neg_loss

# 3. 计算损失
scores, loss = compute_loss_func(q_reps, p_reps, teacher_targets=teacher_targets)
```

### AbsRerankerModel
**文件路径**: [/workspace/FlagEmbedding/abc/finetune/reranker/AbsModeling.py](file:///workspace/FlagEmbedding/abc/finetune/reranker/AbsModeling.py)

`AbsRerankerModel` 是 Reranker 模型的抽象基类，用于对检索结果进行重排序。

#### 核心特性
- 使用交叉熵损失函数
- 支持知识蒸馏
- 核心是 `encode` 方法，将查询-文档对编码为分数

---

## Embedder 微调实现

### 1. Encoder-Only Base 实现
**文件路径**: [/workspace/FlagEmbedding/finetune/embedder/encoder_only/base/modeling.py](file:///workspace/FlagEmbedding/finetune/embedder/encoder_only/base/modeling.py)

#### 类定义: `BiEncoderOnlyEmbedderModel`
```python
class BiEncoderOnlyEmbedderModel(AbsEmbedderModel):
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer = None,
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        sub_batch_size: int = -1,
        kd_loss_type: str = 'kl_div',
        use_mrl: bool = False,
        mrl_dims: List[int] = [],
        sentence_pooling_method: str = 'cls',
        normalize_embeddings: bool = False,
    )
```

#### 核心方法
1. **`_sentence_embedding(last_hidden_state, attention_mask)`**
   - 支持三种池化策略: `cls`, `mean`, `last_token`
   - `cls`: 取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token 输出
   - `mean`: 对所有 token 加权平均
   - `last_token`: 取序列最后一个有效 token

2. **`encode(features)`**
   - 支持子批次编码以避免 OOM
   - 支持 MRL (Matryoshka Representation Learning)
   - 可选 embedding 归一化

3. **`compute_score(q_reps, p_reps)`**
   - 使用内积计算相似度
   - 除以温度系数控制分数分布

### 2. Encoder-Only M3 实现
**文件路径**: [/workspace/FlagEmbedding/finetune/embedder/encoder_only/m3/modeling.py](file:///workspace/FlagEmbedding/finetune/embedder/encoder_only/m3/modeling.py)

#### 类定义: `EncoderOnlyEmbedderM3Model`
BGE-M3 是一个多功能模型，同时支持：
- **Dense Embedding**: 传统向量表示
- **Sparse Embedding**: 基于词汇表的稀疏表示
- **ColBERT**: 细粒度 token 级匹配

#### 核心特性
1. **三种嵌入方式**
   - `_dense_embedding()`: 传统池化得到的密集向量
   - `_sparse_embedding()`: 基于词汇表的稀疏权重
   - `_colbert_embedding()`: 每个 token 的向量表示

2. **多损失联合训练**
   ```python
   # 分别计算三种损失
   dense_scores, dense_loss = compute_loss_func(...)
   sparse_scores, sparse_loss = compute_loss_func(...)
   colbert_scores, colbert_loss = compute_loss_func(...)
   
   # 集成损失
   ensemble_scores, ensemble_loss = compute_loss_func(...)
   
   # 加权求和
   loss = (loss + ensemble_loss + 0.1 * sparse_loss + colbert_loss) / 4
   ```

3. **自蒸馏 (Self-Distillation)**
   - 使用集成分数作为教师信号
   - 对单个分支进行蒸馏

### 3. Decoder-Only Base 实现
**文件路径**: [/workspace/FlagEmbedding/finetune/embedder/decoder_only/base/modeling.py](file:///workspace/FlagEmbedding/finetune/embedder/decoder_only/base/modeling.py)

#### 类定义: `BiDecoderOnlyEmbedderModel`
- 基本架构与 Encoder-Only 相似
- 主要区别在于使用的基础模型类型
- 默认使用 `last_token` 池化策略

### 4. Decoder-Only ICL 实现
**文件路径**: [/workspace/FlagEmbedding/finetune/embedder/decoder_only/icl/modeling.py](file:///workspace/FlagEmbedding/finetune/embedder/decoder_only/icl/modeling.py)

#### 类定义: `BiDecoderOnlyEmbedderICLModel`
- 支持 In-Context Learning 方式的微调
- 可以在输入中加入示例以引导模型生成更好的表示

---

## Reranker 微调实现

### 1. Encoder-Only Base 实现
**文件路径**: [/workspace/FlagEmbedding/finetune/reranker/encoder_only/base/modeling.py](file:///workspace/FlagEmbedding/finetune/reranker/encoder_only/base/modeling.py)

#### 类定义: `CrossEncoderModel`
```python
class CrossEncoderModel(AbsRerankerModel):
    def encode(self, features):
        return self.model(**features, return_dict=True).logits
```
- 标准的 Cross-Encoder 架构
- 将查询和文档拼接后输入模型
- 直接使用模型的 logits 作为分数

### 2. Decoder-Only Base 实现
**文件路径**: [/workspace/FlagEmbedding/finetune/reranker/decoder_only/base/modeling.py](file:///workspace/FlagEmbedding/finetune/reranker/decoder_only/base/modeling.py)

#### 类定义: `CrossDecoderModel`
```python
def encode(self, features):
    outputs = self.model(input_ids=features['input_ids'],
                         attention_mask=features['attention_mask'],
                         position_ids=features['position_ids'] if 'position_ids' in features.keys() else None,
                         output_hidden_states=True)
    scores = outputs.logits[:, -1, self.yes_loc]
    return scores.contiguous()
```
- 使用 Decoder 模型作为 Reranker
- 利用 "Yes" token 的概率作为相关性分数

### 3. Decoder-Only Layerwise 实现
**文件路径**: [/workspace/FlagEmbedding/finetune/reranker/decoder_only/layerwise/modeling.py](file:///workspace/FlagEmbedding/finetune/reranker/decoder_only/layerwise/modeling.py)

#### 类定义: `CrossDecoderModel` (Layerwise 版本)
- 分层训练策略
- 所有层都参与损失计算
- 使用深层作为教师信号蒸馏浅层

```python
def forward(self, pair, teacher_scores=None):
    ranker_logits = self.encode(pair)
    
    if self.training:
        loss = 0
        # 所有层都计算损失
        for logits in ranker_logits:
            loss += self.compute_loss(...)
        
        # 自蒸馏: 深层作为教师
        if teacher_scores is None:
            teacher_scores = ranker_logits[-1]
            teacher_targets = torch.softmax(teacher_scores.detach(), dim=-1)
            for logits in ranker_logits[:-1]:
                loss += distillation_loss
```

---

## 核心训练机制

### 1. 对比损失计算

#### 损失策略选择
`AbsEmbedderModel` 提供了三种损失计算策略：

| 策略 | 函数 | 描述 |
|------|------|------|
| 无批内负例 | `_compute_no_in_batch_neg_loss` | 仅使用数据集中提供的负例 |
| 批内负例 | `_compute_in_batch_neg_loss` | 使用同一 batch 内其他样本作为负例 |
| 跨设备负例 | `_compute_cross_device_neg_loss` | 使用所有设备上的样本作为负例 |

#### 核心公式
```
score(q, p) = q · p / temperature
loss = CrossEntropy(score, target)
```

### 2. 批内负例 (In-Batch Negatives)

**实现位置**: [AbsEmbedderModel._compute_in_batch_neg_loss](file:///workspace/FlagEmbedding/abc/finetune/embedder/AbsModeling.py#L171-L201)

#### 工作原理
1. 对于 batch 中的每个 query，batch 中所有其他 passage 都作为负例
2. 目标是让 query 与自己对应的正例 score 最高
3. 目标索引计算: `target = idx * group_size`

```python
def _compute_in_batch_neg_loss(self, q_reps, p_reps, teacher_targets=None, ...):
    # 计算所有 query-passage 对的分数
    scores = self.compute_score(q_reps, p_reps)  # (batch_size, batch_size * group_size)
    
    # 目标是每个 query 对应自己的正例
    idxs = torch.arange(q_reps.size(0), device=q_reps.device)
    targets = idxs * group_size  # (batch_size)
    
    # 计算损失
    loss = self.compute_loss(scores, targets)
    return scores, loss
```

### 3. 跨设备负例 (Cross-Device Negatives)

**实现位置**: [AbsEmbedderModel._compute_cross_device_neg_loss](file:///workspace/FlagEmbedding/abc/finetune/embedder/AbsModeling.py#L203-L241)

#### 工作原理
1. 使用 `torch.distributed` 收集所有设备上的 embedding
2. 在所有设备的 embedding 上计算分数
3. 扩大负例池，提升训练效果

```python
def _compute_cross_device_neg_loss(self, q_reps, p_reps, ...):
    # 收集所有设备的 embedding
    cross_q_reps = self._dist_gather_tensor(q_reps)
    cross_p_reps = self._dist_gather_tensor(p_reps)
    
    # 在全局范围内计算分数
    cross_scores = self.compute_score(cross_q_reps, cross_p_reps)
    
    # 计算损失
    cross_idxs = torch.arange(cross_q_reps.size(0))
    cross_targets = cross_idxs * group_size
    loss = self.compute_loss(cross_scores, cross_targets)
    return cross_scores, loss
```

#### 分布式张量收集
```python
def _dist_gather_tensor(self, t):
    all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
    dist.all_gather(all_tensors, t)
    all_tensors[self.process_rank] = t  # 使用当前设备的张量（保留梯度）
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors
```

### 4. 知识蒸馏 (KD) 损失

**实现位置**: [AbsEmbedderModel.distill_loss](file:///workspace/FlagEmbedding/abc/finetune/embedder/AbsModeling.py#L303-L342)

#### 两种 KD 损失类型
1. **`kl_div`**: 标准 KL 散度
   ```python
   loss = - mean(sum(log_softmax(student) * teacher_target))
   ```

2. **`m3_kd_loss`**: M3 模型专用的多任务 KD
   - 对每个位置进行交叉熵
   - 使用 mask 避免重复计算

```python
def distill_loss(kd_loss_type, teacher_targets, student_scores, group_size=None):
    if kd_loss_type == 'kl_div':
        return - torch.mean(
            torch.sum(torch.log_softmax(student_scores, dim=-1) * teacher_targets, dim=-1)
        )
    elif kd_loss_type == 'm3_kd_loss':
        labels = torch.arange(student_scores.size(0)) * group_size
        loss = 0
        mask = torch.zeros_like(student_scores)
        for i in range(group_size):
            temp_target = labels + i
            temp_scores = student_scores + mask
            temp_loss = F.cross_entropy(temp_scores, temp_target, reduction="none")
            loss += torch.mean(teacher_targets[:, i] * temp_loss)
            mask = torch.scatter(mask, dim=-1, index=temp_target.unsqueeze(-1),
                                value=torch.finfo(student_scores.dtype).min)
        return loss
```

### 5. MRL (Matryoshka Representation Learning)

**实现位置**: [AbsEmbedderModel.forward](file:///workspace/FlagEmbedding/abc/finetune/embedder/AbsModeling.py#L243-L301)

#### 工作原理
- 同一 embedding 的不同维度截断版本都参与训练
- 使模型学习到从低维到高维的嵌套表示
- 推理时可以根据需要选择维度

```python
if self.use_mrl:
    # 对每个维度分别计算损失
    all_loss = torch.tensor(0.0, device=device)
    for dim_q_reps, dim_p_reps in zip(q_reps, p_reps):
        _, mrl_loss = compute_loss_func(dim_q_reps, dim_p_reps, teacher_targets=teacher_targets)
        all_loss += mrl_loss
    loss = all_loss / len(self.mrl_dims)
```

---

## 数据流

### 训练流程

```
1. 数据加载 (AbsDataset)
   ↓
2. 数据整理 (Collator)
   ↓
3. 模型编码 (encode)
   ↓
4. 分数计算 (compute_score)
   ↓
5. 损失计算 (compute_loss)
   ↓
6. 反向传播 & 参数更新
```

### 数据加载

**文件路径**: [/workspace/FlagEmbedding/abc/finetune/embedder/AbsDataset.py](file:///workspace/FlagEmbedding/abc/finetune/embedder/AbsDataset.py)

#### 核心类
1. **`AbsEmbedderTrainDataset`**: 基础数据集
   - 从 json/jsonl 加载数据
   - 支持正负例采样
   - 支持知识蒸馏数据

2. **`AbsEmbedderSameDatasetTrainDataset`**: 同数据集批采样
   - 每个 batch 来自同一数据集
   - 支持动态 batch size
   - 支持 `no_in_batch_neg` 标志

3. **`AbsEmbedderCollator`**: 数据整理器
   - tokenization
   - padding
   - 子批次处理

#### 数据格式
```json
{
  "query": "什么是机器学习？",
  "pos": ["机器学习是人工智能的一个分支..."],
  "neg": ["苹果是一种水果...", "今天天气很好..."],
  "pos_scores": [0.95],  // 可选，用于 KD
  "neg_scores": [0.1, 0.05]  // 可选，用于 KD
}
```

### 训练器实现

**文件路径**: [/workspace/FlagEmbedding/finetune/embedder/encoder_only/base/trainer.py](file:///workspace/FlagEmbedding/finetune/embedder/encoder_only/base/trainer.py)

#### 核心类: `EncoderOnlyEmbedderTrainer`
继承自 `AbsEmbedderTrainer`，主要实现 `_save` 方法。

```python
class EncoderOnlyEmbedderTrainer(AbsEmbedderTrainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # 保存模型
        self.model.save(output_dir)
        # 保存 tokenizer
        self.tokenizer.save_pretrained(output_dir)
        # 保存训练参数
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
```

---

## 关键数据结构

### EmbedderOutput
```python
@dataclass
class EmbedderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None      # Query 表示
    p_reps: Optional[Tensor] = None      # Passage 表示
    loss: Optional[Tensor] = None        # 损失值
    scores: Optional[Tensor] = None      # 相似度分数
```

### RerankerOutput
```python
@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None        # 损失值
    scores: Optional[Tensor] = None      # 相关性分数
```

---

## 总结

FlagEmbedding 微调模块的设计特点：
1. **抽象层完善**: 清晰的抽象基类定义
2. **多种负例策略**: 支持批内、跨设备负例
3. **高级训练技术**: MRL、知识蒸馏、自蒸馏
4. **多模型架构**: Encoder-only、Decoder-only 都支持
5. **多功能模型**: BGE-M3 同时支持多种检索范式
