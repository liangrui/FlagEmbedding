# FlagEmbedding 推理模块深度分析

本分析文档深入研究 FlagEmbedding 项目的推理模块（`inference`），包括自动模型加载、Embedder 和 Reranker 的实现细节。

## 目录
- [1. 模块整体架构](#1-模块整体架构)
- [2. 自动模型加载机制](#2-自动模型加载机制)
- [3. Embedder 实现详解](#3-embedder-实现详解)
- [4. Reranker 实现详解](#4-reranker-实现详解)

---

## 1. 模块整体架构

推理模块位于 `/workspace/FlagEmbedding/inference/` 目录下，主要包含以下核心组件：

```
inference/
├── auto_embedder.py          # Embedder 自动加载器
├── auto_reranker.py          # Reranker 自动加载器
├── embedder/                 # Embedder 实现
│   ├── encoder_only/         # 编码器-only 模型
│   │   ├── base.py           # 基础实现
│   │   └── m3.py             # BGE-M3 特殊实现
│   ├── decoder_only/         # 解码器-only 模型
│   │   ├── base.py           # 基础实现
│   │   ├── icl.py            # ICL 实现
│   │   └── pseudo_moe.py     # 伪 MoE 实现
│   ├── model_mapping.py      # Embedder 模型映射配置
│   └── __init__.py
├── reranker/                 # Reranker 实现
│   ├── encoder_only/         # 编码器-only
│   ├── decoder_only/         # 解码器-only
│   │   ├── base.py
│   │   ├── layerwise.py      # 层-wise 实现
│   │   └── lightweight.py    # 轻量级实现
│   └── model_mapping.py      # Reranker 模型映射
└── __init__.py
```

### 1.1 核心抽象基类

推理模块继承自 `abc.inference` 模块中的抽象基类：
- `AbsEmbedder` - Embedder 基类
- `AbsReranker` - Reranker 基类

这些抽象基类定义了统一的接口，使得不同类型的模型可以共享相同的推理流程。

---

## 2. 自动模型加载机制

FlagEmbedding 提供了强大的自动模型加载功能，无需手动选择具体的模型类。

### 2.1 FlagAutoModel - Embedder 自动加载器

**文件位置**：`/workspace/FlagEmbedding/inference/auto_embedder.py`

`FlagAutoModel` 是 Embedder 的统一入口点，它通过 `from_finetuned` 静态方法加载模型：

```python
class FlagAutoModel:
    @classmethod
    def from_finetuned(
        cls,
        model_name_or_path: str,
        model_class: Optional[Union[str, EmbedderModelClass]] = None,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        use_bf16: bool = False,
        query_instruction_for_retrieval: Optional[str] = None,
        devices: Optional[Union[str, List[str]]] = None,
        pooling_method: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        query_instruction_format: Optional[str] = None,
        truncate_dim: Optional[int] = None,
        **kwargs,
    ):
        # 模型名解析逻辑...
        # 参数默认值处理...
        # 返回实例化的 Embedder
```

**核心功能**：

1. **模型名解析**：从路径中提取模型名（处理 checkpoint 情况）
2. **自动类选择**：通过 `model_mapping` 找到匹配的模型类
3. **参数默认值处理**：从配置中获取默认值（pooling_method, trust_remote_code, query_instruction_format）
4. **实例化**：返回正确的 Embedder 实例

**示例用法**：
```python
from FlagEmbedding import FlagAutoModel

# 自动加载 BGE-M3 模型
model = FlagAutoModel.from_finetuned("BAAI/bge-m3")
```

### 2.2 FlagAutoReranker - Reranker 自动加载器

**文件位置**：`/workspace/FlagEmbedding/inference/auto_reranker.py`

`FlagAutoReranker` 与 `FlagAutoModel` 类似，用于 Reranker 的自动加载：

```python
class FlagAutoReranker:
    @classmethod
    def from_finetuned(
        cls,
        model_name_or_path: str,
        model_class: Optional[Union[str, RerankerModelClass]] = None,
        use_fp16: bool = False,
        trust_remote_code: Optional[bool] = None,
        **kwargs,
    ):
        # 类似的自动加载逻辑...
```

### 2.3 Embedder 模型映射

**文件位置**：`/workspace/FlagEmbedding/inference/embedder/model_mapping.py`

模型映射配置定义了模型名到具体实现类的映射：

#### 2.3.1 EmbedderModelClass 枚举

```python
class EmbedderModelClass(Enum):
    ENCODER_ONLY_BASE = "encoder-only-base"
    ENCODER_ONLY_M3 = "encoder-only-m3"
    DECODER_ONLY_BASE = "decoder-only-base"
    DECODER_ONLY_ICL = "decoder-only-icl"
    DECODER_ONLY_PSEUDO_MOE = "decoder-only-pseudo_moe"
```

#### 2.3.2 EmbedderConfig 数据类

```python
@dataclass
class EmbedderConfig:
    model_class: Type[AbsEmbedder]
    pooling_method: PoolingMethod
    trust_remote_code: bool = False
    query_instruction_format: str = "{}{}"
```

#### 2.3.3 PoolingMethod 枚举

```python
class PoolingMethod(Enum):
    LAST_TOKEN = "last_token"
    CLS = "cls"
    MEAN = "mean"
```

#### 2.3.4 支持的模型列表

`AUTO_EMBEDDER_MAPPING` 包含了多个系列的预训练模型：

| 模型系列 | 支持的模型 |
|---------|----------|
| BGE | `bge-m3`, `bge-large-en-v1.5`, `bge-base-en-v1.5`, `bge-small-en-v1.5`, `bge-large-zh-v1.5` 等 |
| Qwen3-Embedding | `Qwen3-Embedding-0.6B`, `Qwen3-Embedding-4B`, `Qwen3-Embedding-8B` |
| E5 | `e5-mistral-7b-instruct`, `e5-large-v2`, `multilingual-e5-large` 等 |
| GTE | `gte-Qwen2-7B-instruct`, `gte-large-en-v1.5`, `gte-large-zh` 等 |
| SFR | `SFR-Embedding-2_R`, `SFR-Embedding-Mistral` |

### 2.4 Reranker 模型映射

**文件位置**：`/workspace/FlagEmbedding/inference/reranker/model_mapping.py`

#### 2.4.1 RerankerModelClass 枚举

```python
class RerankerModelClass(Enum):
    ENCODER_ONLY_BASE = "encoder-only-base"
    DECODER_ONLY_BASE = "decoder-only-base"
    DECODER_ONLY_LAYERWISE = "decoder-only-layerwise"
    DECODER_ONLY_LIGHTWEIGHT = "decoder-only-lightweight"
```

#### 2.4.2 RerankerConfig 数据类

```python
@dataclass
class RerankerConfig:
    model_class: Type[AbsReranker]
    trust_remote_code: bool = False
```

#### 2.4.3 支持的 Reranker 模型

| 模型 | 类型 |
|-----|-----|
| `bge-reranker-base` | 编码器-only |
| `bge-reranker-large` | 编码器-only |
| `bge-reranker-v2-m3` | 编码器-only |
| `bge-reranker-v2-gemma` | 解码器-only |
| `bge-reranker-v2-minicpm-layerwise` | 解码器-only layerwise |
| `bge-reranker-v2.5-gemma2-lightweight` | 解码器-only lightweight |

---

## 3. Embedder 实现详解

### 3.1 编码器-only 基础实现 - BaseEmbedder

**文件位置**：`/workspace/FlagEmbedding/inference/embedder/encoder_only/base.py`

`BaseEmbedder` 是所有编码器-only Embedder 的基础实现，继承自 `AbsEmbedder`。

#### 3.1.1 初始化

```python
class BaseEmbedder(AbsEmbedder):
    DEFAULT_POOLING_METHOD = "cls"
    
    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        use_bf16: bool = False,
        query_instruction_for_retrieval: Optional[str] = None,
        query_instruction_format: str = "{}{}",
        devices: Optional[Union[str, List[str]]] = None,
        pooling_method: str = "cls",
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
        batch_size: int = 256,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        convert_to_numpy: bool = True,
        truncate_dim: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(...)
        self.pooling_method = pooling_method
        
        # 加载 tokenizer 和 model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            dtype=self.get_model_torch_dtype(),
        )
```

**初始化要点**：
- 支持多种精度（FP32, FP16, BF16）
- 可配置 pooling 方法（默认 CLS）
- 自动加载 tokenizer 和模型

#### 3.1.2 核心方法 - encode_single_device

这是单个设备上推理的核心实现，包含多个优化技巧：

```python
@torch.no_grad()
def encode_single_device(
    self,
    sentences: Union[List[str], str],
    batch_size: int = 256,
    max_length: int = 512,
    convert_to_numpy: bool = True,
    device: Optional[str] = None,
    **kwargs: Any
):
    # 1. 设备设置
    if device is None:
        device = self.target_devices[0]
    if device == "cpu":
        self.model.float()
    self.model.to(device)
    self.model.eval()
    
    # 2. 预处理：不分词填充，获取正确长度
    all_inputs = []
    for start_index in trange(0, len(sentences), batch_size, desc='pre tokenize'):
        sentences_batch = sentences[start_index:start_index + batch_size]
        inputs_batch = self.tokenizer(
            sentences_batch,
            truncation=True,
            max_length=max_length,
            **kwargs
        )
        # 转换为列表字典格式
        inputs_batch = [{
            k: inputs_batch[k][i] for k in inputs_batch.keys()
        } for i in range(len(sentences_batch))]
        all_inputs.extend(inputs_batch)
    
    # 3. 按长度排序优化（减少 padding）
    length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
    all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]
    
    # 4. 自适应调整 batch 大小（处理 OOM）
    flag = False
    while flag is False:
        try:
            inputs_batch = self.tokenizer.pad(
                all_inputs_sorted[: batch_size],
                padding=True,
                return_tensors='pt',
                **kwargs
            ).to(device)
            last_hidden_state = self.model(**inputs_batch, return_dict=True).last_hidden_state
            embeddings = self.pooling(last_hidden_state, inputs_batch['attention_mask'])
            flag = True
        except RuntimeError as e:
            batch_size = batch_size * 3 // 4  # 每次减少到原来的 75%
        except torch.cuda.OutOfMemoryError as e:
            batch_size = batch_size * 3 // 4
    
    # 5. 批量推理
    all_embeddings = []
    for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings"):
        inputs_batch = all_inputs_sorted[start_index:start_index + batch_size]
        inputs_batch = self.tokenizer.pad(
            inputs_batch,
            padding=True,
            return_tensors='pt',
            **kwargs
        ).to(device)
        
        last_hidden_state = self.model(**inputs_batch, return_dict=True).last_hidden_state
        embeddings = self.pooling(last_hidden_state, inputs_batch['attention_mask'])
        embeddings = self._truncate_embeddings(embeddings)
        
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
        embeddings = cast(torch.Tensor, embeddings)
        if convert_to_numpy:
            embeddings = self._convert_to_numpy(embeddings, device=device)
        all_embeddings.append(embeddings)
    
    # 6. 合并结果并恢复原始顺序
    if convert_to_numpy:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
    else:
        all_embeddings = torch.cat(all_embeddings, dim=0)
    
    all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]
    
    # 7. 返回结果
    if input_was_string:
        return all_embeddings[0]
    return all_embeddings
```

**关键优化点**：

1. **预 tokenization**：先不进行 padding，获取每个样本的真实长度
2. **长度排序**：按长度降序排序，使同一 batch 内的样本长度相近，减少 padding
3. **自适应 batch size**：遇到 OOM 时自动缩小 batch 大小（75%）
4. **归一化**：可选 L2 归一化
5. **维度截断**：支持 truncate_dim 进行维度裁剪

#### 3.1.3 Pooling 策略

```python
def pooling(
    self,
    last_hidden_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
):
    if self.pooling_method == 'cls':
        return last_hidden_state[:, 0]  # 取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token
    elif self.pooling_method == 'mean':
        # 均值 pooling，考虑 attention mask
        s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        return s / d
    else:
        raise NotImplementedError(f"pooling method {self.pooling_method} not implemented")
```

**支持的 pooling 方法**：
- `cls` - 使用 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token 表示整个序列
- `mean` - 使用所有有效 token 的平均值

### 3.2 BGE-M3 特殊实现 - M3Embedder

**文件位置**：`/workspace/FlagEmbedding/inference/embedder/encoder_only/m3.py`

`M3Embedder` 是为 BGE-M3 模型设计的特殊实现，支持多粒度、多功能的 embedding。

#### 3.2.1 核心特性

BGE-M3 支持三种输出：
- **Dense Embedding** - 密集向量表示
- **Sparse Embedding** - 稀疏词汇权重
- **ColBERT** - 多向量表示（每个 token 的 embedding）

#### 3.2.2 encode_single_device

```python
@torch.no_grad()
def encode_single_device(
    self,
    sentences: Union[List[str], str],
    batch_size: int = 256,
    max_length: int = 512,
    return_dense: bool = True,
    return_sparse: bool = False,
    return_colbert_vecs: bool = False,
    device: Optional[str] = None,
    **kwargs: Any
):
    # ... 设备设置、预处理、排序逻辑同 BaseEmbedder ...
    
    # 推理
    all_dense_embeddings, all_lexical_weights, all_colbert_vecs = [], [], []
    for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings"):
        inputs_batch = all_inputs_sorted[start_index:start_index + batch_size]
        inputs_batch = self.tokenizer.pad(
            inputs_batch,
            padding=True,
            return_tensors='pt',
            **kwargs
        ).to(device)
        
        outputs = self.model(
            inputs_batch,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert_vecs,
            truncate_dim=self.truncate_dim
        )
        
        # 处理 dense embedding
        if return_dense:
            all_dense_embeddings.append(
                self._convert_to_numpy(outputs['dense_vecs'], device=device)
            )
        
        # 处理 sparse embedding（lexical weights）
        if return_sparse:
            token_weights = outputs['sparse_vecs'].squeeze(-1)
            all_lexical_weights.extend(
                list(map(
                    _process_token_weights, 
                    self._convert_to_numpy(token_weights, device=device),
                    self._convert_to_numpy(inputs_batch['input_ids'], device=device).tolist()
                ))
            )
        
        # 处理 colbert 向量
        if return_colbert_vecs:
            all_colbert_vecs.extend(
                list(map(
                    _process_colbert_vecs,
                    self._convert_to_numpy(outputs['colbert_vecs'], device=device),
                    self._convert_to_numpy(inputs_batch['attention_mask'], device=device)
                ))
            )
    
    # 合并结果并恢复顺序...
    return {
        "dense_vecs": all_dense_embeddings,
        "lexical_weights": all_lexical_weights,
        "colbert_vecs": all_colbert_vecs
    }
```

**辅助函数**：

1. `_process_token_weights` - 处理稀疏权重
   - 过滤特殊 token（<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>、SEP、PAD、UNK）
   - 将 token id 转换为实际 token
   - 只保留正权重

2. `_process_colbert_vecs` - 处理 ColBERT 向量
   - 移除 padding 对应的向量
   - 不使用 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 的 embedding

#### 3.2.3 相关性分数计算 - compute_score

BGE-M3 可以直接计算查询和文档的相关性分数：

```python
def compute_score(
    self,
    sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
    batch_size: Optional[int] = None,
    max_query_length: Optional[int] = None,
    max_passage_length: Optional[int] = None,
    weights_for_different_modes: Optional[List[float]] = None,
    **kwargs: Any
) -> Dict[
    Literal["colbert", "sparse", "dense", "sparse+dense", "colbert+sparse+dense"],
    List[float]
]:
    # 单设备或多设备分发
    if len(self.target_devices) == 1:
        return self.compute_score_single_device(...)
    else:
        return self.compute_score_multi_process(...)
```

**compute_score_single_device**：
```python
@torch.no_grad()
def compute_score_single_device(
    self,
    sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
    batch_size: int = 256,
    max_query_length: int = 512,
    max_passage_length: int = 512,
    weights_for_different_modes: Optional[List[float]] = None,
    device: Optional[str] = None,
    **kwargs: Any
):
    # 1. 分别编码查询和文档
    queries_inputs = _tokenize([pair[0] for pair in sentences_batch], max_query_length)
    passages_inputs = _tokenize([pair[1] for pair in sentences_batch], max_passage_length)
    
    # 2. 获取三种类型的输出
    queries_output = self.model(
        queries_inputs,
        return_dense=True, return_sparse=True, return_colbert_vecs=True,
        return_sparse_embedding=True
    )
    passages_output = self.model(
        passages_inputs,
        return_dense=True, return_sparse=True, return_colbert_vecs=True,
        return_sparse_embedding=True
    )
    
    # 3. 计算三种分数
    dense_scores = self.model.compute_dense_score(q_dense, p_dense)
    sparse_scores = self.model.compute_sparse_score(q_sparse, p_sparse)
    colbert_scores = self.model.compute_colbert_score(q_colbert, p_colbert, q_mask)
    
    # 4. 加权融合
    if weights_for_different_modes is None:
        weights_for_different_modes = [1., 1., 1.]  # [dense, sparse, colbert]
        weight_sum = 3
    
    # 5. 返回多种组合结果
    return {
        'colbert': colbert_scores,
        'sparse': sparse_scores,
        'dense': dense_scores,
        'sparse+dense': (sparse * w1 + dense * w0) / (w0 + w1),
        'colbert+sparse+dense': (colbert * w2 + sparse * w1 + dense * w0) / sum(weights)
    }
```

### 3.3 解码器-only 基础实现 - BaseLLMEmbedder

**文件位置**：`/workspace/FlagEmbedding/inference/embedder/decoder_only/base.py`

`BaseLLMEmbedder` 是解码器-only 模型的基础实现，使用 last token pooling。

#### 3.3.1 Last Token Pooling

```python
def last_token_pool(last_hidden_states: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
    # 检测是否是左 padding
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]  # 左 padding，取最后一个 token
    else:
        # 右 padding，取最后一个有效 token
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]
```

**设计要点**：
- 自动检测左/右 padding 格式
- 取最后一个有效 token 作为句子表示

### 3.4 ICL 实现 - ICLLLMEmbedder

**文件位置**：`/workspace/FlagEmbedding/inference/embedder/decoder_only/icl.py`

`ICLLLMEmbedder` 支持 Few-Shot 示例，提高模型在特定任务上的表现。

#### 3.4.1 核心特性

```python
def set_examples(self, examples_for_task: Optional[List[dict]] = None):
    """设置 Few-Shot 示例前缀"""
    if examples_for_task is None and self.examples_for_task is None:
        self.prefix = ''
    else:
        # 格式化每个示例
        eg_paris = []
        for example in examples_for_task or self.examples_for_task:
            eg_paris.append(
                self.get_detailed_example(
                    self.examples_instruction_format,
                    example.get('instruct', self.query_instruction_for_retrieval),
                    example.get('query', ''),
                    example.get('response', '')
                )
            )
        self.prefix = '\n\n'.join(eg_paris) + '\n\n'
```

#### 3.4.2 编码流程

```python
@torch.no_grad()
def encode_queries_single_device(
    self,
    queries: Union[List[str], str],
    batch_size: int = 256,
    max_length: int = 512,
    convert_to_numpy: bool = True,
    device: Optional[str] = None,
    **kwargs: Any
):
    # 1. 添加查询指令
    if self.query_instruction_for_retrieval is not None:
        input_texts = [
            self.get_detailed_instruct(
                self.query_instruction_format,
                self.query_instruction_for_retrieval,
                query
            ) for query in queries
        ]
    
    # 2. 拼接前缀 + 文本 + 后缀
    prefix_ids = self.tokenizer(self.prefix, add_special_tokens=False)['input_ids']
    suffix_ids = self.tokenizer(self.suffix, add_special_tokens=False)['input_ids']
    
    # 3. 计算新的最大长度
    new_max_length = (len(prefix_ids) + len(suffix_ids) + max_length + 8) // 8 * 8 + 8
    
    # 4. 编码（类似 BaseLLMEmbedder）...
```

**格式**：`[前缀] + [查询文本] + [后缀]`

### 3.5 伪 MoE 实现 - PseudoMoELLMEmbedder

**文件位置**：`/workspace/FlagEmbedding/inference/embedder/decoder_only/pseudo_moe.py`

`PseudoMoELLMEmbedder` 支持伪 MoE（Mixture of Experts）模型，可以在推理时选择不同的域。

#### 3.5.1 核心功能

```python
def _resolve_domain(self, kwargs: Any) -> Optional[str]:
    """从 kwargs 或默认值中解析域"""
    domain = kwargs.pop("domain_for_pseudo_moe", None)
    if domain is None:
        domain = kwargs.pop("domain", None)
    if domain is None:
        domain = self.domain_for_pseudo_moe
    return domain

@torch.no_grad()
def encode_single_device(
    self,
    sentences: Union[List[str], str],
    batch_size: int = 256,
    max_length: int = 512,
    convert_to_numpy: bool = True,
    device: Optional[str] = None,
    **kwargs: Any
):
    # 解析域
    domain = self._resolve_domain(kwargs)
    
    # 设置模型的域
    if domain is not None and hasattr(self.model, "set_domain"):
        self.model.set_domain(domain)
    
    # 准备 forward 的参数
    model_forward_kwargs = {"return_dict": True}
    if domain is not None:
        model_forward_kwargs["domain"] = domain
    
    # 推理（传入 domain 参数）
    last_hidden_state = self.model(**inputs_batch, **model_forward_kwargs).last_hidden_state
    # ...
```

---

## 4. Reranker 实现详解

### 4.1 编码器-only 基础实现 - BaseReranker

**文件位置**：`/workspace/FlagEmbedding/inference/reranker/encoder_only/base.py`

`BaseReranker` 是编码器-only Reranker 的基础实现，使用序列分类模型。

#### 4.1.1 核心方法 - compute_score_single_gpu

```python
@torch.no_grad()
def compute_score_single_gpu(
    self,
    sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
    batch_size: Optional[int] = None,
    query_max_length: Optional[int] = None,
    max_length: Optional[int] = None,
    normalize: Optional[bool] = None,
    device: Optional[str] = None,
    **kwargs: Any
) -> List[float]:
    # 1. 预处理
    all_inputs = []
    for start_index in trange(0, len(sentence_pairs), batch_size, desc="pre tokenize"):
        sentences_batch = sentence_pairs[start_index:start_index + batch_size]
        queries = [s[0] for s in sentences_batch]
        passages = [s[1] for s in sentences_batch]
        
        # 分别 tokenize 查询和文档
        queries_inputs_batch = self.tokenizer(
            queries,
            return_tensors=None,
            add_special_tokens=False,
            max_length=query_max_length,
            truncation=True,
            **kwargs
        )['input_ids']
        passages_inputs_batch = self.tokenizer(
            passages,
            return_tensors=None,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
            **kwargs
        )['input_ids']
        
        # 拼接：<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> + query + SEP + passage + SEP
        for q_inp, d_inp in zip(queries_inputs_batch, passages_inputs_batch):
            item = self.tokenizer.prepare_for_model(
                q_inp,
                d_inp,
                truncation='only_second',  # 只截断文档
                max_length=max_length,
                padding=False,
            )
            all_inputs.append(item)
    
    # 2. 按长度排序
    length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
    all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]
    
    # 3. 自适应 batch 大小
    flag = False
    while flag is False:
        try:
            test_inputs_batch = self.tokenizer.pad(all_inputs_sorted[:min(...)]).to(device)
            scores = self.model(**test_inputs_batch, return_dict=True).logits.view(-1, ).float()
            flag = True
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            batch_size = batch_size * 3 // 4
    
    # 4. 批量推理
    all_scores = []
    for start_index in tqdm(range(0, len(all_inputs_sorted), batch_size), desc="Compute Scores"):
        inputs = self.tokenizer.pad(all_inputs_sorted[start_index:start_index + batch_size]).to(device)
        scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
        all_scores.extend(scores.cpu().numpy().tolist())
    
    # 5. 恢复顺序
    all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]
    
    # 6. Sigmoid 归一化
    if normalize:
        all_scores = [sigmoid(score) for score in all_scores]
    
    return all_scores
```

**输入格式**：`<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 查询 [SEP] 文档 [SEP]`

### 4.2 解码器-only 基础实现 - BaseLLMReranker

**文件位置**：`/workspace/FlagEmbedding/inference/reranker/decoder_only/base.py`

`BaseLLMReranker` 使用 LLM 作为 Reranker，通过预测 "Yes" 或 "No" 的概率来表示相关性。

#### 4.2.1 输入格式

```python
# 默认提示词
if prompt is None:
    prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."

# 完整输入格式
item['input_ids'] = [bos_token] + query_inputs + sep + passage_inputs + sep + prompt_inputs
```

**格式**：`[BOS] 查询 [SEP] 文档 [SEP] 提示词`

#### 4.2.2 Last Logit Pooling

```python
def last_logit_pool(logits: Tensor,
                   attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return logits[:, -1]  # 左 padding，取最后一个 token
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return torch.stack([
            logits[i, sequence_lengths[i], :] 
            for i in range(batch_size)
        ], dim=0)
```

**分数计算**：取最后一个 token 对 "Yes" 的 logit 作为相关性分数

```python
self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
scores = last_logit_pool(logits, inputs['attention_mask'])
scores = scores[:, self.yes_loc]  # 取 'Yes' 的分数
```

### 4.3 层-wise 实现 - LayerWiseLLMReranker

**文件位置**：`/workspace/FlagEmbedding/inference/reranker/decoder_only/layerwise.py`

`LayerWiseLLMReranker` 支持使用多个层的输出来计算分数，而不仅仅是最后一层。

#### 4.3.1 核心特性

```python
def compute_score_single_gpu(
    self,
    sentence_pairs: ...,
    cutoff_layers: Optional[List[int]] = None,  # 选择哪些层
    ...
):
    # 推理时传入 cutoff_layers
    outputs = self.model(
        **batch_inputs,
        output_hidden_states=True,
        cutoff_layers=cutoff_layers  # 指定要使用的层
    )
    
    # 获取多个层的 logits
    all_logits = outputs.logits  # List[Tensor]
    tmp_all_scores = []
    for logits in all_logits:
        scores = last_logit_pool_layerwise(logits, inputs['attention_mask'])
        tmp_all_scores.append(scores.contiguous())
    
    # 返回每个层的分数
    all_scores = []
    for i in range(len(tmp_all_scores)):
        all_scores.append(tmp_all_scores[i].cpu().float().tolist())
    
    # 恢复顺序
    for i in range(len(all_scores)):
        all_scores[i] = [all_scores[i][idx] for idx in np.argsort(length_sorted_idx)]
        if normalize:
            all_scores[i] = [sigmoid(score) for score in all_scores[i]]
    
    return all_scores
```

**输出**：每个指定层的分数列表

### 4.4 轻量级实现 - LightweightLLMReranker

**文件位置**：`/workspace/FlagEmbedding/inference/reranker/decoder_only/lightweight.py`

`LightweightLLMReranker` 是轻量级 Reranker 实现，支持层压缩以提高效率。

#### 4.4.1 核心特性

```python
def __init__(
    self,
    ...,
    compress_layers: List[int] = [8],  # 要压缩的层
    compress_ratio: int = 1,  # 压缩比例 1, 2, 4, 8
    ...
):
    # 使用 CostWiseGemmaForCausalLM
    self.model = CostWiseGemmaForCausalLM.from_pretrained(...)

def compute_score_single_gpu(
    self,
    ...,
    compress_layers: Optional[List[int]] = None,
    compress_ratio: Optional[int] = None,
    ...
):
    # 传入压缩参数
    outputs = self.model(
        **batch_inputs,
        output_hidden_states=True,
        compress_layer=compress_layers,
        compress_ratio=compress_ratio,
        query_lengths=query_lengths,
        prompt_lengths=prompt_lengths,
        cutoff_layers=cutoff_layers
    )
    
    # 处理多个输出
    scores = []
    for i in range(len(outputs.logits)):
        logits = last_logit_pool_lightweight(
            outputs.logits[i], 
            outputs.attention_masks[i]
        )
        scores.append(logits.cpu().float().tolist())
```

---

## 5. 关键技术总结

### 5.1 优化技巧

| 优化技术 | 应用场景 | 实现位置 |
|---------|---------|---------|
| 长度排序优化 Padding | 批量推理 | BaseEmbedder.encode_single_device |
| 自适应 Batch Size | OOM 处理 | BaseEmbedder.encode_single_device |
| Last Token Pooling | Decoder-only 模型 | BaseLLMEmbedder |
| 多粒度输出 | BGE-M3 | M3Embedder |
| 提示词工程 | LLM Reranker | BaseLLMReranker |

### 5.2 代码引用

- [auto_embedder.py](/workspace/FlagEmbedding/inference/auto_embedder.py)
- [auto_reranker.py](/workspace/FlagEmbedding/inference/auto_reranker.py)
- [embedder/model_mapping.py](/workspace/FlagEmbedding/inference/embedder/model_mapping.py)
- [embedder/encoder_only/base.py](/workspace/FlagEmbedding/inference/embedder/encoder_only/base.py)
- [embedder/encoder_only/m3.py](/workspace/FlagEmbedding/inference/embedder/encoder_only/m3.py)
- [embedder/decoder_only/base.py](/workspace/FlagEmbedding/inference/embedder/decoder_only/base.py)
- [reranker/encoder_only/base.py](/workspace/FlagEmbedding/inference/reranker/encoder_only/base.py)
- [reranker/decoder_only/base.py](/workspace/FlagEmbedding/inference/reranker/decoder_only/base.py)
- [reranker/decoder_only/layerwise.py](/workspace/FlagEmbedding/inference/reranker/decoder_only/layerwise.py)
- [reranker/decoder_only/lightweight.py](/workspace/FlagEmbedding/inference/reranker/decoder_only/lightweight.py)
