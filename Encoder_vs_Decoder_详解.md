
# Encoder-Only 与 Decoder-Only 架构对比详解

本文档详细对比 FlagEmbedding 中 `encoder_only` 和 `decoder_only` 两种架构的区别、特点及使用场景。

---

## 一、核心区别速览

| 维度 | Encoder-Only (编码器架构) | Decoder-Only (解码器架构) |
|------|--------------------------|--------------------------|
| **代表模型** | BGE-v1.5, E5, GTE | BGE-Multilingual-Gemma2, Qwen3-Embedding, E5-Mistral |
| **模型来源** | BERT 类、RoBERTa 类 | LLaMA 类、Gemma 类、Qwen 类等大语言模型 |
| **注意力机制** | 双向注意力（自注意力） | 单向/因果注意力（只能看前面的 token） |
| **Pooling 方法** | `CLS`、`Mean` | 仅 `last_token` |
| **默认 Pooling** | `CLS` | `last_token` |
| **查询指令格式** | 简单格式 `"{}{}"` | 结构化格式 `Instruct: {}\nQuery: {}"` |
| **特殊能力** | 可支持多粒度输出（如 BGE-M3） | 可支持 Few-shot ICL、伪 MoE 等 |
| **实现类** | [BaseEmbedder](file:///workspace/FlagEmbedding/inference/embedder/encoder_only/base.py) | [BaseLLMEmbedder](file:///workspace/FlagEmbedding/inference/embedder/decoder_only/base.py) |

---

## 二、代码实现对比

### 2.1 Encoder-Only 架构 - [BaseEmbedder](file:///workspace/FlagEmbedding/inference/embedder/encoder_only/base.py)

#### 核心特点：

```python
class BaseEmbedder(AbsEmbedder):
    DEFAULT_POOLING_METHOD = "cls"  # 默认 CLS pooling
    
    def __init__(...):
        # 支持多种 pooling 方法
        self.pooling_method = pooling_method  # 可选 'cls' 或 'mean'
        
    def pooling(self, last_hidden_state, attention_mask):
        """支持两种 pooling 策略"""
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]  # 取第一个 token (<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>)
        elif self.pooling_method == 'mean':
            # 对所有有效 token 取平均
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
```

#### 输入示例：
```
<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 这是一个句子。 [SEP]
↑
取这个 token 的表示作为句子 embedding
```

---

### 2.2 Decoder-Only 架构 - [BaseLLMEmbedder](file:///workspace/FlagEmbedding/inference/embedder/decoder_only/base.py)

#### 核心特点：

```python
class BaseLLMEmbedder(AbsEmbedder):
    DEFAULT_POOLING_METHOD = "last_token"  # 仅支持最后一个 token
    
    def __init__(...):
        # 强制检查 pooling 方法
        if self.kwargs.get("pooling_method", "last_token") != "last_token":
            raise ValueError("Pooling method must be 'last_token' for LLM-based models.")
    
    # 使用 last_token_pool 函数
    def last_token_pool(last_hidden_states, attention_mask):
        """取最后一个有效 token 的表示"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]  # 左 padding 时直接取最后一个
        else:
            # 右 padding 时，计算有效长度，取最后一个有效 token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device), 
                sequence_lengths
            ]
```

#### 输入示例：
```
&lt;s&gt; Instruct: 为这个句子生成表示
Query: 这是一个句子。 &lt;/s&gt;
                                      ↑
                                 取这个 token 的表示
```

---

### 2.3 Decoder-Only 的扩展版本 - [ICLLLMEmbedder](file:///workspace/FlagEmbedding/inference/embedder/decoder_only/icl.py)

这是 Decoder-Only 架构的增强版，支持 Few-shot ICL（上下文学习）：

```python
class ICLLLMEmbedder(AbsEmbedder):
    def __init__(...):
        # 支持 Few-shot 示例
        self.examples_for_task = examples_for_task
        self.examples_instruction_format = examples_instruction_format
        
    def set_examples(self, examples_for_task):
        """设置 Few-shot 示例前缀"""
        if examples_for_task is not None:
            eg_paris = []
            for example in examples_for_task:
                eg_paris.append(
                    self.get_detailed_example(
                        self.examples_instruction_format,
                        example.get('instruct', ...),
                        example.get('query', ''),
                        example.get('response', '')
                    )
                )
            self.prefix = '\n\n'.join(eg_paris) + '\n\n'
```

#### ICL 输入格式示例：
```
&lt;s&gt; &lt;instruct&gt;为查询生成表示&lt;/instruct&gt;
&lt;query&gt;什么是人工智能？&lt;/query&gt;
&lt;response&gt;人工智能是...&lt;/response&gt;

&lt;instruct&gt;为查询生成表示&lt;/instruct&gt;
&lt;query&gt;用户的新问题&lt;/query&gt;
&lt;response&gt; &lt;/s&gt;
                                      ↑
                                 取这个位置的表示
```

---

## 三、Pooling 策略详细对比

### 3.1 Encoder-Only 的 Pooling

| Pooling 方法 | 原理 | 适用场景 |
|-------------|------|---------|
| **CLS** | 取 `<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>` 这个特殊 token 的隐藏状态作为句子表示 | BERT 类模型的标准用法，适合大多数检索任务 |
| **Mean** | 对所有有效 token（非 padding）的隐藏状态取平均 | 当 `<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>` 训练不充分时，或需要更均衡的表示时使用 |

#### 代码位置：
- [pooling 方法](file:///workspace/FlagEmbedding/inference/embedder/encoder_only/base.py#L284-L308)

---

### 3.2 Decoder-Only 的 Pooling

| Pooling 方法 | 原理 | 为什么这样设计？ |
|-------------|------|-----------------|
| **last_token** | 取最后一个有效 token 的隐藏状态 | 1. Decoder-Only 是单向注意力，最后一个 token 包含了完整的上下文信息&lt;br&gt;2. 这是 LLM 做 embedding 的标准做法 |

#### 代码位置：
- [last_token_pool 函数](file:///workspace/FlagEmbedding/inference/embedder/decoder_only/base.py#L12-L33)

#### 为什么 Decoder-Only 只能用 last_token？

```mermaid
graph TD
    A[Decoder-Only 的注意力] --&gt; B[单向/因果注意力]
    B --&gt; C["token_i 只能看到 token_0 ~ token_i"]
    C --&gt; D[最后一个 token 看到的信息最多]
    D --&gt; E[所以取最后一个 token 的表示]
```

---

## 四、模型映射表

从 [model_mapping.py](file:///workspace/FlagEmbedding/inference/embedder/model_mapping.py) 可以看到：

### Encoder-Only 模型：
- `bge-m3`, `bge-large-en-v1.5`, `bge-base-en-v1.5`, `bge-small-en-v1.5`
- `e5-large-v2`, `multilingual-e5-large`
- `gte-large-en-v1.5`, `gte-large-zh`

### Decoder-Only 模型：
- `bge-multilingual-gemma2`
- `bge-en-icl` (支持 ICL)
- `Qwen3-Embedding-0.6B/4B/8B`
- `e5-mistral-7b-instruct`
- `gte-Qwen2-7B-instruct`
- `SFR-Embedding-2_R`, `SFR-Embedding-Mistral`
- `Linq-Embed-Mistral`

---

## 五、查询指令格式对比

### Encoder-Only 的查询格式（简单）：
```python
# 来自 model_mapping.py
query_instruction_format: str = "{}{}"  # 简单拼接
```
示例：
```
"为这个句子生成表示以用于检索相关文章：这是一个查询"
```

### Decoder-Only 的查询格式（结构化）：
```python
# 来自 model_mapping.py
query_instruction_format: str = "Instruct: {}\nQuery: {}"  # 或 "<instruct>{}\n<query>{}"
```
示例：
```
"Instruct: 为这个句子生成表示\nQuery: 这是一个查询"
```
或
```
"&lt;instruct&gt;为这个句子生成表示&lt;/instruct&gt;\n&lt;query&gt;这是一个查询&lt;/query&gt;"
```

---

## 六、使用场景推荐

### 什么时候用 Encoder-Only？
✅ **需要高吞吐量、低延迟** - Encoder-Only 模型通常更小更快  
✅ **标准检索任务** - 如文档检索、问答系统等  
✅ **资源受限场景** - 适合在边缘设备或资源有限的环境部署  
✅ **不需要复杂指令跟随** - 任务相对简单直接  

### 什么时候用 Decoder-Only？
✅ **需要更好的指令跟随能力** - LLM 类模型对指令理解更好  
✅ **需要 Few-shot 能力** - 如 `bge-en-icl` 支持提供示例来调整表示  
✅ **多语言复杂任务** - 大语言模型通常对多语言支持更好  
✅ **长文本理解** - 某些 Decoder-Only 模型支持更长的上下文  

---

## 七、代码使用示例

### Encoder-Only 模型使用：
```python
from FlagEmbedding import FlagAutoModel

# 加载 Encoder-Only 模型
model = FlagAutoModel.from_finetuned(
    'BAAI/bge-small-en-v1.5',
    normalize_embeddings=True,
    pooling_method='cls'  # 可选 'cls' 或 'mean'
)

# 编码文本
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "FlagEmbedding is a powerful embedding toolkit."
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (2, 384)
```

### Decoder-Only 模型使用：
```python
from FlagEmbedding import FlagAutoModel

# 加载 Decoder-Only 模型
model = FlagAutoModel.from_finetuned(
    'BAAI/bge-multilingual-gemma2',
    normalize_embeddings=True
    # pooling_method 只能是 'last_token'，无需指定
)

# 编码文本
sentences = [
    "这是一个中文句子。",
    "This is an English sentence."
]

embeddings = model.encode(sentences)
print(embeddings.shape)
```

### Decoder-Only + ICL 使用：
```python
from FlagEmbedding import FlagAutoModel

# 加载支持 ICL 的模型
model = FlagAutoModel.from_finetuned(
    'BAAI/bge-en-icl',
    normalize_embeddings=True
)

# 设置 Few-shot 示例
examples = [
    {
        'instruct': 'Generate a representation for this sentence for retrieval',
        'query': 'What is machine learning?',
        'response': 'Machine learning is a subfield of AI.'
    },
    {
        'instruct': 'Generate a representation for this sentence for retrieval',
        'query': 'How does deep learning work?',
        'response': 'Deep learning uses neural networks.'
    }
]

model.set_examples(examples)

# 编码查询
query = "What is natural language processing?"
embedding = model.encode_queries([query])
```

---

## 八、总结

| 特性 | Encoder-Only | Decoder-Only |
|------|-------------|-------------|
| **速度** | ⚡ 通常更快 | 🐢 通常较慢（模型更大） |
| **内存** | 💾 通常更省内存 | 💾 通常需要更多内存 |
| **指令理解** | 👍 一般 | 🌟 更好 |
| **Few-shot** | ❌ 不支持 | ✅ 部分支持（如 bge-en-icl） |
| **Pooling 选择** | ✅ 灵活（CLS/Mean） | ⚠️ 仅 last_token |
| **适用场景** | 通用检索 | 复杂/多语言/需要指令的任务 |

---

## 代码参考链接

- [encoder_only/base.py](file:///workspace/FlagEmbedding/inference/embedder/encoder_only/base.py) - Encoder-Only 实现
- [decoder_only/base.py](file:///workspace/FlagEmbedding/inference/embedder/decoder_only/base.py) - Decoder-Only 基础实现
- [decoder_only/icl.py](file:///workspace/FlagEmbedding/inference/embedder/decoder_only/icl.py) - Decoder-Only + ICL 实现
- [model_mapping.py](file:///workspace/FlagEmbedding/inference/embedder/model_mapping.py) - 模型映射配置
