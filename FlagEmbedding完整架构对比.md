
# FlagEmbedding 完整架构对比

本文档对 FlagEmbedding 项目中的所有核心实现进行全面对比分析。

---

## 目录结构概览

```
inference/
├── embedder/
│   ├── encoder_only/
│   │   └── base.py              # BaseEmbedder
│   └── decoder_only/
│       ├── base.py              # BaseLLMEmbedder
│       └── icl.py               # ICLLLMEmbedder
└── reranker/
    ├── encoder_only/
    │   └── base.py              # BaseReranker
    └── decoder_only/
        ├── base.py              # BaseLLMReranker
        ├── layerwise.py         # LayerWiseLLMReranker
        └── lightweight.py       # LightweightLLMReranker
```

---

## 一、核心类对比表

| 类名 | 所在文件 | 继承自 | 架构类型 | 主要功能 |
|------|---------|--------|---------|---------|
| **BaseEmbedder** | embedder/encoder_only/base.py | AbsEmbedder | Encoder-Only | 生成文本嵌入向量 |
| **BaseLLMEmbedder** | embedder/decoder_only/base.py | AbsEmbedder | Decoder-Only | 生成文本嵌入向量（LLM） |
| **ICLLLMEmbedder** | embedder/decoder_only/icl.py | AbsEmbedder | Decoder-Only | 支持ICL的嵌入生成 |
| **BaseReranker** | reranker/encoder_only/base.py | AbsReranker | Encoder-Only | 计算查询-段落相关性 |
| **BaseLLMReranker** | reranker/decoder_only/base.py | AbsReranker | Decoder-Only | LLM重排序 |
| **LayerWiseLLMReranker** | reranker/decoder_only/layerwise.py | AbsReranker | Decoder-Only | 层-wise重排序 |
| **LightweightLLMReranker** | reranker/decoder_only/lightweight.py | AbsReranker | Decoder-Only | 轻量级重排序 |

---

## 二、Embedder 系列详细对比

### 2.1 Embedder 初始化对比

| 特性 | BaseEmbedder | BaseLLMEmbedder | ICLLLMEmbedder |
|------|-------------|----------------|---------------|
| **模型类** | `AutoModel` | `AutoModel` | `AutoModel` |
| **Pooling方法** | CLS/Mean可选 | last_token强制 | last_token强制 |
| **Default Pooling** | `cls` | `last_token` | `last_token` |
| **支持ICL** | ❌ | ❌ | ✅ |
| **PEFT支持** | ❌ | ❌ | ✅ |
| **默认BatchSize** | 256 | 256 | 256 |
| **默认FP16** | True | True | True |

### 2.2 Pooling 实现对比

#### BaseEmbedder - CLS/Mean Pooling
```python
def pooling(self, last_hidden_state, attention_mask):
    if self.pooling_method == 'cls':
        return last_hidden_state[:, 0]
    elif self.pooling_method == 'mean':
        s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        return s / d
```

#### BaseLLMEmbedder/ICLLLMEmbedder - Last Token Pooling
```python
def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
```

### 2.3 ICLLLMEmbedder 特殊功能

**核心参数：**
- `examples_for_task`: Few-shot示例列表
- `examples_instruction_format`: 示例格式模板
- `suffix`: 后缀（通常为`\n&lt;response&gt;`）

**输入拼接格式：**
```
&lt;BOS&gt; [Few-shot示例1]
&lt;instruct&gt;...&lt;/instruct&gt;
&lt;query&gt;...&lt;/query&gt;
&lt;response&gt;...&lt;/response&gt;

[Few-shot示例2]
...

&lt;instruct&gt;任务指令&lt;/instruct&gt;
&lt;query&gt;实际查询&lt;/query&gt;
&lt;response&gt; &lt;EOS&gt;
                ↑
        取这个位置的embedding
```

---

## 三、Reranker 系列详细对比

### 3.1 Reranker 初始化对比

| 特性 | BaseReranker | BaseLLMReranker | LayerWiseLLMReranker | LightweightLLMReranker |
|------|-------------|----------------|---------------------|------------------------|
| **模型类** | `AutoModelForSequenceClassification` | `AutoModelForCausalLM` | `LayerWiseMiniCPMForCausalLM` | `CostWiseGemmaForCausalLM` |
| **输出方式** | 直接logits | Yes-token logits | 多层logits | 多层logits（带压缩） |
| **特殊Token** | 标准特殊token | BOS（如果存在） | BOS | BOS |
| **Prompt默认值** | ❌ 无 | "Given a query A...Yes or No." | "Given a query A...Yes or No." | "Predict whether passage B..." |
| **Yes-loc** | ❌ | ✅ | ❌ | ❌ |
| **层选择** | ❌ | ❌ | `cutoff_layers` | `cutoff_layers` |
| **层压缩** | ❌ | ❌ | ❌ | `compress_layers`, `compress_ratio` |
| **PEFT支持** | ❌ | ✅ | ✅ | ✅ |
| **默认BatchSize** | 128 | 128 | 128 | 128 |

### 3.2 输入拼接格式对比

#### Encoder-Only Reranker (BaseReranker)
```python
# 分开tokenize，然后用prepare_for_model拼接
item = self.tokenizer.prepare_for_model(
    query_inputs,
    passage_inputs,
    truncation='only_second',  # 只截断passage
    max_length=max_length
)
# 直接输入到模型，输出logits
```

#### Decoder-Only Reranker (所有变体)
```python
# 拼接格式
&lt;BOS&gt; A: [查询]\n B: [段落]\n [Prompt] &lt;EOS&gt;
                                      ↑
                             取这个位置的score

# 代码实现
item = self.tokenizer.prepare_for_model(
    [bos_id] + query_inputs,
    sep_inputs + passage_inputs,
    truncation='only_second'
)
item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
```

### 3.3 Score 计算方式对比

| 类型 | Score计算方式 | 归一化 |
|------|--------------|--------|
| **BaseReranker** | `model(**inputs).logits.view(-1)` | Sigmoid |
| **BaseLLMReranker** | `model(**inputs).logits[:, yes_loc]` | Sigmoid |
| **LayerWiseLLMReranker** | `[model(**inputs).logits[i] for i in layers]` | Sigmoid（每层） |
| **LightweightLLMReranker** | `[model(**inputs).logits[i] for i in layers]` | Sigmoid（每层） |

#### BaseLLMReranker Yes-Token 方式
```python
self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
# ...
scores = last_logit_pool(logits, attention_mask)
scores = scores[:, self.yes_loc]  # 只取"Yes" token的logit
```

### 3.4 特殊功能详解

#### LayerWiseLLMReranker
- **cutoff_layers**: 指定使用哪些层的输出
- **输出**: 多层的score列表（而不是单个score）

#### LightweightLLMReranker
- **compress_layers**: 选择要压缩的层
- **compress_ratio**: 压缩比（1, 2, 4, 8）
- **query_lengths/prompt_lengths**: 用于优化计算
- **特殊模型**: `CostWiseGemmaForCausalLM`

---

## 四、数据结构与辅助类

### 4.1 Embedder 相关

Embedder 系列没有额外的复杂辅助类，主要使用：
- `AutoTokenizer` / `AutoModel`
- 内部 `encode_single_device()` 方法处理batch

### 4.2 Decoder-Only Reranker 辅助类

#### DatasetForReranker (用于BaseLLMReranker)
```python
class DatasetForReranker(Dataset):
    """
    当use_dataloader=True时使用
    预拼接query-passage-prompt，支持多进程加载
    """
    def __getitem__(self, item):
        # &lt;BOS&gt;A:query\nB:passage\n[Prompt]
        return {
            'input_ids': ...,
            'attention_mask': ...
        }
```

#### Collater 系列
| Collater类 | 用于 | 特点 |
|-----------|------|------|
| **Collater** | BaseLLMReranker | 标准padding，pad_to_multiple_of=8 |
| **Collater_for_lightweight** | LightweightLLMReranker | 额外处理query_lengths/prompt_lengths |

---

## 五、完整推理流程对比

### 5.1 Encoder-Only Embedder 流程
```
1. Tokenize 每个句子（独立）
   ↓
2. 按长度排序（减少padding）
   ↓
3. 批量前向传播
   ↓
4. Pooling (CLS/Mean)
   ↓
5. 可选：截断/归一化
   ↓
6. 恢复原始顺序，输出embedding
```

### 5.2 Decoder-Only Embedder 流程
```
1. Tokenize 每个句子
   ↓
2. 按长度排序
   ↓
3. 批量前向传播
   ↓
4. Last-Token Pooling
   ↓
5. 恢复顺序，输出embedding
```

### 5.3 ICL Embedder 额外步骤
```
在步骤1前：
  对每个查询，拼接前缀（Few-shot示例）+ 查询 + 后缀
```

### 5.4 Encoder-Only Reranker 流程
```
1. 分别Tokenize queries和passages
   ↓
2. 两两拼接（prepare_for_model）
   ↓
3. 按总长度排序
   ↓
4. 批量前向传播
   ↓
5. 直接取logits作为score
   ↓
6. 恢复顺序，输出scores
```

### 5.5 Decoder-Only Reranker 流程
```
1. 分别Tokenize queries和passages
   ↓
2. 拼接: &lt;BOS&gt;A:q\nB:p\n[Prompt]
   ↓
3. 按长度排序
   ↓
4. 批量前向传播
   ↓
5. Last-Token Pooling + Yes-token选择
   ↓
6. 恢复顺序，输出scores
```

---

## 六、关键代码片段索引

### Embedder 核心方法
| 方法 | 位置 | 功能 |
|------|------|------|
| `encode_single_device` | embedder/encoder_only/base.py#L175 | Encoder-Only编码 |
| `encode_single_device` | embedder/decoder_only/base.py#L194 | Decoder-Only编码 |
| `encode_queries_single_device` | embedder/decoder_only/icl.py#L323 | ICL查询编码 |
| `set_examples` | embedder/decoder_only/icl.py#L133 | 设置Few-shot示例 |
| `pooling` | embedder/encoder_only/base.py#L284 | CLS/Mean pooling |
| `last_token_pool` | embedder/decoder_only/base.py#L15 | Last-token pooling |

### Reranker 核心方法
| 方法 | 位置 | 功能 |
|------|------|------|
| `compute_score_single_gpu` | reranker/encoder_only/base.py#L78 | Encoder-Only重排序 |
| `compute_score_single_gpu` | reranker/decoder_only/base.py#L257 | Decoder-Only重排序 |
| `compute_score_single_gpu` | reranker/decoder_only/layerwise.py#L136 | Layer-wise重排序 |
| `compute_score_single_gpu` | reranker/decoder_only/lightweight.py#L206 | 轻量级重排序 |
| `last_logit_pool` | reranker/decoder_only/base.py#L15 | LLM score pooling |

---

## 七、模型选择指南

### 7.1 Embedder 选择
| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 通用检索 | BGE-* (Encoder-Only) | 速度快、成熟 |
| 已有LLM资源 | BGE-Multilingual-Gemma (Decoder-Only) | 复用LLM |
| 需要Few-shot适应 | BGE-EN-ICL | 支持ICL |

### 7.2 Reranker 选择
| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 标准重排序 | BGE-Reranker-* (Encoder-Only) | 速度快、成熟 |
| 追求极致性能 | BGE-Reranker-Llama3 (Decoder-Only) | 利用LLM能力 |
| 预算有限 | BGE-Reranker-Lightweight (Decoder-Only) | 支持层压缩 |
| 需要多层输出 | BGE-Reranker-LayerWise (Decoder-Only) | 可输出多层score |

---

## 八、两阶段检索标准流程

```
阶段 1: 粗检索（Embedder）
┌─────────────────────────────────────┐
│  对所有文档编码 → 存入向量数据库    │
│  查询编码 → 快速检索 Top-100         │
└─────────────────────────────────────┘
           ↓

阶段 2: 精排序（Reranker）
┌─────────────────────────────────────┐
│  对Top-100文档两两计算相关性         │
│  按score重新排序 → 输出最终结果      │
└─────────────────────────────────────┘
```

---

## 总结

FlagEmbedding 项目提供了灵活多样的实现：
- **Encoder-Only**: 快速、成熟，适合大部分场景
- **Decoder-Only**: 利用LLM能力，支持更多高级特性（ICL、层输出等）
- **Embedder + Reranker**: 两阶段方案兼顾速度和精度

