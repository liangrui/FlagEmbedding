
"""
FlagEmbedding 推理模块核心代码详解与示例

这个文件详细解释了 03_inference_module.md 中提到的核心推理流程，
特别是这四段关键代码：
    1. last_hidden_state = self.model(**inputs_batch, return_dict=True).last_hidden_state
    2. embeddings = self.pooling(last_hidden_state, inputs_batch['attention_mask'])
    3. embeddings = self._truncate_embeddings(embeddings)
    4. if self.normalize_embeddings: embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
"""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer


class SimpleEmbedder:
    """
    简化版 Embedder，用于演示核心推理流程
    """
    
    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-small-en-v1.5",
        normalize_embeddings: bool = True,
        pooling_method: str = "cls",
        truncate_dim: int = None
    ):
        print(f"[初始化] 加载模型: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method
        self.truncate_dim = truncate_dim
        self.model.eval()
    
    def encode(self, sentences):
        """
        完整的编码流程演示
        """
        print("\n" + "="*80)
        print("步骤 1: Tokenization（分词）")
        print("="*80)
        
        inputs_batch = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        print(f"\n输入文本: {sentences}")
        print(f"\nTokenized 结果:")
        print(f"  input_ids 形状: {inputs_batch['input_ids'].shape}")
        print(f"  attention_mask 形状: {inputs_batch['attention_mask'].shape}")
        print(f"  input_ids 内容 (前5个token): {inputs_batch['input_ids'][0][:5]}")
        print(f"  解码 tokens: {self.tokenizer.convert_ids_to_tokens(inputs_batch['input_ids'][0])}")
        
        print("\n" + "="*80)
        print("步骤 2: 模型前向传播（获取 last_hidden_state）")
        print("="*80)
        
        with torch.no_grad():
            outputs = self.model(**inputs_batch, return_dict=True)
        
        last_hidden_state = outputs.last_hidden_state
        print(f"\nlast_hidden_state 形状: {last_hidden_state.shape}")
        print(f"  含义: [batch_size, sequence_length, hidden_size]")
        print(f"  每个 token 都有一个 {last_hidden_state.shape[-1]} 维的向量表示")
        print(f"  <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token 向量 (前5维): {last_hidden_state[0, 0, :5]}")
        
        print("\n" + "="*80)
        print("步骤 3: Pooling（池化）")
        print("="*80)
        
        if self.pooling_method == 'cls':
            print(f"\n使用 CLS pooling: 取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token (index 0) 的向量")
            embeddings = last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            print(f"\n使用 Mean pooling: 计算所有有效 token 的平均值")
            mask = inputs_batch['attention_mask'].unsqueeze(-1)
            s = torch.sum(last_hidden_state * mask.float(), dim=1)
            d = mask.sum(dim=1).float()
            embeddings = s / d
        
        print(f"Pooling 后形状: {embeddings.shape}")
        print(f"Embedding 向量 (前10维): {embeddings[0, :10]}")
        
        print("\n" + "="*80)
        print("步骤 4: 维度截断（可选）")
        print("="*80)
        
        if self.truncate_dim is not None and self.truncate_dim < embeddings.shape[-1]:
            print(f"\n截断维度: {embeddings.shape[-1]} -&gt; {self.truncate_dim}")
            embeddings = embeddings[:, :self.truncate_dim]
        else:
            print("\n不进行维度截断")
        
        print(f"截断后形状: {embeddings.shape}")
        
        print("\n" + "="*80)
        print("步骤 5: 归一化（可选）")
        print("="*80)
        
        if self.normalize_embeddings:
            print(f"\n进行 L2 归一化")
            norm_before = torch.norm(embeddings[0])
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            norm_after = torch.norm(embeddings[0])
            print(f"  归一化前的 L2 范数: {norm_before:.4f}")
            print(f"  归一化后的 L2 范数: {norm_after:.4f}")
        else:
            print("\n不进行归一化")
        
        print(f"\n最终 embedding 向量 (前10维): {embeddings[0, :10]}")
        
        return embeddings


def demonstrate_similarity_search():
    """
    演示如何使用 embeddings 进行相似度搜索
    """
    print("\n" + "#"*80)
    print("相似度搜索示例")
    print("#"*80)
    
    embedder = SimpleEmbedder(
        model_name_or_path="BAAI/bge-small-en-v1.5",
        normalize_embeddings=True
    )
    
    corpus = [
        "The cat sits on the mat.",
        "A dog runs in the park.",
        "Machine learning is interesting.",
        "Python is a programming language.",
        "Deep learning uses neural networks."
    ]
    
    query = "What is artificial intelligence?"
    
    print(f"\n语料库: {corpus}")
    print(f"\n查询: {query}")
    
    # 编码语料库
    print("\n--- 编码语料库 ---")
    corpus_embeddings = embedder.encode(corpus)
    
    # 编码查询
    print("\n--- 编码查询 ---")
    query_embedding = embedder.encode([query])
    
    # 计算相似度
    print("\n" + "="*80)
    print("步骤 6: 计算相似度")
    print("="*80)
    
    # 因为已经归一化，直接点积就是余弦相似度
    similarities = torch.matmul(query_embedding, corpus_embeddings.T)[0]
    
    print(f"\n查询与各文档的相似度:")
    for i, (text, sim) in enumerate(zip(corpus, similarities)):
        print(f"  文档 {i+1}: {text}")
        print(f"    相似度: {sim:.4f}")
    
    # 排序
    sorted_indices = torch.argsort(similarities, descending=True)
    print(f"\n最相关的文档:")
    for rank, idx in enumerate(sorted_indices):
        print(f"  第 {rank+1} 名: 文档 {idx+1} (相似度: {similarities[idx]:.4f})")


def compare_pooling_methods():
    """
    比较不同 pooling 方法的效果
    """
    print("\n" + "#"*80)
    print("Pooling 方法对比")
    print("#"*80)
    
    sentences = ["Hello world, how are you?"]
    
    print("\n--- CLS Pooling ---")
    embedder_cls = SimpleEmbedder(pooling_method='cls', normalize_embeddings=False)
    emb_cls = embedder_cls.encode(sentences)
    
    print("\n--- Mean Pooling ---")
    embedder_mean = SimpleEmbedder(pooling_method='mean', normalize_embeddings=False)
    emb_mean = embedder_mean.encode(sentences)
    
    # 计算两个 embedding 的相似度
    sim = torch.nn.functional.cosine_similarity(emb_cls, emb_mean)
    print(f"\nCLS 和 Mean pooling 的余弦相似度: {sim[0]:.4f}")


if __name__ == "__main__":
    print("#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*15 + "FlagEmbedding 推理模块详解示例" + " "*36 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    # 1. 基础编码流程演示
    print("\n" + "="*80)
    print("示例 1: 基础编码流程")
    print("="*80)
    
    embedder = SimpleEmbedder(
        model_name_or_path="BAAI/bge-small-en-v1.5",
        normalize_embeddings=True,
        pooling_method='cls',
        truncate_dim=None
    )
    
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "FlagEmbedding is a powerful embedding toolkit."
    ]
    
    embeddings = embedder.encode(sentences)
    
    # 2. 相似度搜索示例
    demonstrate_similarity_search()
    
    # 3. Pooling 方法对比
    compare_pooling_methods()
    
    print("\n" + "#"*80)
    print("总结")
    print("#"*80)
    print("""
核心推理流程总结：
1. Tokenization: 将文本转换为 token IDs
2. 模型前向传播: 获取 last_hidden_state [batch, seq_len, hidden_size]
3. Pooling: 
   - CLS: 取第一个 token 的向量
   - Mean: 取所有有效 token 的平均
4. 维度截断（可选）: 裁剪 embedding 维度以提高效率
5. 归一化（可选）: L2 归一化使向量长度为 1，便于计算余弦相似度

关键代码位置:
- /workspace/FlagEmbedding/inference/embedder/encoder_only/base.py
""")
