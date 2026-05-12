# FlagEmbedding 支持的嵌入及重排模型综述 - 创作计划

## 1. 综述目的
撰写一篇全面的综述文章，介绍 FlagEmbedding 库支持的所有嵌入（Embedder）和重排序（Reranker）模型，包括它们的特点、性能、适用场景等。

## 2. 综述结构大纲

### 2.1 摘要/简介
- FlagEmbedding 的项目背景
- 本文综述的目标
- 主要内容概览

### 2.2 嵌入模型（Embedder）综述
#### 2.2.1 BGE 系列模型
- BGE v1.0 系列
- BGE v1.5 系列
- BGE-M3 模型（重点介绍）
  - 多功能特性（dense + sparse + colbert）
  - 多语言支持
  - 性能特点
- BGE-CODE 模型
- BGE-Gemma2 多语言模型
- BGE-ICL 模型
- BGE-Qwen3 8B 模型

#### 2.2.2 Qwen3-Embedding 系列
- 0.6B / 4B / 8B 三个规格
- 特点和适用场景

#### 2.2.3 E5 系列模型
- 基础 E5 模型
- E5 v2 模型
- E5-Mistral-7B 模型
- Multilingual E5 系列

#### 2.2.4 GTE 系列模型
- 基础 GTE 模型
- GTE-Qwen 系列
- GTE v1.5 模型
- GTE 多语言模型

#### 2.2.5 SFR 系列模型
- SFR-Embedding-Mistral
- SFR-Embedding-2_R

#### 2.2.6 Linq 模型
- Linq-Embed-Mistral

#### 2.2.7 BCE 模型
- BCE-Embedding-Base

### 2.3 重排序模型（Reranker）综述
#### 2.3.1 BGE 系列重排序器
- bge-reranker-base
- bge-reranker-large
- bge-reranker-v2-m3
- bge-reranker-v2-gemma
- bge-reranker-v2-minicpm-layerwise
- bge-reranker-v2.5-gemma2-lightweight

#### 2.3.2 其他重排序模型
- Jina 系列
- GTE 多语言
- BCE 重排序器

### 2.4 模型分类与对比
#### 2.4.1 按架构分类
- Encoder-only 模型
- Decoder-only 模型
- Layer-wise 模型
- Lightweight 模型

#### 2.4.2 按功能分类
- 基础嵌入模型
- 多功能嵌入模型（M3）
- 上下文学习模型（ICL）
- 代码特定模型
- 多语言模型

#### 2.4.3 性能对比
- 检索性能
- 推理速度
- 内存占用

### 2.5 选择指南
- 根据场景选择模型
- 平衡性能与资源
- 最佳实践建议

### 2.6 总结与展望
- FlagEmbedding 模型生态总结
- 未来发展方向
- 对社区的贡献

## 3. 技术实现细节（从代码中提取）

### 3.1 嵌入模型的 Pooling 策略
- CLS pooling
- Mean pooling
- Last token pooling

### 3.2 查询指令格式
- 不同模型的查询格式
- 影响与最佳实践

### 3.3 模型加载机制
- AutoModel 自动选择
- 模型映射表
- Trust remote code 设置

## 4. 数据来源

主要从以下文件提取信息：
- `/workspace/FlagEmbedding/inference/embedder/model_mapping.py`
- `/workspace/FlagEmbedding/inference/reranker/model_mapping.py`
- `/workspace/ReadCode/` 目录下的分析文档
- 项目 README 中的性能数据

## 5. 交付物

- Markdown 格式的综述文档
- 包含 Mermaid 图表的架构图
- 表格形式的对比数据
- 保存至 `/workspace/ReadCode/08_model_survey.md`

## 6. 执行步骤

1. 创建文档框架和目录
2. 收集和整理所有支持的模型信息
3. 撰写各章节内容
4. 添加图表和表格
5. 进行内容审核和优化
6. 保存最终文档
