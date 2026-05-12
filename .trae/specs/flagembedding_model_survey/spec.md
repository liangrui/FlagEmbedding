# FlagEmbedding 模型综述 - 产品需求文档

## Overview
- **Summary**: 完善并交付 FlagEmbedding 支持的嵌入及重排模型的完整综述文档，包含详细的图表、对比表格和高质量内容，最终生成 PDF 文档。
- **Purpose**: 为用户提供全面、结构化的 FlagEmbedding 模型生态指南，帮助用户理解和选择合适的模型。
- **Target Users**: NLP 开发者、检索系统工程师、研究人员、FlagEmbedding 用户

## Goals
- 完善和优化 08_model_survey.md 文档内容质量
- 确保所有 Mermaid 图表正确显示和渲染
- 生成完整美观的 PDF 文档
- 保持代码引用链接的正确性
- 提供清晰的模型对比和选择指南

## Non-Goals (Out of Scope)
- 不修改 FlagEmbedding 核心库代码
- 不实现新的模型训练功能
- 不提供 API 服务开发
- 不修改其他 ReadCode 文档

## Background & Context
- 已存在 08_model_survey.md 文档草稿
- 已包含基础模型系列介绍和部分图表
- 已存在其他 ReadCode 分析文档（01-07）
- 需要整合到最终 PDF 中

## Functional Requirements
- **FR-1**: 完善嵌入模型系列介绍，包含所有支持的模型
- **FR-2**: 完善重排序模型系列介绍，包含所有支持的模型
- **FR-3**: 添加完整的模型对比表格
- **FR-4**: 确保所有 Mermaid 图表正确渲染
- **FR-5**: 生成包含所有内容的高质量 PDF 文档
- **FR-6**: 保持所有代码引用链接的可访问性
- **FR-7**: 添加选择指南和最佳实践章节

## Non-Functional Requirements
- **NFR-1**: 文档结构清晰，易于理解
- **NFR-2**: 图表美观，信息丰富
- **NFR-3**: PDF 文档排版专业
- **NFR-4**: 所有链接正确工作
- **NFR-5**: 文档内容完整，覆盖所有支持的模型

## Constraints
- **Technical**: 使用 Markdown + Mermaid，使用 WeasyPrint 生成 PDF
- **Business**: 项目需在当前环境中完成，不依赖外部付费服务
- **Dependencies**: 依赖现有的 ReadCode/08_model_survey.md 文档和其他分析文档

## Assumptions
- 用户可以查看和使用生成的 PDF 文档
- Mermaid 图表可以通过外部工具或直接渲染
- 现有代码库文件路径保持不变

## Acceptance Criteria

### AC-1: 文档内容完整性
- **Given**: FlagEmbedding 模型综述项目
- **When**: 检查 08_model_survey.md 文档
- **Then**: 包含所有嵌入模型系列介绍（BGE/Qwen3/E5/GTE/SFR/Linq/BCE）
- **Verification**: `human-judgment`

### AC-2: 重排序模型覆盖
- **Given**: FlagEmbedding 模型综述项目
- **When**: 检查 08_model_survey.md 文档
- **Then**: 包含所有重排序模型系列介绍（BGE/Jina/GTE/BCE）
- **Verification**: `human-judgment`

### AC-3: 图表质量
- **Given**: 08_model_survey.md 文档
- **When**: 检查所有 Mermaid 图表
- **Then**: 图表设计合理、信息丰富、渲染正确
- **Verification**: `human-judgment`

### AC-4: 对比表格存在
- **Given**: 08_model_survey.md 文档
- **When**: 检查文档内容
- **Then**: 包含至少 3 个模型对比表格（架构/功能/性能）
- **Verification**: `human-judgment`

### AC-5: PDF 生成成功
- **Given**: 完善后的 Markdown 文档
- **When**: 执行 PDF 生成
- **Then**: 生成完整、美观的 PDF 文档
- **Verification**: `programmatic`

### AC-6: 代码引用链接
- **Given**: 08_model_survey.md 文档
- **When**: 检查所有代码引用链接
- **Then**: 所有链接格式正确、可访问
- **Verification**: `human-judgment`

### AC-7: 选择指南完整性
- **Given**: 08_model_survey.md 文档
- **When**: 检查选择指南章节
- **Then**: 包含场景选择、资源平衡、最佳实践建议
- **Verification**: `human-judgment`

## Open Questions
- [ ] 是否需要将所有 ReadCode 文档（01-07）都整合到最终 PDF？
- [ ] Mermaid 图表是使用在线渲染还是转换为图片？
- [ ] PDF 是否需要添加目录和导航？
