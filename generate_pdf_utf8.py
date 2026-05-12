#!/usr/bin/env python3
"""
FlagEmbedding Documentation PDF Generator - UTF-8 Chinese Support Version
修复中文乱码问题
"""

import os
import re
import base64
import subprocess
from pathlib import Path
from markdown import Markdown
import tempfile
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 读取所有 Markdown 文件
def read_markdown_files(directory):
    """按顺序读取所有 Markdown 文件"""
    md_files = [
        '01_overview.md',
        '02_abc_layer.md',
        '03_inference_module.md',
        '04_finetune_module.md',
        '05_evaluation_module.md',
        '06_key_algorithms.md',
        '07_research_projects.md',
        '08_model_survey.md'
    ]
    
    contents = []
    for md_file in md_files:
        file_path = Path(directory) / md_file
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                contents.append((md_file, content))
                print(f"✓ Loaded {md_file}")
        else:
            print(f"✗ Warning: {md_file} not found")
    
    return contents

def draw_simple_diagram(mermaid_code, output_path):
    """使用 matplotlib 绘制简化的图表"""
    try:
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('white')
        
        # 简单显示为示意图
        ax.text(7, 4, 'Mermaid Diagram', fontsize=20, ha='center', 
                weight='bold', color='#1976d2')
        
        ax.text(7, 3, '(See original Markdown for details)', 
                fontsize=10, ha='center', color='#666')
        
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        plt.tight_layout(pad=2)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Diagram drawing error: {e}")
        return False

def process_mermaid_blocks(content, output_dir, counter):
    """处理 Markdown 中的 Mermaid 代码块，保留原文"""
    pattern = r'```mermaid\n(.*?)```'
    
    def replace_mermaid(match):
        nonlocal counter
        mermaid_code = match.group(1)
        
        # 生成唯一的文件名
        img_path = os.path.join(output_dir, f'mermaid_{counter:03d}.png')
        counter += 1
        
        # 保留原文，同时添加图片（如果渲染可用）
        # 这里我们只显示原文，避免乱码问题
        return f'\n```mermaid\n{mermaid_code}```\n'
    
    processed_content = re.sub(pattern, replace_mermaid, content, flags=re.DOTALL)
    return processed_content, counter

def create_css():
    """创建 PDF 样式，确保中文支持"""
    return """
    @charset "UTF-8";
    
    @font-face {
        font-family: 'Noto Sans CJK SC';
        font-style: normal;
        font-weight: normal;
    }
    
    @page {
        size: A4;
        margin: 1.5cm 2cm;
    }
    
    body {
        font-family: "Noto Sans CJK SC", "Noto Sans CJK", "Source Han Sans CN", 
                     "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei", 
                     "DejaVu Sans", sans-serif;
        font-size: 11pt;
        line-height: 1.8;
        color: #2c3e50;
        text-align: justify;
    }
    
    h1 {
        font-size: 22pt;
        color: #1976d2;
        border-bottom: 3px solid #1976d2;
        padding-bottom: 8px;
        margin-top: 40px;
        margin-bottom: 20px;
        page-break-after: avoid;
        text-align: left;
        font-family: "Noto Sans CJK SC", "Noto Sans CJK", "Source Han Sans CN", 
                     "Microsoft YaHei", sans-serif;
    }
    
    h2 {
        font-size: 16pt;
        color: #1565c0;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 5px;
        margin-top: 30px;
        margin-bottom: 15px;
        page-break-after: avoid;
        font-family: "Noto Sans CJK SC", "Noto Sans CJK", "Source Han Sans CN", 
                     "Microsoft YaHei", sans-serif;
    }
    
    h3 {
        font-size: 13pt;
        color: #333;
        margin-top: 20px;
        margin-bottom: 10px;
        page-break-after: avoid;
        font-family: "Noto Sans CJK SC", "Noto Sans CJK", "Source Han Sans CN", 
                     "Microsoft YaHei", sans-serif;
    }
    
    h4 {
        font-size: 11pt;
        color: #555;
        margin-top: 15px;
        margin-bottom: 8px;
        page-break-after: avoid;
        font-family: "Noto Sans CJK SC", "Noto Sans CJK", "Source Han Sans CN", 
                     "Microsoft YaHei", sans-serif;
    }
    
    img {
        max-width: 100%;
        max-height: 400px;
        height: auto;
        display: block;
        margin: 25px auto;
        page-break-inside: avoid;
        border: 1px solid #ddd;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        page-break-inside: avoid;
        font-size: 9pt;
    }
    
    th, td {
        border: 1px solid #bdbdbd;
        padding: 8px 10px;
        text-align: left;
        vertical-align: top;
    }
    
    th {
        background-color: #e3f2fd;
        font-weight: bold;
        color: #1565c0;
    }
    
    tr:nth-child(even) {
        background-color: #fafafa;
    }
    
    tr:hover {
        background-color: #f5f5f5;
    }
    
    code {
        font-family: "Source Code Pro", "Consolas", "Monaco", monospace;
        background-color: #f5f5f5;
        padding: 1px 4px;
        border-radius: 3px;
        font-size: 8.5pt;
        color: #c7254e;
    }
    
    pre {
        background-color: #263238;
        color: #aed581;
        border-radius: 5px;
        padding: 12px;
        overflow-x: auto;
        page-break-inside: avoid;
        margin: 15px 0;
        font-size: 8pt;
        line-height: 1.5;
    }
    
    pre code {
        background-color: transparent;
        padding: 0;
        color: inherit;
    }
    
    blockquote {
        border-left: 4px solid #1976d2;
        margin: 20px 0;
        padding: 10px 15px;
        background-color: #e3f2fd;
        color: #0d47a1;
    }
    
    ul, ol {
        margin: 10px 0;
        padding-left: 25px;
    }
    
    li {
        margin: 6px 0;
    }
    
    hr {
        border: none;
        border-top: 2px solid #e0e0e0;
        margin: 40px 0;
    }
    
    strong {
        font-weight: bold;
        color: #c62828;
    }
    
    em {
        font-style: italic;
        color: #1565c0;
    }
    
    .cover {
        text-align: center;
        padding: 150px 50px;
        page-break-after: always;
    }
    
    .document {
        margin-bottom: 30px;
    }
    """

def merge_and_convert_to_pdf(input_dir, output_pdf):
    """合并 Markdown 文件并转换为 PDF"""
    print("\n" + "="*60)
    print("FlagEmbedding Documentation PDF Generator - UTF-8")
    print("="*60 + "\n")
    
    # 创建输出目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 读取所有 Markdown 文件
        print("1. Reading Markdown files...")
        md_contents = read_markdown_files(input_dir)
        
        if not md_contents:
            print("Error: No Markdown files found!")
            return False
        
        # 处理每个文件
        print("\n2. Processing content...")
        processed_contents = []
        global_counter = 0
        
        for filename, content in md_contents:
            print(f"   Processing {filename}...")
            processed, global_counter = process_mermaid_blocks(content, temp_dir, global_counter)
            processed_contents.append((filename, processed))
        
        # 创建合并的 HTML
        print("\n3. Generating HTML with UTF-8 support...")
        
        html_parts = []
        doc_num = 0
        
        for filename, content in processed_contents:
            doc_num += 1
            # 转换为 HTML
            md = Markdown(extensions=['tables', 'fenced_code', 'codehilite', 'nl2br', 'sane_lists'])
            html_content = md.convert(content)
            
            # 文档标题
            doc_title = filename.replace('.md', '').replace('_', ' ').replace('-', ' ')
            
            html_parts.append(f'''
            <div class="document" id="doc-{doc_num}">
                <h1>{doc_title}</h1>
                {html_content}
                <hr/>
            </div>
            ''')
        
        # 生成完整的 HTML
        full_html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlagEmbedding 代码分析文档</title>
    <style>
        {create_css()}
    </style>
</head>
<body>
    <div class="cover">
        <h1 style="font-size: 42pt; color: #1976d2; border: none; margin-bottom: 20px;">
            ⚡ FlagEmbedding
        </h1>
        <h2 style="font-size: 24pt; color: #333; border: none; margin-bottom: 30px;">
            代码分析文档
        </h2>
        <p style="font-size: 14pt; color: #666; margin: 30px 0;">
            One-Stop Retrieval Toolkit For Search and RAG
        </p>
        <hr style="width: 50%; margin: 50px auto;">
        <p style="font-size: 12pt; color: #888;">
            北京智源人工智能研究院 (BAAI)
        </p>
        <p style="font-size: 10pt; color: #aaa; margin-top: 100px;">
            基于源代码的深度技术分析
        </p>
    </div>
    
    <div class="content">
        {''.join(html_parts)}
    </div>
</body>
</html>'''
        
        # 保存 HTML 文件
        html_path = os.path.join(temp_dir, 'content.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        print(f"   ✓ HTML saved to: {html_path}")
        
        # 尝试使用不同的方法生成 PDF
        print("\n4. Converting to PDF with UTF-8 support...")
        
        # 方法 1: 使用 wkhtmltopdf（如果可用）
        try:
            print("   Trying wkhtmltopdf...")
            result = subprocess.run([
                'wkhtmltopdf', '--encoding', 'UTF-8', 
                '--enable-local-file-access', html_path, output_pdf
            ], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"\n{'='*60}")
                print(f"✓ SUCCESS! PDF generated with wkhtmltopdf:")
                print(f"  {output_pdf}")
                print(f"{'='*60}")
                return True
        except Exception as e:
            print(f"   wkhtmltopdf not available: {e}")
        
        # 方法 2: 使用 WeasyPrint
        try:
            print("   Trying WeasyPrint...")
            from weasyprint import HTML, CSS
            
            HTML(filename=html_path).write_pdf(output_pdf)
            print(f"\n{'='*60}")
            print(f"✓ SUCCESS! PDF generated with WeasyPrint:")
            print(f"  {output_pdf}")
            print(f"{'='*60}")
            return True
            
        except Exception as e:
            print(f"WeasyPrint error: {e}")
        
        # 方法 3: 如果都失败，保存 HTML 文件让用户自己转换
        print("\n   Saving HTML file for manual conversion...")
        html_output = output_pdf.replace('.pdf', '.html')
        shutil.copy(html_path, html_output)
        print(f"   ✓ HTML file saved to: {html_output}")
        print("\n   Please open the HTML file in a browser and print to PDF.")
        print("   Make sure to set 'Print Background Graphics' option if needed.")
        return False
        
    finally:
        # 清理临时文件
        print("\n5. Cleanup...")
        try:
            shutil.rmtree(temp_dir)
            print("   ✓ Temporary files cleaned")
        except:
            pass

if __name__ == '__main__':
    # 配置路径
    input_directory = '/workspace/ReadCode'
    output_file = '/workspace/FlagEmbedding_Analysis_UTF8.pdf'
    
    # 运行转换
    success = merge_and_convert_to_pdf(input_directory, output_file)
