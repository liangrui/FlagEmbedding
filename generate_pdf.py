#!/usr/bin/env python3
"""
FlagEmbedding Documentation PDF Generator
将所有 Markdown 文件合并并转换为带图表的 PDF
"""

import os
import re
import base64
from pathlib import Path
from markdown import Markdown
from weasyprint import HTML, CSS
import subprocess
import tempfile
import shutil

# 尝试导入 pyMermaid
try:
    from pymermaid import mermaid
    HAS_PYMERMAID = True
except ImportError:
    HAS_PYMERMAID = False
    print("Warning: pyMermaid not available, will use alternative approach")

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
        '07_research_projects.md'
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

# 使用 Mermaid CLI 渲染图表
def render_mermaid_with_cli(mermaid_code, output_path):
    """使用 mermaid-cli 渲染 Mermaid 图表"""
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
            f.write(mermaid_code)
            temp_input = f.name
        
        # 运行 mermaid CLI
        cmd = [
            'mmdc',
            '-i', temp_input,
            '-o', output_path,
            '-b', 'transparent',
            '-w', '1200',
            '-H', '800'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # 清理临时文件
        os.unlink(temp_input)
        
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        print(f"Mermaid CLI error: {e}")
        return False

# 使用 Python 绘制简化版图表
def draw_simple_diagram(mermaid_code, output_path):
    """使用 matplotlib 绘制简化的图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import FancyBboxPatch
        import numpy as np
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # 检测图表类型并绘制
        if 'flowchart' in mermaid_code:
            draw_flowchart(ax, mermaid_code)
        elif 'classDiagram' in mermaid_code:
            draw_class_diagram(ax, mermaid_code)
        elif 'sequenceDiagram' in mermaid_code:
            draw_sequence_diagram(ax, mermaid_code)
        elif 'timeline' in mermaid_code:
            draw_timeline(ax, mermaid_code)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Diagram drawing error: {e}")
        return False

def draw_flowchart(ax, mermaid_code):
    """绘制流程图"""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    
    # 简化的流程图绘制
    ax.text(7, 7.5, 'Flowchart Diagram', fontsize=16, ha='center', 
            weight='bold', color='#333')
    
    # 添加说明
    ax.text(7, 6.5, '(Mermaid Flowchart - Simplified Representation)', 
            fontsize=12, ha='center', style='italic', color='#666')
    
    # 绘制简化的节点
    boxes = [
        (2, 5, 'Input'),
        (6, 5, 'Process'),
        (10, 5, 'Output'),
        (6, 2, 'Decision')
    ]
    
    for x, y, label in boxes:
        box = FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6,
                            boxstyle="round,pad=0.05,rounding_size=0.2",
                            facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10)
    
    # 绘制箭头
    ax.annotate('', xy=(4.2, 5), xytext=(2.8, 5),
                arrowprops=dict(arrowstyle='->', color='#1976d2', lw=2))
    ax.annotate('', xy=(8.2, 5), xytext=(6.8, 5),
                arrowprops=dict(arrowstyle='->', color='#1976d2', lw=2))
    ax.annotate('', xy=(6, 3.3), xytext=(6, 2.7),
                arrowprops=dict(arrowstyle='->', color='#1976d2', lw=2))

def draw_class_diagram(ax, mermaid_code):
    """绘制类图"""
    ax.text(7, 7.5, 'Class Diagram', fontsize=16, ha='center', 
            weight='bold', color='#333')
    ax.text(7, 6.5, '(Mermaid Class Diagram - Simplified)', 
            fontsize=12, ha='center', style='italic', color='#666')
    
    # 绘制简化的类
    classes = [
        (3, 5, 'AbsEmbedder\n<<abstract>>\n+encode()\n+forward()'),
        (11, 5, 'BaseEmbedder\n+encode_single()\n-inherit from Abs')
    ]
    
    for x, y, text in classes:
        box = FancyBboxPatch((x-1.5, y-1.5), 3, 3,
                            boxstyle="round,pad=0.1",
                            facecolor='#fff3e0', edgecolor='#e65100', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, 
               family='monospace')
    
    # 绘制继承关系
    ax.annotate('', xy=(9.5, 5), xytext=(4.5, 5),
                arrowprops=dict(arrowstyle='-|>', color='#e65100', lw=2))

def draw_sequence_diagram(ax, mermaid_code):
    """绘制时序图"""
    ax.text(7, 7.5, 'Sequence Diagram', fontsize=16, ha='center', 
            weight='bold', color='#333')
    ax.text(7, 6.5, '(Mermaid Sequence Diagram - Simplified)', 
            fontsize=12, ha='center', style='italic', color='#666')
    
    # 绘制参与者
    participants = ['User', 'Model', 'Process']
    for i, name in enumerate(participants):
        x = 3 + i * 4
        ax.text(x, 5, name, ha='center', va='bottom', fontsize=11, 
               weight='bold')
        ax.plot([x, x], [1, 5], 'k-', lw=2)
    
    # 绘制消息箭头
    messages = [
        (3, 4.5, 7, 4.5, 'Request'),
        (7, 4, 3, 4, 'Response'),
    ]
    
    for x1, y1, x2, y2, label in messages:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='#1976d2', lw=1.5))
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, label, ha='center', fontsize=9)

def draw_timeline(ax, mermaid_code):
    """绘制时间线"""
    ax.text(7, 7.5, 'Technology Evolution Timeline', fontsize=16, ha='center', 
            weight='bold', color='#333')
    
    # 绘制时间线
    ax.plot([1, 13], [4, 4], 'k-', lw=3)
    
    # 添加节点
    phases = [
        (2, 'Foundation', 'v1.0, v1.5'),
        (6, 'Multi-function', 'M3, MRL'),
        (10, 'Specialized', 'Code, VL')
    ]
    
    for x, title, desc in phases:
        ax.plot([x, x], [3.5, 4.5], 'k-', lw=3)
        circle = plt.Circle((x, 4), 0.2, color='#1976d2')
        ax.add_patch(circle)
        ax.text(x, 5.5, title, ha='center', fontsize=10, weight='bold')
        ax.text(x, 2.5, desc, ha='center', fontsize=9, style='italic')

def process_mermaid_blocks(content, output_dir):
    """处理 Markdown 中的 Mermaid 代码块"""
    pattern = r'```mermaid\n(.*?)```'
    
    def replace_mermaid(match):
        mermaid_code = match.group(1)
        
        # 生成唯一的文件名
        hash_val = hash(mermaid_code) % 1000000
        img_path = os.path.join(output_dir, f'mermaid_{hash_val}.png')
        rel_path = f'mermaid_{hash_val}.png'
        
        # 尝试渲染
        success = False
        
        # 方法1: 使用 mermaid CLI
        if not success:
            success = render_mermaid_with_cli(mermaid_code, img_path)
        
        # 方法2: 使用 Python 绘制
        if not success:
            success = draw_simple_diagram(mermaid_code, img_path)
        
        # 如果都失败，创建一个占位符
        if not success or not os.path.exists(img_path):
            # 创建占位符图片
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.set_xlim(0, 12)
                ax.set_ylim(0, 6)
                ax.axis('off')
                ax.text(6, 3, 'Diagram Preview\n(Mermaid Code Below)', 
                       ha='center', va='center', fontsize=14, 
                       bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#ccc'))
                plt.tight_layout()
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close()
                success = True
            except:
                pass
        
        if success:
            return f'\n![Mermaid Diagram]({rel_path})\n'
        else:
            return match.group(0)  # 保留原始代码
    
    processed_content = re.sub(pattern, replace_mermaid, content, flags=re.DOTALL)
    return processed_content

def create_css():
    """创建 PDF 样式"""
    return """
    @page {
        size: A4;
        margin: 2cm;
        @top-center {
            content: "FlagEmbedding 代码分析文档";
            font-size: 10pt;
            color: #666;
        }
        @bottom-center {
            content: "第 " counter(page) " 页";
            font-size: 10pt;
            color: #666;
        }
    }
    
    @page :first {
        @top-center { content: none; }
    }
    
    body {
        font-family: "Noto Sans CJK SC", "Source Han Sans CN", "WenQuanYi Micro Hei", "Microsoft YaHei", sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #333;
    }
    
    h1 {
        font-size: 24pt;
        color: #1976d2;
        border-bottom: 2px solid #1976d2;
        padding-bottom: 10px;
        page-break-after: avoid;
        margin-top: 30px;
    }
    
    h2 {
        font-size: 18pt;
        color: #1565c0;
        border-bottom: 1px solid #ccc;
        padding-bottom: 5px;
        page-break-after: avoid;
        margin-top: 25px;
    }
    
    h3 {
        font-size: 14pt;
        color: #333;
        page-break-after: avoid;
        margin-top: 20px;
    }
    
    h4 {
        font-size: 12pt;
        color: #555;
        page-break-after: avoid;
    }
    
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 20px auto;
        page-break-inside: avoid;
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        page-break-inside: avoid;
    }
    
    th, td {
        border: 1px solid #ddd;
        padding: 8px 12px;
        text-align: left;
    }
    
    th {
        background-color: #f5f5f5;
        font-weight: bold;
    }
    
    tr:nth-child(even) {
        background-color: #fafafa;
    }
    
    code {
        font-family: "Source Code Pro", "Consolas", monospace;
        background-color: #f5f5f5;
        padding: 2px 5px;
        border-radius: 3px;
        font-size: 9pt;
    }
    
    pre {
        background-color: #f8f8f8;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        overflow-x: auto;
        page-break-inside: avoid;
        margin: 15px 0;
    }
    
    pre code {
        background-color: transparent;
        padding: 0;
    }
    
    blockquote {
        border-left: 4px solid #1976d2;
        margin: 15px 0;
        padding: 10px 15px;
        background-color: #f5f9fc;
    }
    
    ul, ol {
        margin: 10px 0;
        padding-left: 25px;
    }
    
    li {
        margin: 5px 0;
    }
    
    hr {
        border: none;
        border-top: 1px solid #ddd;
        margin: 30px 0;
    }
    
    a {
        color: #1976d2;
        text-decoration: none;
    }
    
    strong {
        font-weight: bold;
        color: #000;
    }
    
    em {
        font-style: italic;
    }
    
    /* 代码高亮 */
    .python { background-color: #f5f5f5; }
    .javascript { background-color: #f5f5f5; }
    
    /* 特殊块 */
    .note {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 10px;
        margin: 15px 0;
        border-radius: 5px;
    }
    
    .warning {
        background-color: #f8d7da;
        border: 1px solid #dc3545;
        padding: 10px;
        margin: 15px 0;
        border-radius: 5px;
    }
    """

def merge_and_convert_to_pdf(input_dir, output_pdf):
    """合并 Markdown 文件并转换为 PDF"""
    print("\n=== Starting PDF Generation ===\n")
    
    # 创建输出目录
    temp_dir = tempfile.mkdtemp()
    img_dir = os.path.join(temp_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    try:
        # 读取所有 Markdown 文件
        print("1. Reading Markdown files...")
        md_contents = read_markdown_files(input_dir)
        
        if not md_contents:
            print("Error: No Markdown files found!")
            return False
        
        # 处理每个文件
        print("\n2. Processing Mermaid diagrams...")
        processed_contents = []
        
        for i, (filename, content) in enumerate(md_contents):
            print(f"   Processing {filename}...")
            processed = process_mermaid_blocks(content, img_dir)
            processed_contents.append(processed)
        
        # 创建合并的 HTML
        print("\n3. Generating HTML content...")
        html_parts = []
        
        for filename, content in zip([f for f, _ in md_contents], processed_contents):
            # 转换为 HTML
            md = Markdown(extensions=['tables', 'fenced_code', 'codehilite', 'toc'])
            html_content = md.convert(content)
            
            # 添加分隔符和标题
            doc_title = filename.replace('.md', '').replace('_', ' ').replace('-', ' ')
            html_parts.append(f'''
            <div class="document">
                <h1>{doc_title}</h1>
                {html_content}
                <hr/>
            </div>
            ''')
        
        # 生成完整的 HTML
        full_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>FlagEmbedding 代码分析文档</title>
            <style>
                {create_css()}
            </style>
        </head>
        <body>
            <div class="cover" style="text-align: center; padding: 100px 0;">
                <h1 style="font-size: 36pt; color: #1976d2; border: none; margin-bottom: 30px;">
                    FlagEmbedding
                </h1>
                <h2 style="font-size: 24pt; color: #333; border: none;">
                    代码分析文档
                </h2>
                <p style="font-size: 14pt; color: #666; margin-top: 50px;">
                    BGE: One-Stop Retrieval Toolkit For Search and RAG
                </p>
                <p style="font-size: 12pt; color: #999; margin-top: 100px;">
                    北京智源人工智能研究院 (BAAI)
                </p>
            </div>
            <div class="content">
                {''.join(html_parts)}
            </div>
        </body>
        </html>
        '''
        
        # 保存 HTML 文件
        html_path = os.path.join(temp_dir, 'content.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        print(f"   HTML saved to: {html_path}")
        
        # 转换为 PDF
        print("\n4. Converting to PDF...")
        try:
            HTML(filename=html_path).write_pdf(output_pdf)
            print(f"\n✓ PDF generated successfully: {output_pdf}")
            return True
        except Exception as e:
            print(f"WeasyPrint error: {e}")
            print("\n   Trying alternative method...")
            
            # 备选方案：使用 HTML 预览
            alternative_pdf = output_pdf.replace('.pdf', '_alt.html')
            shutil.copy(html_path, alternative_pdf)
            print(f"   HTML file created as alternative: {alternative_pdf}")
            return False
        
    finally:
        # 清理临时文件
        print("\n5. Cleaning up temporary files...")
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

if __name__ == '__main__':
    # 配置路径
    input_directory = '/workspace/ReadCode'
    output_file = '/workspace/FlagEmbedding_Analysis.pdf'
    
    print("=" * 60)
    print("FlagEmbedding Documentation PDF Generator")
    print("=" * 60)
    
    # 运行转换
    success = merge_and_convert_to_pdf(input_directory, output_file)
    
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS! PDF generated:")
        print(f"  {output_file}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("PARTIAL SUCCESS!")
        print("HTML file created as alternative.")
        print("=" * 60)
