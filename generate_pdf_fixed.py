#!/usr/bin/env python3
"""
FlagEmbedding Documentation PDF Generator - Fixed Version
使用更好的方法渲染 Mermaid 图表
"""

import os
import re
import base64
from pathlib import Path
from markdown import Markdown
import subprocess
import tempfile
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects
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

def draw_flowchart(ax, mermaid_code):
    """绘制流程图"""
    ax.text(7, 7.5, 'Module Architecture Overview', fontsize=16, ha='center', 
            weight='bold', color='#1976d2',
            bbox=dict(boxstyle='round', facecolor='#e3f2fd', edgecolor='#1976d2', pad=0.5))
    
    # 添加说明
    ax.text(7, 6.8, '(Mermaid Flowchart)', 
            fontsize=10, ha='center', style='italic', color='#666')
    
    # 解析节点
    nodes = []
    arrows = []
    
    # 提取节点信息（简化版）
    node_pattern = r'(\w+)\[([^\]]+)\]'
    for match in re.finditer(node_pattern, mermaid_code):
        node_id, node_label = match.groups()
        nodes.append((node_id, node_label.replace('<br/>', '\n')))
    
    # 提取子图
    subgraph_pattern = r'subgraph\s+(\w+)\s*\["([^"]+)"\]'
    for match in re.finditer(subgraph_pattern, mermaid_code):
        subgraph_id, subgraph_label = match.groups()
        nodes.append((subgraph_id, subgraph_label))
    
    # 绘制节点
    if nodes:
        positions = {}
        cols = min(4, len(nodes))
        rows = (len(nodes) + cols - 1) // cols
        
        for i, (node_id, label) in enumerate(nodes[:8]):  # 最多8个节点
            row = i // cols
            col = i % cols
            x = 2 + col * 3
            y = 5 - row * 1.5
            
            positions[node_id] = (x, y)
            
            # 绘制节点框
            box = FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8,
                                boxstyle="round,pad=0.05,rounding_size=0.1",
                                facecolor='#bbdefb', edgecolor='#1976d2', 
                                linewidth=1.5, alpha=0.9)
            ax.add_patch(box)
            
            # 添加文本
            text = ax.text(x, y, label[:20], ha='center', va='center', 
                          fontsize=7, wrap=True)
    
    # 绘制箭头
    arrow_pattern = r'(\w+)\s*(-->|-.->)\s*(\w+)'
    for match in re.finditer(arrow_pattern, mermaid_code):
        src, arrow, dst = match.groups()
        if src in positions and dst in positions:
            sx, sy = positions[src]
            dx, dy = positions[dst]
            ax.annotate('', xy=(dx, dy), xytext=(sx, sy - 0.4),
                       arrowprops=dict(arrowstyle='->', color='#1976d2', lw=1.5,
                                      connectionstyle="arc3,rad=0"))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

def draw_class_diagram(ax, mermaid_code):
    """绘制类图"""
    ax.text(7, 7.5, 'Class Diagram', fontsize=16, ha='center', 
            weight='bold', color='#e65100',
            bbox=dict(boxstyle='round', facecolor='#fff3e0', edgecolor='#e65100', pad=0.5))
    
    ax.text(7, 6.8, '(Mermaid Class Diagram)', 
            fontsize=10, ha='center', style='italic', color='#666')
    
    # 提取类信息
    classes = []
    class_pattern = r'class\s+(\w+)\s*\{([^}]+)\}'
    for match in re.finditer(class_pattern, mermaid_code):
        class_name, methods = match.groups()
        class_methods = [m.strip() for m in methods.split('\n') if m.strip()]
        classes.append((class_name, class_methods))
    
    # 绘制类框
    if classes:
        for i, (class_name, methods) in enumerate(classes[:4]):
            x = 3 + (i % 2) * 7
            y = 5 - (i // 2) * 2.5
            
            # 类名框
            name_box = FancyBboxPatch((x-2, y-0.3), 4, 0.6,
                                    boxstyle="round,pad=0.05",
                                    facecolor='#ffe0b2', edgecolor='#e65100', 
                                    linewidth=2)
            ax.add_patch(name_box)
            ax.text(x, y, f"«{class_name}»" if 'abstract' in mermaid_code.lower() else class_name, 
                   ha='center', va='center', fontsize=10, weight='bold')
            
            # 方法列表
            method_text = '\n'.join(methods[:3])
            ax.text(x, y - 0.8, method_text, ha='center', va='top', 
                   fontsize=7, family='monospace')
    
    # 绘制继承关系
    inherit_pattern = r'(\w+)\s*<\|--\s*(\w+)'
    for match in re.finditer(inherit_pattern, mermaid_code):
        parent, child = match.groups()
        ax.annotate('', xy=(6, 4), xytext=(4, 4),
                   arrowprops=dict(arrowstyle='-|>', color='#e65100', lw=2))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

def draw_sequence_diagram(ax, mermaid_code):
    """绘制时序图"""
    ax.text(7, 7.5, 'Sequence Diagram', fontsize=16, ha='center', 
            weight='bold', color='#2e7d32',
            bbox=dict(boxstyle='round', facecolor='#e8f5e9', edgecolor='#2e7d32', pad=0.5))
    
    ax.text(7, 6.8, '(Mermaid Sequence Diagram)', 
            fontsize=10, ha='center', style='italic', color='#666')
    
    # 提取参与者
    participants = []
    participant_pattern = r'participant\s+(\w+)\s+as\s+(\w+)'
    for match in re.finditer(participant_pattern, mermaid_code):
        full_name, short_name = match.groups()
        participants.append((full_name, short_name))
    
    if not participants:
        participant_pattern = r'participant\s+(\w+)'
        for match in re.finditer(participant_pattern, mermaid_code):
            participants.append((match.group(1), match.group(1)))
    
    # 绘制参与者生命线
    n = min(len(participants), 5)
    for i, (full_name, short_name) in enumerate(participants[:n]):
        x = 2 + i * 2.5
        # 参与者框
        box = FancyBboxPatch((x-0.8, 5.5), 1.6, 0.5,
                            boxstyle="round,pad=0.05",
                            facecolor='#c8e6c9', edgecolor='#2e7d32', 
                            linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, 5.75, short_name, ha='center', va='center', fontsize=9, weight='bold')
        
        # 生命线
        ax.plot([x, x], [1, 5.5], '#2e7d32', lw=1.5, linestyle='--', alpha=0.5)
    
    # 绘制消息
    message_pattern = r'(\w+)->>(\w+):\s*(.+)'
    y_pos = 4.5
    for match in re.finditer(message_pattern, mermaid_code):
        src, dst, msg = match.groups()
        msg_short = msg[:15] + '...' if len(msg) > 15 else msg
        
        # 找到参与者位置
        src_idx = next((i for i, (f, s) in enumerate(participants[:n]) if s == src or f == src), 0)
        dst_idx = next((i for i, (f, s) in enumerate(participants[:n]) if s == dst or f == dst), 0)
        
        x1 = 2 + src_idx * 2.5
        x2 = 2 + dst_idx * 2.5
        
        # 箭头
        if x1 < x2:
            ax.annotate('', xy=(x2-0.1, y_pos), xytext=(x1+0.1, y_pos),
                       arrowprops=dict(arrowstyle='->', color='#1976d2', lw=1.5))
        else:
            ax.annotate('', xy=(x2+0.1, y_pos), xytext=(x1-0.1, y_pos),
                       arrowprops=dict(arrowstyle='<-', color='#1976d2', lw=1.5))
        
        ax.text((x1+x2)/2, y_pos+0.1, msg_short, ha='center', va='bottom', fontsize=7)
        y_pos -= 0.8
        
        if y_pos < 1.5:
            break
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

def draw_timeline(ax, mermaid_code):
    """绘制时间线"""
    ax.text(7, 7.5, 'Technology Evolution Timeline', fontsize=16, ha='center', 
            weight='bold', color='#7b1fa2',
            bbox=dict(boxstyle='round', facecolor='#f3e5f5', edgecolor='#7b1fa2', pad=0.5))
    
    # 提取时间线内容
    sections = []
    section_pattern = r'section\s+(\w+)\s*\n((?:\s+\w+.+:.*\n)*)'
    for match in re.finditer(section_pattern, mermaid_code):
        section_name = match.group(1)
        items = match.group(2).strip().split('\n')
        sections.append((section_name, items))
    
    if sections:
        # 绘制主时间线
        ax.plot([1, 13], [4, 4], color='#7b1fa2', lw=4, solid_capstyle='round')
        
        # 添加节点
        phase_positions = [2.5, 7, 11.5]
        for i, (phase, items) in enumerate(sections[:3]):
            x = phase_positions[i]
            
            # 节点圆
            circle = plt.Circle((x, 4), 0.25, color='#7b1fa2', zorder=5)
            ax.add_patch(circle)
            
            # 阶段标题
            ax.text(x, 5.5, phase, ha='center', va='bottom', fontsize=10, 
                   weight='bold', color='#7b1fa2')
            
            # 绘制向上的垂直线
            ax.plot([x, x], [4.25, 5.3], color='#7b1fa2', lw=1.5)
            
            # 节点内容
            for j, item in enumerate(items[:3]):
                item = item.strip().replace(':', ':')
                if item:
                    ax.text(x, 4 - 0.4 - j*0.5, item.strip(), ha='center', va='top', 
                           fontsize=7, style='italic')
    else:
        # 简化版本
        ax.plot([1, 13], [4, 4], color='#7b1fa2', lw=4)
        
        for i, label in enumerate(['Foundation', 'Expansion', 'Specialization']):
            x = 2.5 + i * 4
            circle = plt.Circle((x, 4), 0.25, color='#7b1fa2')
            ax.add_patch(circle)
            ax.text(x, 5, label, ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

def draw_simple_diagram(mermaid_code, output_path):
    """使用 matplotlib 绘制简化的图表"""
    try:
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('white')
        
        # 检测图表类型并绘制
        if 'flowchart' in mermaid_code or 'subgraph' in mermaid_code:
            draw_flowchart(ax, mermaid_code)
        elif 'classDiagram' in mermaid_code:
            draw_class_diagram(ax, mermaid_code)
        elif 'sequenceDiagram' in mermaid_code:
            draw_sequence_diagram(ax, mermaid_code)
        elif 'timeline' in mermaid_code:
            draw_timeline(ax, mermaid_code)
        else:
            # 默认显示为文本
            ax.text(0.5, 0.5, f'Diagram: {mermaid_code[:100]}...', 
                   transform=ax.transAxes, fontsize=12, ha='center')
            ax.axis('off')
        
        plt.tight_layout(pad=2)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Diagram drawing error: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_mermaid_blocks(content, output_dir, counter):
    """处理 Markdown 中的 Mermaid 代码块"""
    pattern = r'```mermaid\n(.*?)```'
    
    def replace_mermaid(match):
        nonlocal counter
        mermaid_code = match.group(1)
        
        # 生成唯一的文件名
        img_path = os.path.join(output_dir, f'mermaid_{counter:03d}.png')
        rel_path = f'mermaid_{counter:03d}.png'
        counter += 1
        
        # 尝试渲染
        success = draw_simple_diagram(mermaid_code, img_path)
        
        if success and os.path.exists(img_path):
            return f'\n\n### Diagram {counter-1}\n![Mermaid Diagram]({rel_path})\n\n'
        else:
            # 如果失败，返回原始代码作为代码块
            return f'\n```mermaid\n{mermaid_code}```\n'
    
    processed_content = re.sub(pattern, replace_mermaid, content, flags=re.DOTALL)
    return processed_content, counter

def create_css():
    """创建 PDF 样式"""
    return """
    @page {
        size: A4;
        margin: 1.5cm 2cm;
    }
    
    body {
        font-family: "Noto Sans CJK SC", "Source Han Sans CN", "WenQuanYi Micro Hei", 
                     "Microsoft YaHei", "DejaVu Sans", sans-serif;
        font-size: 10pt;
        line-height: 1.7;
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
    }
    
    h2 {
        font-size: 16pt;
        color: #1565c0;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 5px;
        margin-top: 30px;
        margin-bottom: 15px;
        page-break-after: avoid;
    }
    
    h3 {
        font-size: 13pt;
        color: #333;
        margin-top: 20px;
        margin-bottom: 10px;
        page-break-after: avoid;
    }
    
    h4 {
        font-size: 11pt;
        color: #555;
        margin-top: 15px;
        margin-bottom: 8px;
        page-break-after: avoid;
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
    print("FlagEmbedding Documentation PDF Generator v2.0")
    print("="*60 + "\n")
    
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
        print("\n2. Processing and rendering diagrams...")
        processed_contents = []
        global_counter = 0
        
        for filename, content in md_contents:
            print(f"   Processing {filename}...")
            processed, global_counter = process_mermaid_blocks(content, img_dir, global_counter)
            processed_contents.append((filename, processed))
        
        # 创建合并的 HTML
        print("\n3. Generating HTML content with styled diagrams...")
        
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
        
        # 复制图片目录到工作目录
        final_img_dir = '/workspace/ReadCode/images'
        shutil.copytree(img_dir, final_img_dir, dirs_exist_ok=True)
        print(f"   ✓ Images saved to: {final_img_dir}")
        
        # 转换为 PDF
        print("\n4. Converting to PDF...")
        try:
            from weasyprint import HTML, CSS
            
            HTML(filename=html_path).write_pdf(output_pdf)
            print(f"\n{'='*60}")
            print(f"✓ SUCCESS! PDF generated:")
            print(f"  {output_pdf}")
            print(f"{'='*60}")
            return True
            
        except Exception as e:
            print(f"WeasyPrint error: {e}")
            print("\n   Alternative: HTML file created for manual PDF export")
            html_output = output_pdf.replace('.pdf', '.html')
            shutil.copy(html_path, html_output)
            print(f"   ✓ HTML file: {html_output}")
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
    output_file = '/workspace/FlagEmbedding_Analysis.pdf'
    
    # 运行转换
    success = merge_and_convert_to_pdf(input_directory, output_file)
