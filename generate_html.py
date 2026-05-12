#!/usr/bin/env python3
"""
FlagEmbedding Documentation HTML Generator
直接生成HTML文件，用户可以用浏览器打开并打印为PDF，完美支持中文
"""

import os
import re
from pathlib import Path
from markdown import Markdown

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

def convert_images_to_data_url(content, img_dir):
    """转换图片引用为本地路径"""
    pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
    
    def replace_image(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        
        # 如果是本地文件，转换为绝对路径
        if not img_path.startswith('http'):
            full_path = os.path.join(img_dir, img_path)
            if os.path.exists(full_path):
                return f'![{alt_text}]({full_path})'
        
        return match.group(0)
    
    return re.sub(pattern, replace_image, content)

def create_html():
    """生成完整的HTML"""
    print("\n" + "="*60)
    print("FlagEmbedding Documentation HTML Generator")
    print("="*60 + "\n")
    
    input_directory = '/workspace/ReadCode'
    output_html = '/workspace/FlagEmbedding_Analysis.html'
    img_dir = '/workspace/ReadCode/images'
    
    # 读取Markdown文件
    print("1. Reading Markdown files...")
    md_contents = read_markdown_files(input_directory)
    
    if not md_contents:
        print("Error: No Markdown files found!")
        return
    
    # 处理内容
    print("\n2. Processing content...")
    html_parts = []
    doc_num = 0
    
    for filename, content in md_contents:
        doc_num += 1
        print(f"   Processing {filename}...")
        
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
    print("\n3. Generating HTML...")
    
    full_html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlagEmbedding 代码分析文档</title>
    <style>
        @charset "UTF-8";
        
        * {{
            box-sizing: border-box;
        }}
        
        @page {{
            size: A4;
            margin: 1.5cm 2cm;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans",
                         "Noto Sans CJK SC", "Source Han Sans CN", "Microsoft YaHei",
                         "WenQuanYi Micro Hei", sans-serif;
            font-size: 11pt;
            line-height: 1.8;
            color: #2c3e50;
            text-align: justify;
            max-width: 21cm;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f7fa;
        }}
        
        .container {{
            background-color: white;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            font-size: 22pt;
            color: #1976d2;
            border-bottom: 3px solid #1976d2;
            padding-bottom: 8px;
            margin-top: 40px;
            margin-bottom: 20px;
            page-break-after: avoid;
            text-align: left;
            font-weight: bold;
        }}
        
        h2 {{
            font-size: 16pt;
            color: #1565c0;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 5px;
            margin-top: 30px;
            margin-bottom: 15px;
            page-break-after: avoid;
            font-weight: bold;
        }}
        
        h3 {{
            font-size: 13pt;
            color: #333;
            margin-top: 20px;
            margin-bottom: 10px;
            page-break-after: avoid;
            font-weight: bold;
        }}
        
        h4 {{
            font-size: 11pt;
            color: #555;
            margin-top: 15px;
            margin-bottom: 8px;
            page-break-after: avoid;
            font-weight: bold;
        }}
        
        img {{
            max-width: 100%;
            max-height: 400px;
            height: auto;
            display: block;
            margin: 25px auto;
            page-break-inside: avoid;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            page-break-inside: avoid;
            font-size: 9pt;
        }}
        
        th, td {{
            border: 1px solid #bdbdbd;
            padding: 8px 10px;
            text-align: left;
            vertical-align: top;
        }}
        
        th {{
            background-color: #e3f2fd;
            font-weight: bold;
            color: #1565c0;
        }}
        
        tr:nth-child(even) {{
            background-color: #fafafa;
        }}
        
        tr:hover {{
            background-color: #f5f5f5;
        }}
        
        code {{
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
            background-color: #f5f5f5;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 8.5pt;
            color: #c7254e;
        }}
        
        pre {{
            background-color: #263238;
            color: #aed581;
            border-radius: 5px;
            padding: 12px;
            overflow-x: auto;
            page-break-inside: avoid;
            margin: 15px 0;
            font-size: 8pt;
            line-height: 1.5;
        }}
        
        pre code {{
            background-color: transparent;
            padding: 0;
            color: inherit;
        }}
        
        blockquote {{
            border-left: 4px solid #1976d2;
            margin: 20px 0;
            padding: 10px 15px;
            background-color: #e3f2fd;
            color: #0d47a1;
        }}
        
        ul, ol {{
            margin: 10px 0;
            padding-left: 25px;
        }}
        
        li {{
            margin: 6px 0;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #e0e0e0;
            margin: 40px 0;
        }}
        
        strong {{
            font-weight: bold;
            color: #c62828;
        }}
        
        em {{
            font-style: italic;
            color: #1565c0;
        }}
        
        .cover {{
            text-align: center;
            padding: 150px 50px;
            page-break-after: always;
        }}
        
        .document {{
            margin-bottom: 30px;
        }}
        
        .toc {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
            border: 1px solid #e0e0e0;
        }}
        
        .toc h2 {{
            margin-top: 0;
            border-bottom: 2px solid #1976d2;
        }}
        
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        
        .toc li {{
            margin: 8px 0;
        }}
        
        .toc a {{
            color: #1976d2;
            text-decoration: none;
        }}
        
        .toc a:hover {{
            text-decoration: underline;
        }}
        
        @media print {{
            body {{
                background-color: white;
                padding: 0;
            }}
            
            .container {{
                box-shadow: none;
                padding: 0;
            }}
            
            a {{
                color: inherit;
                text-decoration: none;
            }}
        }}
        
        /* 按钮样式 */
        .print-btn {{
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #1976d2;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 14px;
            border-radius: 5px;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            z-index: 1000;
        }}
        
        .print-btn:hover {{
            background-color: #1565c0;
        }}
        
        /* Mermaid 代码块样式 */
        .mermaid {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            text-align: center;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <button class="print-btn" onclick="window.print()">🖨️ 打印 / 导出为 PDF</button>
    
    <div class="container">
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
            <hr style="width: 50%; margin: 50px auto; border-top: 2px solid #e0e0e0;">
            <p style="font-size: 12pt; color: #888;">
                北京智源人工智能研究院 (BAAI)
            </p>
            <p style="font-size: 10pt; color: #aaa; margin-top: 100px;">
                基于源代码的深度技术分析
            </p>
        </div>
        
        <div class="toc">
            <h2>📋 目录</h2>
            <ul>
                <li><a href="#doc-1">01 Overview</a></li>
                <li><a href="#doc-2">02 Abc Layer</a></li>
                <li><a href="#doc-3">03 Inference Module</a></li>
                <li><a href="#doc-4">04 Finetune Module</a></li>
                <li><a href="#doc-5">05 Evaluation Module</a></li>
                <li><a href="#doc-6">06 Key Algorithms</a></li>
                <li><a href="#doc-7">07 Research Projects</a></li>
                <li><a href="#doc-8">08 Model Survey</a></li>
            </ul>
        </div>
        
        <div class="content">
            {''.join(html_parts)}
        </div>
    </div>
    
    <script>
        // 可选：如果页面有Mermaid，这里可以添加渲染代码
    </script>
</body>
</html>'''
    
    # 保存 HTML 文件
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"\n{'='*60}")
    print(f"✓ SUCCESS! HTML generated:")
    print(f"  {output_html}")
    print(f"{'='*60}")
    print("\n📖 使用说明:")
    print("1. 在浏览器中打开该 HTML 文件")
    print("2. 点击右上角的 '打印 / 导出为 PDF' 按钮")
    print("3. 选择 '保存为 PDF' 作为目标")
    print("4. 确保勾选 '背景图形' 选项（如需要）")
    print("5. 保存 PDF 文件\n")

if __name__ == '__main__':
    create_html()
