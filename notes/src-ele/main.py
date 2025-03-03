import os
import re
from pathlib import Path

def modify_headings(tex_content, mode='promote'):
    rules = {
        'promote': [
            # 修正顺序：从高级别到低级别替换
            ('section', 'chapter'),
            ('subsection', 'section'),
            ('subsubsection', 'subsection')
        ],
        'demote': [
            ('chapter', 'section'),
            ('section', 'subsection'),
            ('subsection', 'subsubsection'),
            ('subsubsection', 'paragraph'),
            ('paragraph', 'subparagraph')
        ]
    }

    replace_rules = rules[mode]
    
    # 正则表达式优化（支持可选参数和嵌套内容）
    pattern = r'\\%s(\*?)($$.*?$$)?\{((?:[^{}]|{(?:[^{}]|{[^{}]*})*})*)\}'

    for original, target in replace_rules:
        compiled = re.compile(pattern % original, re.DOTALL)
        replacement = r'\\%s\g<1>\g<2>{\g<3>}' % target
        tex_content = compiled.sub(replacement, tex_content)
    
    return tex_content

def process_directory(root_dir, mode='promote'):
    """
    递归处理目录下所有 .tex 文件
    """
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.tex'):
                file_path = Path(root) / file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                modified = modify_headings(content, mode)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified)
                print(f'已处理: {file_path}')

if __name__ == "__main__":
    process_directory('.', mode='promote')  # 或 mode='demote'