"""
tools.py - 工具定义与实现

7 个工具：python_execute, web_search, read_file, write_file,
         add_notebook_cell, todo_write, list_files

设计模式来自 learn-claude-code s02: TOOLS 定义 schema, TOOL_HANDLERS 映射执行函数。
"""

import ast
import json
import os
import subprocess
import tempfile
import re
from pathlib import Path

from notebook_builder import NotebookBuilder

# ═══════════════════════════════════════════════════════════
#  全局状态
# ═══════════════════════════════════════════════════════════

WORKDIR = Path.cwd()
DATA_DIR = WORKDIR / "data"
OUTPUT_DIR = WORKDIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

CURRENT_TODOS: list[dict] = []
NOTEBOOK: NotebookBuilder | None = None


def get_notebook() -> NotebookBuilder:
    global NOTEBOOK
    if NOTEBOOK is None:
        NOTEBOOK = NotebookBuilder()
    return NOTEBOOK


# ═══════════════════════════════════════════════════════════
#  工具实现
# ═══════════════════════════════════════════════════════════

def run_python_execute(code: str) -> str:
    """执行 Python 代码，捕获输出。自动处理 matplotlib 图片保存。"""
    code = _patch_matplotlib(code)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                      dir=OUTPUT_DIR, encoding="utf-8")
    try:
        tmp.write(code)
        tmp.close()
        r = subprocess.run(
            ["python", tmp.name],
            capture_output=True, text=True, timeout=300,
            cwd=str(OUTPUT_DIR),
            env={**os.environ, "MPLBACKEND": "Agg"}
        )
        out = (r.stdout + r.stderr).strip()
        return out[:30000] if out else "(执行完成，无输出)"
    except subprocess.TimeoutExpired:
        return "Error: 执行超时 (300s)"
    except Exception as e:
        return f"Error: {e}"
    finally:
        os.unlink(tmp.name)


def _patch_matplotlib(code: str) -> str:
    """将 plt.show() 替换为 savefig，方便在无 GUI 环境运行。"""
    if "plt.show()" in code:
        fig_name = f"figure_{hash(code) % 10000}.png"
        code = code.replace(
            "plt.show()",
            f"plt.savefig('{fig_name}', dpi=150, bbox_inches='tight')\n"
            f"print(f'图片已保存: {fig_name}')"
        )
    return code


def run_web_search(query: str) -> str:
    """使用 DuckDuckGo 搜索（免费，无需 API key）。"""
    try:
        import requests
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15
        )
        from html.parser import HTMLParser

        results = []

        class DDGParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self._in_result = False
                self._current = ""

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                if tag == "a" and "result__a" in attrs_dict.get("class", ""):
                    self._in_result = True
                    self._current = attrs_dict.get("href", "")

            def handle_data(self, data):
                if self._in_result:
                    results.append(f"{data.strip()} — {self._current}")
                    self._in_result = False

        parser = DDGParser()
        parser.feed(resp.text)
        return "\n".join(results[:8]) if results else "(无搜索结果)"
    except Exception as e:
        return f"搜索失败: {e}"


def run_read_file(path: str) -> str:
    """读取文件内容。仅适用于小文件（配置、题目等），大数据文件应使用 inspect_data。"""
    try:
        target = (WORKDIR / path).resolve()
        if not target.exists():
            target = (DATA_DIR / path).resolve()
        if not target.exists():
            return f"Error: 文件不存在: {path}"

        size = target.stat().st_size
        if size > 100_000:
            return (
                f"[文件过大] {target.name} ({size / 1024:.1f}KB)\n"
                f"该文件不适合直接读取，请使用 inspect_data 工具查看数据概况，"
                f"或用 python_execute 按需加载。"
            )

        content = target.read_text(encoding="utf-8", errors="replace")
        return content
    except Exception as e:
        return f"Error: {e}"


def run_inspect_data(path: str, sample_rows: int = 5) -> str:
    """智能数据探查：返回数据概况（shape/dtypes/统计/样本），不会撑爆上下文。"""
    try:
        target = (WORKDIR / path).resolve()
        if not target.exists():
            target = (DATA_DIR / path).resolve()
        if not target.exists():
            return f"Error: 文件不存在: {path}"

        suffix = target.suffix.lower()
        size = target.stat().st_size
        size_str = f"{size / 1024 / 1024:.2f}MB" if size > 1024 * 1024 else f"{size / 1024:.1f}KB"

        # 构造探查代码
        code = _build_inspect_code(str(target), suffix, sample_rows)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                          dir=OUTPUT_DIR, encoding="utf-8")
        try:
            tmp.write(code)
            tmp.close()
            r = subprocess.run(
                ["python", tmp.name],
                capture_output=True, text=True, timeout=60,
                cwd=str(OUTPUT_DIR)
            )
            out = (r.stdout + r.stderr).strip()
        finally:
            os.unlink(tmp.name)

        header = f"=== 数据概况: {target.name} ({size_str}) ===\n"
        return header + (out[:8000] if out else "(探查无输出)")
    except Exception as e:
        return f"Error: {e}"


def _build_inspect_code(file_path: str, suffix: str, sample_rows: int) -> str:
    """根据文件类型生成探查脚本。"""
    escaped_path = file_path.replace("\\", "\\\\")

    if suffix in (".csv", ".tsv", ".txt"):
        sep = "\\t" if suffix == ".tsv" else ","
        return f'''
import pandas as pd
import sys

path = r"{file_path}"
try:
    df = pd.read_csv(path, sep="{sep}", nrows=100000)
except:
    df = pd.read_csv(path, sep=None, engine="python", nrows=100000)

print(f"形状: {{df.shape[0]}} 行 x {{df.shape[1]}} 列")
print(f"\\n列名与类型:")
for col in df.columns:
    null_pct = df[col].isnull().mean() * 100
    print(f"  {{col}}: {{df[col].dtype}}  (缺失{{null_pct:.1f}}%)")

print(f"\\n数值列统计:")
desc = df.describe()
print(desc.to_string())

print(f"\\n前 {sample_rows} 行:")
print(df.head({sample_rows}).to_string())

print(f"\\n后 3 行:")
print(df.tail(3).to_string())

cat_cols = df.select_dtypes(include=["object", "category"]).columns
if len(cat_cols) > 0:
    print(f"\\n类别列唯一值数量:")
    for col in cat_cols[:10]:
        print(f"  {{col}}: {{df[col].nunique()}} 个唯一值 → {{df[col].value_counts().head(5).to_dict()}}")
'''
    elif suffix in (".xlsx", ".xls"):
        return f'''
import pandas as pd
path = r"{file_path}"
xls = pd.ExcelFile(path)
print(f"Sheet 数量: {{len(xls.sheet_names)}}")
print(f"Sheet 名: {{xls.sheet_names}}")
for sheet in xls.sheet_names[:3]:
    df = pd.read_excel(xls, sheet_name=sheet, nrows=100000)
    print(f"\\n--- Sheet: {{sheet}} ---")
    print(f"形状: {{df.shape[0]}} 行 x {{df.shape[1]}} 列")
    for col in df.columns:
        null_pct = df[col].isnull().mean() * 100
        print(f"  {{col}}: {{df[col].dtype}}  (缺失{{null_pct:.1f}}%)")
    print(df.head({sample_rows}).to_string())
'''
    elif suffix == ".json":
        return f'''
import json, pandas as pd
path = r"{file_path}"
with open(path, "r", encoding="utf-8") as f:
    raw = json.load(f)
if isinstance(raw, list):
    print(f"JSON 数组，共 {{len(raw)}} 条记录")
    df = pd.DataFrame(raw[:1000])
    print(f"列: {{list(df.columns)}}")
    print(df.head({sample_rows}).to_string())
elif isinstance(raw, dict):
    print(f"JSON 对象，顶层键: {{list(raw.keys())[:20]}}")
    for k in list(raw.keys())[:5]:
        v = raw[k]
        print(f"  {{k}}: {{type(v).__name__}} (长度={{len(v) if hasattr(v, '__len__') else 'N/A'}})")
'''
    elif suffix in (".parquet",):
        return f'''
import pandas as pd
path = r"{file_path}"
df = pd.read_parquet(path)
print(f"形状: {{df.shape[0]}} 行 x {{df.shape[1]}} 列")
for col in df.columns:
    null_pct = df[col].isnull().mean() * 100
    print(f"  {{col}}: {{df[col].dtype}}  (缺失{{null_pct:.1f}}%)")
print(df.describe().to_string())
print(df.head({sample_rows}).to_string())
'''
    elif suffix == ".mat":
        return f'''
import scipy.io
path = r"{file_path}"
mat = scipy.io.loadmat(path)
keys = [k for k in mat.keys() if not k.startswith("__")]
print(f"MAT 文件变量: {{keys}}")
for k in keys[:10]:
    v = mat[k]
    print(f"  {{k}}: shape={{getattr(v, 'shape', 'N/A')}}, dtype={{getattr(v, 'dtype', type(v).__name__)}}")
'''
    else:
        return f'''
path = r"{file_path}"
with open(path, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()
print(f"文本文件，共 {{len(lines)}} 行")
print(f"前 10 行:")
for line in lines[:10]:
    print(repr(line.rstrip()))
'''


def run_write_file(path: str, content: str) -> str:
    """写入文件。"""
    try:
        target = (OUTPUT_DIR / path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"已写入 {len(content)} 字节到 {target}"
    except Exception as e:
        return f"Error: {e}"


def run_add_notebook_cell(cell_type: str, source: str) -> str:
    """向 Notebook 追加一个 cell。"""
    nb = get_notebook()
    if cell_type == "code":
        idx = nb.add_code_cell(source)
    else:
        idx = nb.add_markdown_cell(source)
    nb.save(str(OUTPUT_DIR / "modeling.ipynb"))
    return f"已追加 {cell_type} cell (index={idx})，当前共 {nb.get_cell_count()} cells"


def run_todo_write(todos: list) -> str:
    """任务规划与进度追踪。"""
    global CURRENT_TODOS
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            try:
                todos = ast.literal_eval(todos)
            except (SyntaxError, ValueError):
                return "Error: todos 必须是 JSON 数组"
    if not isinstance(todos, list):
        return "Error: todos 必须是数组"
    for i, t in enumerate(todos):
        if not isinstance(t, dict):
            return f"Error: todos[{i}] 必须是对象"
        if "content" not in t or "status" not in t:
            return f"Error: todos[{i}] 缺少 content 或 status"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return f"Error: todos[{i}] status 无效"
    CURRENT_TODOS = todos
    lines = ["\n== 当前任务 =="]
    icons = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
    for t in CURRENT_TODOS:
        lines.append(f"  {icons[t['status']]} {t['content']}")
    print("\n".join(lines))
    return f"已更新 {len(CURRENT_TODOS)} 个任务"


def run_list_files(directory: str = ".") -> str:
    """列出目录内容，带文件类型和大小信息。"""
    try:
        target = (WORKDIR / directory).resolve()
        if not target.exists():
            return f"Error: 目录不存在: {directory}"
        entries = []
        for item in sorted(target.iterdir()):
            if item.is_dir():
                entries.append(f"  [DIR]  {item.name}/")
            else:
                size = item.stat().st_size
                suffix = item.suffix.lower()
                if size > 1024 * 1024:
                    size_str = f"{size / 1024 / 1024:.1f}MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                entries.append(f"  [{size_str}]  {item.name}  ({suffix or 'no ext'})")
        return "\n".join(entries) if entries else "(空目录)"
    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════
#  工具 Schema 定义 (OpenAI function calling 格式)
# ═══════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python_execute",
            "description": "执行 Python 代码。用于数据分析、建模、绘图等。代码在 output/ 目录下运行。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的 Python 代码"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "联网搜索信息或数据。返回搜索结果摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取小文件内容（<100KB，如配置、题目、小数据）。大数据文件请用 inspect_data。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径（相对于工作目录）"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_data",
            "description": "智能数据探查：自动识别文件格式，返回数据概况（行列数、字段类型、缺失率、统计摘要、样本行）。适用于任何大小的数据文件（csv/xlsx/json/parquet/mat 等）。不会塞满上下文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "数据文件路径（如 data/train/xxx.csv）"},
                    "sample_rows": {"type": "integer", "description": "显示样本行数，默认5", "default": 5}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入文件到 output/ 目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件名（保存在 output/ 下）"},
                    "content": {"type": "string", "description": "文件内容"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_notebook_cell",
            "description": "向 Jupyter Notebook 追加一个 cell。所有分析代码和说明都应通过此工具写入 Notebook。",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_type": {"type": "string", "enum": ["code", "markdown"],
                                  "description": "cell 类型"},
                    "source": {"type": "string", "description": "cell 内容（代码或 Markdown 文本）"}
                },
                "required": ["cell_type", "source"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": "规划和管理任务列表。开始建模前必须先规划步骤，执行过程中更新状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {"type": "string",
                                           "enum": ["pending", "in_progress", "completed"]}
                            },
                            "required": ["content", "status"]
                        },
                        "description": "任务列表"
                    }
                },
                "required": ["todos"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出目录中的文件和子目录，显示文件大小和类型。数据文件在 data/ 目录下。",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "目录路径（如 data/），默认为当前目录",
                                  "default": "."}
                }
            }
        }
    },
]

# ═══════════════════════════════════════════════════════════
#  工具分发表
# ═══════════════════════════════════════════════════════════

TOOL_HANDLERS = {
    "python_execute": lambda **kw: run_python_execute(kw["code"]),
    "web_search": lambda **kw: run_web_search(kw["query"]),
    "read_file": lambda **kw: run_read_file(kw["path"]),
    "inspect_data": lambda **kw: run_inspect_data(kw["path"], kw.get("sample_rows", 5)),
    "write_file": lambda **kw: run_write_file(kw["path"], kw["content"]),
    "add_notebook_cell": lambda **kw: run_add_notebook_cell(kw["cell_type"], kw["source"]),
    "todo_write": lambda **kw: run_todo_write(kw["todos"]),
    "list_files": lambda **kw: run_list_files(kw.get("directory", ".")),
}


def dispatch(name: str, arguments: str) -> str:
    """根据工具名和 JSON 参数字符串分发执行。"""
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return f"未知工具: {name}"
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return f"参数解析失败: {arguments}"
    try:
        return handler(**args)
    except Exception as e:
        return f"工具执行错误 [{name}]: {e}"
