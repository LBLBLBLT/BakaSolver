"""
notebook_builder.py - Jupyter Notebook 生成器

维护 nbformat v4 的 JSON 结构，Agent 通过工具调用逐步追加 cell。
"""

import json
from pathlib import Path


class NotebookBuilder:
    def __init__(self, title: str = "数学建模分析"):
        self._cell_counter = 0
        self.notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0"
                }
            },
            "cells": []
        }
        self.add_markdown_cell(f"# {title}")

    def _make_cell(self, cell_type: str, source: str) -> dict:
        self._cell_counter += 1
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        return cell

    def add_code_cell(self, source: str) -> int:
        cell = self._make_cell("code", source)
        self.notebook["cells"].append(cell)
        return len(self.notebook["cells"]) - 1

    def add_code_cell_with_output(self, source: str, stdout_text: str = "") -> int:
        """追加一个带真实 stdout 输出的 code cell。

        stdout_text 作为 nbformat stream output 嵌入，
        这样打开 notebook 就能看到代码的真实运行结果。
        """
        cell = self._make_cell("code", source)
        cell["execution_count"] = self._cell_counter
        if stdout_text:
            cell["outputs"] = [{
                "output_type": "stream",
                "name": "stdout",
                "text": stdout_text.splitlines(keepends=True),
            }]
        self.notebook["cells"].append(cell)
        return len(self.notebook["cells"]) - 1

    def add_markdown_cell(self, source: str) -> int:
        cell = self._make_cell("markdown", source)
        self.notebook["cells"].append(cell)
        return len(self.notebook["cells"]) - 1

    def get_cell_count(self) -> int:
        return len(self.notebook["cells"])

    @classmethod
    def load(cls, path: str) -> "NotebookBuilder":
        """从已有的 .ipynb 文件恢复 builder，使后续 cell 追加而非覆盖。

        进程重启（续做模式）时调用，避免第一次 save() 把上次运行积累的
        cell 全部冲掉。文件不存在或解析失败时返回一个全新 builder。
        """
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()

        builder = cls.__new__(cls)
        builder.notebook = data
        builder.notebook.setdefault("cells", [])
        # _cell_counter 至少为已有 cell 数，保证新 execution_count 不回退
        existing_counts = [
            c.get("execution_count") or 0
            for c in builder.notebook["cells"]
            if c.get("cell_type") == "code"
        ]
        builder._cell_counter = max(
            [len(builder.notebook["cells"])] + existing_counts
        )
        return builder

    def save(self, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.notebook, ensure_ascii=False, indent=1),
                     encoding="utf-8")
        return f"Notebook saved: {p} ({self.get_cell_count()} cells)"
