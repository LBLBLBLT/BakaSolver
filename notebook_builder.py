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

    def add_markdown_cell(self, source: str) -> int:
        cell = self._make_cell("markdown", source)
        self.notebook["cells"].append(cell)
        return len(self.notebook["cells"]) - 1

    def get_cell_count(self) -> int:
        return len(self.notebook["cells"])

    def save(self, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.notebook, ensure_ascii=False, indent=1))
        return f"Notebook saved: {p} ({self.get_cell_count()} cells)"
