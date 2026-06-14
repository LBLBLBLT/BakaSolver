#!/usr/bin/env python3
"""
agent.py - 数学建模 Auto Agent 主入口

核心循环模式来自 learn-claude-code s01，适配 OpenAI function calling 格式。
增加了 todo reminder 注入（s05）和上下文压缩（s08）。

用法：
    pip install -r requirements.txt
    cp .env.example .env  # 填入 API key
    # 将题目写入 problem.md
    python agent.py                  # 默认读取 problem.md
    python agent.py my_problem.md    # 或指定其他文件
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 确保能 import 同目录模块
sys.path.insert(0, str(Path(__file__).parent))

from tools import TOOLS, dispatch, CURRENT_TODOS, get_notebook
from compaction import prepare_context, reactive_compact, estimate_size
from prompts import build_system_prompt, build_reminder

load_dotenv(override=True)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
MODEL = os.environ.get("MODEL_ID", "deepseek-chat")
SYSTEM_PROMPT = build_system_prompt()


# ═══════════════════════════════════════════════════════════
#  进度输出
# ═══════════════════════════════════════════════════════════

def print_status(turn: int, msg_count: int):
    """每轮打印状态行，让用户知道 Agent 在干什么。"""
    from tools import CURRENT_TODOS
    # 找当前进行中的任务
    current_stage = None
    completed = 0
    total = len(CURRENT_TODOS)
    for t in CURRENT_TODOS:
        if t["status"] == "in_progress":
            current_stage = t["content"]
        if t["status"] == "completed":
            completed += 1

    progress = f"[{completed}/{total}]" if total > 0 else ""
    stage = f" | {current_stage}" if current_stage else ""
    ctx_size = estimate_size([])  # placeholder
    print(f"\033[90m── 轮次 {turn+1} {progress}{stage} | 消息数={msg_count} ──\033[0m")


# ═══════════════════════════════════════════════════════════
#  Agent Loop — 核心循环
# ═══════════════════════════════════════════════════════════

def agent_loop(messages: list):
    """
    核心循环：调用模型 → 执行工具 → 反馈结果 → 重复
    直到模型不再调用工具为止。
    """
    rounds_since_todo = 0
    reactive_retries = 0
    max_turns = 80

    for turn in range(max_turns):
        # 打印当前状态
        print_status(turn, len(messages))

        # todo reminder 注入（来自 s05 模式）
        reminder = build_reminder(rounds_since_todo)
        if reminder:
            messages.append({"role": "user", "content": reminder})
            rounds_since_todo = 0

        # 上下文压缩管线（来自 s08 模式）
        prepare_context(messages, client=client, model=MODEL)

        # 调用模型
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                tools=TOOLS,
                max_tokens=4096,
            )
            reactive_retries = 0
        except Exception as e:
            err_str = str(e).lower()
            if ("too many tokens" in err_str or "context_length" in err_str) \
                    and reactive_retries < 1:
                print("[reactive compact] 上下文超限，紧急压缩...")
                messages[:] = reactive_compact(messages, client, MODEL)
                reactive_retries += 1
                continue
            raise

        msg = response.choices[0].message

        # 将 assistant 消息追加到历史（转为可序列化 dict）
        assistant_msg = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        # 如果模型没有调用工具 → 输出最终文本，退出循环
        if not msg.tool_calls:
            if msg.content:
                print(f"\n\033[32m[Agent 输出]\033[0m\n{msg.content}")
            return

        # 如果模型有思考文本（同时也调用了工具），显示出来
        if msg.content:
            print(f"\033[33m[Agent 思考] {msg.content[:200]}\033[0m")

        # 执行每个工具调用
        rounds_since_todo += 1
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = tool_call.function.arguments

            # 更详细的工具调用日志
            print(f"\033[36m> {func_name}\033[0m", end="")
            if func_name == "python_execute":
                args = json.loads(func_args) if isinstance(func_args, str) else func_args
                code_preview = args.get("code", "")[:80].replace("\n", " ")
                print(f" | {code_preview}...")
            elif func_name == "web_search":
                args = json.loads(func_args) if isinstance(func_args, str) else func_args
                print(f" | query={args.get('query', '')}")
            elif func_name == "inspect_data":
                args = json.loads(func_args) if isinstance(func_args, str) else func_args
                print(f" | {args.get('path', '')}")
            elif func_name == "read_file":
                args = json.loads(func_args) if isinstance(func_args, str) else func_args
                print(f" | {args.get('path', '')}")
            elif func_name == "list_files":
                args = json.loads(func_args) if isinstance(func_args, str) else func_args
                print(f" | {args.get('directory', '.')}")
            elif func_name == "add_notebook_cell":
                args = json.loads(func_args) if isinstance(func_args, str) else func_args
                print(f" | {args.get('cell_type', '')} cell")
            elif func_name == "todo_write":
                print(" | 更新任务列表")
            else:
                print()

            # 分发执行
            result = dispatch(func_name, func_args)

            # 工具结果输出：inspect_data 和 todo_write 显示更多，其他截断
            if func_name == "inspect_data":
                print(f"\033[90m{result[:500]}\033[0m")
            elif func_name == "todo_write":
                pass  # run_todo_write 内部已经打印了
            elif func_name == "python_execute" and len(result) > 200:
                print(f"  \033[90m{result[:300]}\033[0m")
            else:
                print(f"  \033[90m{result[:150]}\033[0m")

            # 追加 tool 结果消息
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

            # 重置 todo 计数器
            if func_name == "todo_write":
                rounds_since_todo = 0

    print("\n[警告] 达到最大轮次限制，Agent 停止。")


# ═══════════════════════════════════════════════════════════
#  入口：从 Markdown 文件读取题目
# ═══════════════════════════════════════════════════════════

PROBLEM_FILE = "problem.md"


def load_problem(path: str = None) -> str:
    """从指定 Markdown 文件读取建模题目。"""
    target = Path(path) if path else Path(__file__).parent / PROBLEM_FILE
    if not target.exists():
        print(f"\n[错误] 题目文件不存在: {target}")
        print(f"请在以下位置创建题目文件并写入建模题目：")
        print(f"  {target.resolve()}")
        print(f"\n示例内容：")
        print(f"  # 题目：房价预测")
        print(f"  根据附件 data.csv 中的数据，建立多元线性回归模型...")
        sys.exit(1)
    content = target.read_text(encoding="utf-8").strip()
    if not content:
        print(f"\n[错误] 题目文件为空: {target}")
        sys.exit(1)
    return content


def main():
    # 支持命令行指定题目文件，默认 problem.md
    problem_path = sys.argv[1] if len(sys.argv) > 1 else None
    problem = load_problem(problem_path)

    print("=" * 50)
    print("  数学建模 Auto Agent")
    print("=" * 50)
    print(f"模型: {MODEL}")
    print(f"题目文件: {problem_path or PROBLEM_FILE}")
    print(f"输出目录: {Path(__file__).parent / 'output'}")
    print("-" * 50)
    print(problem[:200] + ("..." if len(problem) > 200 else ""))
    print("-" * 50)
    print("\nAgent 开始工作...\n")

    history = [{"role": "user", "content": f"请完成以下数学建模题目：\n\n{problem}"}]
    agent_loop(history)

    # 保存 notebook
    from tools import NOTEBOOK
    if NOTEBOOK and NOTEBOOK.get_cell_count() > 1:
        nb_path = Path(__file__).parent / "output" / "modeling.ipynb"
        print(f"\n\033[32mNotebook 已保存: {nb_path}\033[0m")

    print("\n[完成] Agent 已结束工作，请查看 output/ 目录。")


if __name__ == "__main__":
    main()
