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
import time
from pathlib import Path

# Windows 终端 UTF-8 输出
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.system("chcp 65001 >nul 2>&1")

from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError

# 确保能 import 同目录模块
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
    LLM_MAX_ATTEMPTS,
    LLM_MAX_TOKENS,
    AGENT_MAX_TURNS,
    SUBAGENT_MAX_TURNS,
    SUBAGENT_CONCURRENCY,
)
from tools import TOOLS, PARENT_TOOLS, CHILD_TOOLS, dispatch, CURRENT_TODOS, get_notebook
from compaction import prepare_context, reactive_compact, estimate_size
from prompts import build_system_prompt, build_reminder

load_dotenv(override=True)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url=os.environ.get("OPENAI_BASE_URL"),
    max_retries=LLM_MAX_RETRIES,
    timeout=LLM_TIMEOUT,
)
MODEL = os.environ.get("MODEL_ID", "deepseek-chat")
SYSTEM_PROMPT = build_system_prompt()

# 重试配置
MAX_LLM_ATTEMPTS = LLM_MAX_ATTEMPTS


# ═══════════════════════════════════════════════════════════
#  LLM 调用封装（带重试）
# ═══════════════════════════════════════════════════════════

def _repair_messages(messages: list) -> list:
    """修复消息完整性：确保每个 assistant(tool_calls) 后紧跟对应的 tool 消息。"""
    repaired = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        repaired.append(msg)

        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
            expected_ids = {tc["id"] for tc in msg["tool_calls"]}
            found_ids = set()
            j = i + 1
            while j < len(messages) and isinstance(messages[j], dict) and messages[j].get("role") == "tool":
                found_ids.add(messages[j].get("tool_call_id"))
                j += 1

            missing_ids = expected_ids - found_ids
            if missing_ids:
                for k in range(i + 1, j):
                    repaired.append(messages[k])
                for tid in missing_ids:
                    repaired.append({
                        "role": "tool",
                        "tool_call_id": tid,
                        "content": "[结果已在上下文压缩时丢失]"
                    })
                i = j
                continue

        i += 1
    return repaired


def call_llm(system_prompt: str, messages: list, tools: list, max_tokens: int = LLM_MAX_TOKENS):
    """带重试的 LLM 调用。内置 max_retries 处理瞬时错误，这里处理连续失败。"""
    clean_messages = _repair_messages(messages)

    for attempt in range(MAX_LLM_ATTEMPTS):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system_prompt}] + clean_messages,
                tools=tools,
                max_tokens=max_tokens,
            )
        except (APIConnectionError, APITimeoutError) as e:
            if attempt < MAX_LLM_ATTEMPTS - 1:
                wait = 10 * (attempt + 1)
                print(f"\033[31m[网络错误] {e.__class__.__name__}，{wait}s 后重试 ({attempt+1}/{MAX_LLM_ATTEMPTS})...\033[0m")
                time.sleep(wait)
            else:
                raise
        except RateLimitError as e:
            if attempt < MAX_LLM_ATTEMPTS - 1:
                wait = 30 * (attempt + 1)
                print(f"\033[31m[限流] {wait}s 后重试 ({attempt+1}/{MAX_LLM_ATTEMPTS})...\033[0m")
                time.sleep(wait)
            else:
                raise


# ═══════════════════════════════════════════════════════════
#  Subagent 运行器（s04 模式）
# ═══════════════════════════════════════════════════════════

def run_subagent(prompt: str) -> str:
    """独立上下文执行子任务，只返回摘要文本。不污染父 agent 上下文。"""
    from prompts import build_subagent_prompt
    from tools import CHILD_TOOLS, dispatch

    sub_system = build_subagent_prompt()
    sub_messages = [{"role": "user", "content": prompt}]
    max_sub_turns = SUBAGENT_MAX_TURNS

    print(f"\033[35m[subagent 启动] {prompt[:80]}...\033[0m")

    for turn in range(max_sub_turns):
        response = call_llm(sub_system, sub_messages, CHILD_TOOLS, max_tokens=4096)
        msg = response.choices[0].message

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
        sub_messages.append(assistant_msg)

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            func_name = tc.function.name
            print(f"\033[35m  [sub] > {func_name}\033[0m")
            result = dispatch(func_name, tc.function.arguments)
            sub_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })

    summary = msg.content or "(子任务无输出)"
    print(f"\033[35m[subagent 完成] 返回 {len(summary)} 字符摘要\033[0m")
    return summary


def run_parallel_subagents(prompts: list) -> str:
    """并发执行多个 subagent，并发数受 SUBAGENT_CONCURRENCY 限制。"""
    from concurrent.futures import ThreadPoolExecutor

    if isinstance(prompts, str):
        prompts = [prompts]
    prompts = [p for p in prompts if p and str(p).strip()]
    if not prompts:
        return "(parallel_tasks: 未提供有效的子任务)"

    workers = min(SUBAGENT_CONCURRENCY, len(prompts))
    print(f"\033[35m[parallel] 启动 {len(prompts)} 个子任务，并发数={workers}\033[0m")

    results: list = [None] * len(prompts)

    def _run_one(idx: int, p: str):
        try:
            return idx, run_subagent(f"[子任务 #{idx+1}] {p}")
        except Exception as e:
            return idx, f"(子任务 #{idx+1} 执行失败: {e})"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, i, p) for i, p in enumerate(prompts)]
        for fut in futures:
            idx, summary = fut.result()
            results[idx] = summary

    parts = [f"=== 子任务 #{i+1} 结果 ===\n{r}" for i, r in enumerate(results)]
    print(f"\033[35m[parallel] {len(prompts)} 个子任务全部完成\033[0m")
    return "\n\n".join(parts)


# 持久化 todo 文件路径
TODO_FILE = Path(__file__).parent / "output" / ".agent_todo.json"


# ═══════════════════════════════════════════════════════════
#  Todo 持久化：跨次执行的任务延续
# ═══════════════════════════════════════════════════════════

def save_todo():
    """将当前 todo 列表保存到磁盘。"""
    from tools import CURRENT_TODOS
    TODO_FILE.parent.mkdir(parents=True, exist_ok=True)
    TODO_FILE.write_text(json.dumps(CURRENT_TODOS, ensure_ascii=False, indent=2),
                         encoding="utf-8")


def load_todo() -> list[dict] | None:
    """加载上次未完成的 todo 列表，如果存在且有未完成项。"""
    if not TODO_FILE.exists():
        return None
    try:
        todos = json.loads(TODO_FILE.read_text(encoding="utf-8"))
        pending = [t for t in todos if t.get("status") != "completed"]
        if pending:
            return todos
        TODO_FILE.unlink(missing_ok=True)
        return None
    except (json.JSONDecodeError, KeyError):
        return None


def format_todo_for_prompt(todos: list[dict]) -> str:
    """将 todo 列表格式化为注入 prompt 的文本。"""
    icons = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
    lines = ["[上次执行未完成，以下是待续任务列表]："]
    for t in todos:
        lines.append(f"  {icons.get(t['status'], '[ ]')} {t['content']}")
    lines.append("\n请从上次中断的位置继续工作。已完成的步骤不要重复。")
    return "\n".join(lines)


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
    max_turns = AGENT_MAX_TURNS

    for turn in range(max_turns):
        # 每 5 轮自动保存历史（防硬杀丢进度）
        if turn > 0 and turn % 5 == 0:
            save_history(messages)

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
            response = call_llm(SYSTEM_PROMPT, messages, PARENT_TOOLS)
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

            # 安全解析参数（LLM 可能返回截断的 JSON）
            try:
                parsed_args = json.loads(func_args) if isinstance(func_args, str) else func_args
            except (json.JSONDecodeError, ValueError) as e:
                print(f"\033[36m> {func_name}\033[0m")
                err_msg = (
                    f"Error: 工具参数 JSON 解析失败 — {e}\n"
                    f"原始参数(前200字符): {func_args[:200] if isinstance(func_args, str) else func_args}\n"
                    f"请重新生成完整的工具调用。"
                )
                print(f"  \033[31m{err_msg[:150]}\033[0m")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": err_msg
                })
                continue

            # 更详细的工具调用日志
            print(f"\033[36m> {func_name}\033[0m", end="")
            if func_name == "python_execute":
                code_preview = parsed_args.get("code", "")[:80].replace("\n", " ")
                print(f" | {code_preview}...")
            elif func_name == "web_search":
                print(f" | query={parsed_args.get('query', '')}")
            elif func_name == "inspect_data":
                print(f" | {parsed_args.get('path', '')}")
            elif func_name == "read_file":
                print(f" | {parsed_args.get('path', '')}")
            elif func_name == "list_files":
                print(f" | {parsed_args.get('directory', '.')}")
            elif func_name == "add_notebook_cell":
                print(f" | {parsed_args.get('cell_type', '')} cell")
            elif func_name == "todo_write":
                print(" | 更新任务列表")
            elif func_name == "task":
                print(f" | 子任务: {parsed_args.get('prompt', '')[:60]}...")
            else:
                print()

            # 分发执行
            result = dispatch(func_name, func_args)

            # 工具结果输出：inspect_data 和 todo_write 显示更多，其他截断
            if func_name == "inspect_data":
                print(f"\033[90m{result[:500]}\033[0m")
            elif func_name == "todo_write":
                pass  # run_todo_write 内部已经打印了
            elif func_name == "task":
                print(f"  \033[35m{result[:300]}\033[0m")
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
                save_todo()

    print("\n[警告] 达到最大轮次限制，Agent 停止。")


# ═══════════════════════════════════════════════════════════
#  入口：从 Markdown 文件读取题目
# ═══════════════════════════════════════════════════════════

HISTORY_FILE = Path(__file__).parent / "output" / ".history.json"


def save_history(messages: list):
    """持久化对话历史到磁盘，用于崩溃恢复。"""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(
        json.dumps(messages, default=str, ensure_ascii=False, indent=1),
        encoding="utf-8"
    )
    print(f"\033[90m[历史已保存: {HISTORY_FILE}]\033[0m")


def load_history() -> list | None:
    """尝试加载上次崩溃保存的历史记录。"""
    if not HISTORY_FILE.exists():
        return None
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list) and len(data) > 0:
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def clear_history():
    """正常结束后清理历史文件。"""
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()


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
    # 特殊参数 --resume 从崩溃历史恢复
    args = sys.argv[1:]
    resume_mode = "--resume" in args
    if resume_mode:
        args.remove("--resume")
    problem_path = args[0] if args else None
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

    # 尝试从崩溃历史恢复
    history = None
    if resume_mode:
        history = load_history()
        if history:
            print(f"\n\033[33m[恢复模式] 从上次中断处继续，已有 {len(history)} 条消息\033[0m")
            # 恢复 notebook
            nb_file = Path(__file__).parent / "output" / "modeling.ipynb"
            if nb_file.exists():
                import tools
                from notebook_builder import NotebookBuilder
                tools.NOTEBOOK = NotebookBuilder.load(str(nb_file))
                print(f"\033[33m[续做] 已加载现有 notebook：{tools.NOTEBOOK.get_cell_count()} cells\033[0m")
        else:
            # 没有完整历史，但可能有 todo 可续做
            prev_todos = load_todo()
            if prev_todos:
                print(f"\n\033[33m[续做模式] 从上次任务列表继续：\033[0m")
                icons = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
                for t in prev_todos:
                    print(f"  {icons.get(t['status'], '[ ]')} {t['content']}")
                import tools
                tools.CURRENT_TODOS = prev_todos

                # 恢复 notebook：从已有 modeling.ipynb 加载，使新 cell 追加而非覆盖
                nb_file = Path(__file__).parent / "output" / "modeling.ipynb"
                if nb_file.exists():
                    from notebook_builder import NotebookBuilder
                    tools.NOTEBOOK = NotebookBuilder.load(str(nb_file))
                    print(f"\033[33m[续做] 已加载现有 notebook：{tools.NOTEBOOK.get_cell_count()} cells\033[0m")

                continuation = format_todo_for_prompt(prev_todos)
                history = [
                    {"role": "user", "content": f"请完成以下数学建模题目：\n\n{problem}"},
                    {"role": "user", "content": continuation}
                ]
            else:
                print(f"\n\033[33m[恢复模式] 未找到历史记录，从头开始\033[0m")
    elif load_history() or load_todo():
        print(f"\n\033[33m[提示] 检测到上次未完成的记录，使用 --resume 可恢复：")
        print(f"  python agent.py --resume\033[0m")

    if not history:
        history = [{"role": "user", "content": f"请完成以下数学建模题目：\n\n{problem}"}]

    print("\nAgent 开始工作...\n")

    try:
        agent_loop(history)
        clear_history()
        # 检查是否全部完成
        from tools import CURRENT_TODOS
        if CURRENT_TODOS and all(t["status"] == "completed" for t in CURRENT_TODOS):
            TODO_FILE.unlink(missing_ok=True)
    except KeyboardInterrupt:
        print("\n\n\033[33m[中断] 用户手动停止\033[0m")
        save_history(history)
        save_todo()
        print("下次使用 --resume 可继续：python agent.py --resume")
    except Exception as e:
        print(f"\n\n\033[31m[异常退出] {type(e).__name__}: {e}\033[0m")
        save_history(history)
        save_todo()
        print("下次使用 --resume 可从中断处继续：python agent.py --resume")
        raise

    # 保存 notebook
    from tools import NOTEBOOK
    if NOTEBOOK and NOTEBOOK.get_cell_count() > 1:
        nb_path = Path(__file__).parent / "output" / "modeling.ipynb"
        print(f"\n\033[32mNotebook 已保存: {nb_path}\033[0m")

    print("\n[完成] Agent 已结束工作，请查看 output/ 目录。")


if __name__ == "__main__":
    main()
