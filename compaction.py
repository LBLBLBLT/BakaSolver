"""
compaction.py - 上下文压缩管线

移植自 learn-claude-code s08，适配 OpenAI 消息格式。

OpenAI 消息格式的关键差异：
- tool 结果是独立消息 {"role": "tool", "tool_call_id": ..., "content": ...}
- assistant 消息的 tool_calls 字段（非 content blocks）
- 配对保护：不在 assistant(含tool_calls) 和紧跟的 role=tool 消息之间切断

四层管线（cheap first, expensive last）：
  L1: tool_result_budget — 大输出落盘
  L2: snip_compact — 裁剪中间消息
  L3: micro_compact — 旧 tool 消息占位替换
  L4: compact_history — 模型摘要（1 次 API 调用）
  Emergency: reactive_compact — API 报错后紧急压缩
"""

import json
import time
from pathlib import Path

from config import (
    CONTEXT_LIMIT,
    KEEP_RECENT_TOOLS,
    PERSIST_THRESHOLD,
    MAX_MESSAGES,
    COMPACT_SUMMARY_TOKENS,
    REACTIVE_SUMMARY_TOKENS,
)

TRANSCRIPT_DIR = Path.cwd() / "output" / ".transcripts"
PERSIST_DIR = Path.cwd() / "output" / ".tool_outputs"


def estimate_size(messages: list) -> int:
    return len(json.dumps(messages, default=str, ensure_ascii=False))


def _has_tool_calls(msg) -> bool:
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        return True
    if isinstance(msg, dict):
        return bool(msg.get("tool_calls"))
    return False


def _is_tool_message(msg) -> bool:
    if isinstance(msg, dict):
        return msg.get("role") == "tool"
    return getattr(msg, "role", None) == "tool"


def _get_role(msg) -> str:
    if isinstance(msg, dict):
        return msg.get("role", "")
    return getattr(msg, "role", "")


def _get_content(msg) -> str:
    if isinstance(msg, dict):
        return msg.get("content") or ""
    return getattr(msg, "content", "") or ""


def _set_content(msg, content: str):
    if isinstance(msg, dict):
        msg["content"] = content
    else:
        msg.content = content


# ═══════════════════════════════════════════════════════════
#  L1: tool_result_budget — 大输出持久化到磁盘
# ═══════════════════════════════════════════════════════════

def tool_result_budget(messages: list, max_bytes: int = 150_000) -> list:
    tool_msgs = [(i, m) for i, m in enumerate(messages) if _is_tool_message(m)]
    total = sum(len(_get_content(m)) for _, m in tool_msgs)
    if total <= max_bytes:
        return messages

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    ranked = sorted(tool_msgs, key=lambda p: len(_get_content(p[1])), reverse=True)
    for _, msg in ranked:
        content = _get_content(msg)
        if len(content) <= PERSIST_THRESHOLD:
            continue
        tid = msg.get("tool_call_id", "unknown") if isinstance(msg, dict) else "unknown"
        path = PERSIST_DIR / f"{tid}.txt"
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        preview = content[:1500]
        _set_content(msg, f"[输出已持久化: {path}]\n预览:\n{preview}\n...")
        total = sum(len(_get_content(m)) for _, m in tool_msgs)
        if total <= max_bytes:
            break
    return messages


# ═══════════════════════════════════════════════════════════
#  L2: snip_compact — 裁剪中间消息，保留首尾
# ═══════════════════════════════════════════════════════════

def snip_compact(messages: list, max_messages: int = MAX_MESSAGES) -> list:
    if len(messages) <= max_messages:
        return messages

    keep_head = 4
    keep_tail = max_messages - keep_head - 1
    head_end = keep_head
    tail_start = len(messages) - keep_tail

    # 配对保护：不在 assistant(tool_calls) 和紧跟的 tool 消息之间切断
    if head_end > 0 and _has_tool_calls(messages[head_end - 1]):
        while head_end < len(messages) and _is_tool_message(messages[head_end]):
            head_end += 1

    if tail_start > 0 and _is_tool_message(messages[tail_start]):
        tail_start -= 1
        while tail_start > 0 and _is_tool_message(messages[tail_start]):
            tail_start -= 1

    if head_end >= tail_start:
        return messages

    snipped = tail_start - head_end
    placeholder = {"role": "user", "content": f"[已裁剪 {snipped} 条中间消息]"}
    return messages[:head_end] + [placeholder] + messages[tail_start:]


# ═══════════════════════════════════════════════════════════
#  L3: micro_compact — 旧 tool 消息内容替换为占位符
# ═══════════════════════════════════════════════════════════

def micro_compact(messages: list) -> list:
    tool_indices = [i for i, m in enumerate(messages) if _is_tool_message(m)]
    if len(tool_indices) <= KEEP_RECENT_TOOLS:
        return messages

    old_indices = tool_indices[:-KEEP_RECENT_TOOLS]
    for i in old_indices:
        content = _get_content(messages[i])
        if len(content) > 100:
            _set_content(messages[i], "[早期工具输出已压缩，如需可重新执行]")
    return messages


# ═══════════════════════════════════════════════════════════
#  L4: compact_history — 调用模型生成摘要
# ═══════════════════════════════════════════════════════════

def write_transcript(messages: list) -> Path:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str, ensure_ascii=False) + "\n")
    return path


def compact_history(messages: list, client, model: str) -> list:
    """用模型生成摘要替换全部历史。需要传入 client 和 model。"""
    write_transcript(messages)

    conversation = json.dumps(messages, default=str, ensure_ascii=False)[:60000]
    prompt = (
        "请摘要以下数学建模 Agent 的对话历史，以便工作可以继续。\n"
        "必须保留：1.当前建模目标 2.已完成的步骤和关键发现 "
        "3.已读取/生成的文件 4.剩余工作 5.用户约束条件\n"
        "重点：明确标注哪些步骤已经完成、得出了什么具体结论/数值，"
        "后续 Agent 不应重复已完成的工作。\n"
        "注意：你只需要输出纯文本摘要，不要生成任何 tool_calls、XML标签或代码调用。\n"
        "简洁但具体。\n\n" + conversation
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=COMPACT_SUMMARY_TOKENS
    )
    summary = response.choices[0].message.content or "(空摘要)"

    print("\n" + "=" * 50)
    print("  [上下文压缩] 以下是当前进度摘要：")
    print("-" * 50)
    print(summary)
    print("=" * 50 + "\n")

    result = [{"role": "user", "content": f"[上下文已压缩]\n\n{summary}"}]
    todo_state = _get_current_todo_text()
    if todo_state:
        result.append({"role": "user", "content": todo_state})
    nb_manifest = _get_notebook_manifest_text()
    if nb_manifest:
        result.append({"role": "user", "content": nb_manifest})
    return result


# ═══════════════════════════════════════════════════════════
#  Emergency: reactive_compact — API 报错后的紧急压缩
# ═══════════════════════════════════════════════════════════

def reactive_compact(messages: list, client, model: str) -> list:
    """保留摘要 + 最近几条消息。"""
    write_transcript(messages)

    conversation = json.dumps(messages, default=str, ensure_ascii=False)[:60000]
    prompt = (
        "紧急压缩：摘要以下对话，保留关键进展和目标。\n\n" + conversation
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=REACTIVE_SUMMARY_TOKENS
    )
    summary = response.choices[0].message.content or "(空摘要)"

    print("\n" + "=" * 50)
    print("  [紧急压缩] 上下文超限，以下是保留的摘要：")
    print("-" * 50)
    print(summary)
    print("=" * 50 + "\n")

    tail_start = max(0, len(messages) - 6)
    if tail_start > 0 and _is_tool_message(messages[tail_start]):
        tail_start -= 1

    result = [{"role": "user", "content": f"[紧急压缩]\n\n{summary}"}]
    todo_state = _get_current_todo_text()
    if todo_state:
        result.append({"role": "user", "content": todo_state})
    nb_manifest = _get_notebook_manifest_text()
    if nb_manifest:
        result.append({"role": "user", "content": nb_manifest})
    return result + messages[tail_start:]


def _get_current_todo_text() -> str | None:
    """获取当前 todo 状态文本，用于压缩后重新注入。"""
    try:
        from tools import CURRENT_TODOS
        if not CURRENT_TODOS:
            return None
        icons = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
        lines = [
            "[重要：以下是你当前的任务进度，不要重复已完成的工作，从当前进行中的任务继续]：",
        ]
        for t in CURRENT_TODOS:
            lines.append(f"  {icons.get(t['status'], '[ ]')} {t['content']}")
        lines.append("\n严格从 [>] 标记的任务继续执行，已完成的步骤不要重复。")
        return "\n".join(lines)
    except ImportError:
        return None


def _get_notebook_manifest_text() -> str | None:
    """获取已记录到 notebook 的真实步骤清单，压缩后注入，防止遗忘/重复/编造。"""
    try:
        from tools import _read_manifest
        manifest = _read_manifest()
        if not manifest:
            return None
        lines = [
            "[以下是已经真实执行并记录进 notebook 的代码步骤（cell 顺序）。"
            "这些步骤已完成且产出已落盘，不要重复执行，也不要在 notebook 里重写它们]：",
        ]
        for i, entry in enumerate(manifest):
            note = entry.get("note") or "(无说明)"
            preview = (entry.get("code_preview") or "").replace("\n", " ")[:80]
            lines.append(f"  cell[{entry.get('cell_index', '?')}] {note} | 代码: {preview}...")
        lines.append("\n后续新步骤继续用 python_execute(record_to_notebook=true) 追加记录。")
        return "\n".join(lines)
    except (ImportError, Exception):
        return None


# ═══════════════════════════════════════════════════════════
#  配对修复：确保 tool_calls 和 tool 响应一一对应
# ═══════════════════════════════════════════════════════════

def repair_tool_pairs(messages: list) -> list:
    """修复压缩后可能产生的 tool_calls/tool 不配对问题。"""
    result = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if _has_tool_calls(msg):
            result.append(msg)
            if isinstance(msg, dict):
                tc_ids = [tc["id"] for tc in msg.get("tool_calls", [])]
            else:
                tc_ids = [tc.id for tc in (msg.tool_calls or [])]

            needed = set(tc_ids)
            found = set()
            j = i + 1

            while j < len(messages) and _is_tool_message(messages[j]):
                tool_msg = messages[j]
                tid = tool_msg.get("tool_call_id") if isinstance(tool_msg, dict) else getattr(tool_msg, "tool_call_id", None)
                if tid in needed:
                    result.append(tool_msg)
                    found.add(tid)
                j += 1

            for tid in tc_ids:
                if tid not in found:
                    result.append({
                        "role": "tool",
                        "tool_call_id": tid,
                        "content": "[工具响应在上下文压缩中丢失，请重新执行该工具调用]"
                    })

            i = j
        elif _is_tool_message(msg):
            tid = msg.get("tool_call_id") if isinstance(msg, dict) else getattr(msg, "tool_call_id", None)
            has_parent = False
            for prev in reversed(result):
                if _has_tool_calls(prev):
                    if isinstance(prev, dict):
                        parent_ids = [tc["id"] for tc in prev.get("tool_calls", [])]
                    else:
                        parent_ids = [tc.id for tc in (prev.tool_calls or [])]
                    if tid in parent_ids:
                        has_parent = True
                    break
                elif not _is_tool_message(prev):
                    break
            if has_parent:
                result.append(msg)
            i += 1
        else:
            result.append(msg)
            i += 1

    return result


# ═══════════════════════════════════════════════════════════
#  管线入口
# ═══════════════════════════════════════════════════════════

def prepare_context(messages: list, client=None, model: str = "") -> list:
    """执行四层压缩管线。L4 需要 client 和 model。"""
    messages[:] = tool_result_budget(messages)
    messages[:] = snip_compact(messages)
    messages[:] = micro_compact(messages)

    if client and estimate_size(messages) > CONTEXT_LIMIT:
        print("[auto compact] 上下文过大，正在生成摘要...")
        messages[:] = compact_history(messages, client, model)

    # 最终修复：确保 tool_calls 和 tool 消息配对完整
    messages[:] = repair_tool_pairs(messages)

    return messages
