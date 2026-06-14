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


CONTEXT_LIMIT = 60000
KEEP_RECENT_TOOLS = 4
PERSIST_THRESHOLD = 20000
MAX_MESSAGES = 60

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
        "简洁但具体。\n\n" + conversation
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )
    summary = response.choices[0].message.content or "(空摘要)"

    # 输出摘要让用户可见
    print("\n" + "=" * 50)
    print("  [上下文压缩] 以下是当前进度摘要：")
    print("-" * 50)
    print(summary)
    print("=" * 50 + "\n")

    return [{"role": "user", "content": f"[上下文已压缩]\n\n{summary}"}]


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
        max_tokens=1500
    )
    summary = response.choices[0].message.content or "(空摘要)"

    # 输出摘要让用户可见
    print("\n" + "=" * 50)
    print("  [紧急压缩] 上下文超限，以下是保留的摘要：")
    print("-" * 50)
    print(summary)
    print("=" * 50 + "\n")

    tail_start = max(0, len(messages) - 6)
    # 配对保护
    if tail_start > 0 and _is_tool_message(messages[tail_start]):
        tail_start -= 1

    return [{"role": "user", "content": f"[紧急压缩]\n\n{summary}"}] + messages[tail_start:]


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

    return messages
