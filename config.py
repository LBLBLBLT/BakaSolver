"""
config.py - 统一配置层

所有可调常量集中于此，优先从环境变量读取，否则使用默认值。
在 .env 中设置对应变量即可覆盖（agent.py 启动时会 load_dotenv）。
"""

import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ═══════════════════════════════════════════════════════════
#  LLM 调用
# ═══════════════════════════════════════════════════════════
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 3)        # OpenAI client 内置重试
LLM_TIMEOUT = _env_float("LLM_TIMEOUT", 120.0)          # client 超时（秒）
LLM_MAX_ATTEMPTS = _env_int("LLM_MAX_ATTEMPTS", 3)      # call_llm 外层手动重试次数
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 4096)       # 单次回复最大 token

# ═══════════════════════════════════════════════════════════
#  Agent 循环
# ═══════════════════════════════════════════════════════════
AGENT_MAX_TURNS = _env_int("AGENT_MAX_TURNS", 50)           # 父 agent 单次最大轮数
SUBAGENT_MAX_TURNS = _env_int("SUBAGENT_MAX_TURNS", 30)     # 子 agent 最大轮数
SUBAGENT_CONCURRENCY = _env_int("SUBAGENT_CONCURRENCY", 3)  # 最大并发 subagent 数

# ═══════════════════════════════════════════════════════════
#  python_execute
# ═══════════════════════════════════════════════════════════
PYEXEC_DEFAULT_TIMEOUT = _env_int("PYEXEC_DEFAULT_TIMEOUT", 300)   # 默认超时（秒）
PYEXEC_MAX_TIMEOUT = _env_int("PYEXEC_MAX_TIMEOUT", 1800)          # 最大允许超时（秒）

# ═══════════════════════════════════════════════════════════
#  上下文压缩
# ═══════════════════════════════════════════════════════════
CONTEXT_LIMIT = _env_int("CONTEXT_LIMIT", 60000)                    # 触发摘要压缩的阈值
KEEP_RECENT_TOOLS = _env_int("KEEP_RECENT_TOOLS", 4)               # micro_compact 保留最近工具数
PERSIST_THRESHOLD = _env_int("PERSIST_THRESHOLD", 20000)          # 单条工具输出落盘阈值
MAX_MESSAGES = _env_int("MAX_MESSAGES", 60)                        # snip_compact 最大消息数
COMPACT_SUMMARY_TOKENS = _env_int("COMPACT_SUMMARY_TOKENS", 2000)  # 常规压缩摘要 token
REACTIVE_SUMMARY_TOKENS = _env_int("REACTIVE_SUMMARY_TOKENS", 1500)  # 紧急压缩摘要 token
