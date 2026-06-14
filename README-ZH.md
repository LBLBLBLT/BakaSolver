# BakaSolver

[![English Version](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.md)

一个面向数学建模竞赛的 AI 自主智能体（Autonomous Agent）。只需把题目丢给它，它就能自动完成数据分析、跑模型、画图，并吐出包含代码、分析和 LaTeX 公式的完整报告。

## 核心功能

BakaSolver 读取 Markdown 格式的数学建模题目后，会自动跑完以下全套流程：

1. **题目分析** —— 自动识别建模目标、约束条件和评价指标。
2. **数据探索** —— 自动读取数据文件，只提取结构和统计信息，防止原始数据太大撑爆上下文（Token）。
3. **数据清洗与 EDA** —— 自动编写并运行 Python 代码进行预处理和探索性可视化。
4. **模型构建** —— 根据问题选择并实现合适的算法（回归、优化、机器学习等）。
5. **模型评估** —— 计算评价指标（R²、RMSE、AIC...）、交叉验证以及敏感性分析。
6. **报告生成** —— 自动生成包含 LaTeX 公式和结论的 Markdown 报告。

所有的代码运行轨迹和分析过程都会自动记录到一个 Jupyter Notebook (`.ipynb`) 中。

## 项目亮点

- **智能体闭环（Agentic loop）** —— LLM 自己决定下一步做什么、调什么工具、观察运行结果，并不断循环直到解决问题。
- **轻量化数据检查** —— 支持 CSV/Excel/JSON/Parquet/MAT 等多种文件，只给模型返回 Summary，不把原始明细死脑筋地塞进 Context。
- **四层上下文压缩（Context Compaction）** —— 专门设计了 4 层压缩管线（输出持久化 → 消息裁剪 → 微观紧凑化 → 模型自动摘要），彻底解决长跑时 API 报 Token 超限（Out-Of-Token）的问题。
- **任务看板（Task Tracking）** —— 内置 Todo 系统，自动提醒并逼迫模型更新当前进度状态。
- **联网搜索** —— 整合 DuckDuckGo，允许 Agent 遇到不懂的知识或缺数据时自己去网上查资料。
- **Matplotlib 自动拦截** —— 自动把代码里的 `plt.show()` 替换为 `savefig()`，无头（Headless）环境下也能正常跑自动化脚本画图。

## 快速开始

```bash
git clone [https://github.com/yourname/BakaSolver.git](https://github.com/yourname/BakaSolver.git)
cd BakaSolver
pip install -r requirements.txt
```

创建一个 `.env` 配置文件：

**代码段**

```
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=[https://api.deepseek.com](https://api.deepseek.com)   # 或者任何兼容 OpenAI 格式的中转/官方接口
MODEL_ID=deepseek-chat                      # 你要调用的模型名称
```

把你的题目写进 `problem.md`，把数据集放进 `data/` 目录，然后启动：

**Bash**

```
python agent.py
```

也可以手动指定其他的题目文件：

**Bash**

```
python agent.py 你的题目文件.md
```

运行结果会自动生成在 `output/` 目录下：

* `modeling.ipynb` —— 包含所有完整代码和执行结果的 Jupyter Notebook
* `report.md` —— 包含公式和结论的最终 Markdown 论文报告
* `figure_*.png` —— 自动生成的各类分析图表

## 环境要求

* Python 3.10+
* 任何兼容 OpenAI 格式的 API 接口（DeepSeek、OpenAI、本地大模型等均可）

## 项目结构

```
BakaSolver/
├── agent.py             # 主入口，Agent 核心控制循环
├── prompts.py           # 系统提示词（System Prompt）组装
├── tools.py             # 工具箱定义与具体实现（执行代码、搜索等）
├── compaction.py        # 上下文压缩管线核心逻辑
├── notebook_builder.py  # Jupyter Notebook 自动生成器
├── problem.md           # 默认存放数学建模题目的地方
├── data/                # 存放输入的数据集文件
├── output/              # 存放最终生成的各类成果物
└── requirements.txt
```

## 运行原理

Agent 会进入一个死循环：调用 LLM → 如果模型返回需要调用工具（Tool Calls），则在本地执行工具（如跑 Python 代码或搜网页） → 将执行结果塞回给上下文 → 重复该过程，直到模型自己觉得搞定了，吐出最终文本（或达到 80 轮的最大上限）。

为了防止长对话导致 Token 暴涨，项目内置的压缩系统会在后台盯着：大段的工具输出会被直接持久化到本地磁盘，老旧的历史对话会被模型定时总结成摘要，如果 API 触发了长度报错，还会激活紧急压缩机制。

## 开源协议

MIT
