# BakaSolver

[![中文版](https://img.shields.io/badge/语言-中文-red?style=for-the-badge)](./README-ZH.md)

AI-powered autonomous agent for mathematical modeling competitions. Feed it a problem, get back a complete solution with code, analysis, and a formatted report.

## What It Does

BakaSolver reads a math modeling problem (in Markdown), then autonomously works through the full pipeline:

1. **Problem Analysis** — Identifies modeling objectives, constraints, and evaluation metrics
2. **Data Acquisition** — Explores data files, inspects structure and statistics without blowing up context
3. **Data Cleaning & EDA** — Runs Python code for preprocessing and exploratory visualization
4. **Model Building** — Selects and implements appropriate models (regression, optimization, ML, etc.)
5. **Model Evaluation** — Computes metrics (R², RMSE, AIC...), cross-validation, sensitivity analysis
6. **Report Generation** — Produces a Markdown report with LaTeX formulas and conclusions

All code and analysis are recorded into a Jupyter Notebook automatically.

## Key Features

- **Agentic loop** — LLM decides what to do next, executes tools, observes results, repeats
- **Smart data inspection** — Handles CSV/Excel/JSON/Parquet/MAT files; returns summaries instead of dumping raw data into context
- **Context compaction** — 4-layer compression pipeline (output persistence → message snipping → micro-compact → model summarization) keeps the conversation within token limits across long runs
- **Task tracking** — Built-in todo system with progress display; auto-reminds the model to update status
- **Web search** — Can search DuckDuckGo for external data sources or references
- **Auto matplotlib** — Converts `plt.show()` to `savefig()` for headless execution

## Quick Start

```bash
git clone https://github.com/yourname/BakaSolver.git
cd BakaSolver
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.deepseek.com   # or any OpenAI-compatible endpoint
MODEL_ID=deepseek-chat                      # model to use
```

Write your problem into `problem.md`, put data files in `data/`, then run:

```bash
python agent.py
```

Or specify a different problem file:

```bash
python agent.py my_problem.md
```

Results go to `output/`:

- `modeling.ipynb` — Full Jupyter Notebook with all code and analysis
- `report.md` — Final report with formulas and conclusions
- `figure_*.png` — Generated plots

## Requirements

- Python 3.10+
- An OpenAI-compatible API endpoint (DeepSeek, OpenAI, local models, etc.)

## Project Structure

```
BakaSolver/
├── agent.py             # Main entry point and agent loop
├── prompts.py           # System prompt assembly
├── tools.py             # Tool definitions and implementations
├── compaction.py        # Context compression pipeline
├── notebook_builder.py  # Jupyter notebook generator
├── problem.md           # Your modeling problem goes here
├── data/                # Input data files
├── output/              # Generated outputs
└── requirements.txt
```

## How It Works

The agent runs a loop: call the LLM → if it requests tool calls, execute them and feed results back → repeat until the model produces a final text response (or hits 80 turns). The context compaction system ensures long-running sessions don't exceed token limits — large tool outputs get persisted to disk, old messages get summarized, and emergency compression kicks in if the API returns a context-length error.

## License

MIT
