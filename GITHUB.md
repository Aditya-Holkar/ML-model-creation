# DataForge

Automated data-to-ML pipeline. Upload raw Excel/CSV → profile → clean → format → train a sklearn model or fine-tune an LLM. Zero persistent disk writes — runs entirely in browser memory via Streamlit session state.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/streamlit-1.40%2B-red) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

### Screen 1 — Upload
Drag-and-drop CSV or Excel files. Auto-detect columns, select subset or use entire dataset.

### Screen 2 — Profile & Clean
- Auto-generated data quality profile (missing values, distributions, correlations)
- 10 cleaning operations: drop high-null columns, deduplicate, impute missing values, parse dates, strip whitespace, lowercase text, remove empty rows, replace placeholder nulls, remove outliers, rename to snake_case
- Before/after comparison tables

### Screen 3 — Format
Generate JSONL for fine-tuning:
- **Instruction format**: `instruction` / `input` / `output`
- **Chat format**: `messages` with system/user/assistant roles
- Download JSONL or push directly to Hugging Face Hub

### Screen 4 — Train
3-step flow:
1. **Describe** what you want in plain English (e.g., "classify iris species")
2. **Suggest/Generate** — LLM (Groq) writes the sklearn model definition; validated against whitelist and kwargs-stripped automatically
3. **Train** — fits the model in-memory, shows metrics (accuracy, R², etc.)

**Fallbacks built in**: hallucinated model names → safe default per task type; missing kwargs → stripped; no label column selected → auto-pick last column.

### Screen 5 — Model Playground
4 tabs:
- **Predict** — sliders for single prediction + bulk CSV upload for batch prediction
- **Data Explorer** — histograms, correlation heatmap, scatter matrix, PCA scatter (for clustering)
- **Feature Analysis** — importance bar chart + data summary table
- **Chat** — Groq-powered chat that answers questions about your data and model; auto-detects prediction intents and runs the model on real data

## Quick Start

```bash
pip install -r requirements-core.txt
streamlit run app.py
```

For LLM fine-tuning (torch, transformers):
```bash
pip install -r requirements-finetune.txt
```

## Architecture

```
dataforge/
├── app.py                  # Streamlit entry point (5 screens, ~620 lines)
├── pipeline/
│   ├── ingest.py           # CSV/Excel reader
│   ├── profile.py          # ydata-profiling integration
│   ├── clean.py            # 10 cleaning functions
│   ├── format.py           # JSONL formatting (instruction + chat)
│   └── playground.py       # Screen 5: Predict, Explore, Feature, Chat
├── finetune/
│   ├── model_generator.py  # LLM model snippet generation + validation
│   ├── model_config.py     # Model definitions & shim classes
│   ├── train_local.py      # QLoRA training (TinyLlama)
│   ├── cloud_tune.py       # Together AI fine-tuning API
│   └── colab_notebook.py   # Colab notebook generator
├── utils/
│   └── display.py          # Streamlit UI helpers (before/after, preview)
├── docs/
│   └── future-features.md  # 20 planned features
├── requirements-core.txt   # streamlit, pandas, plotly, etc.
├── requirements-finetune.txt  # torch, transformers, peft, etc.
├── Dockerfile              # HF Spaces compatible (port 7860)
└── .env.example            # API key template
```

## Key Design Decisions

- **Zero persistent disk writes** — all data in `st.session_state`, model code in tempfile for importlib, downloads from `BytesIO`. No `output/` directory usage. Designed for Hugging Face Spaces (ephemeral storage).
- **LLM code validation** — LLM only outputs `MODEL = ...` and metric type; injected into a hardcoded template. Model name whitelist + kwargs stripping prevents runtime errors.
- **Dual requirements** — `requirements-core.txt` (fast install) and `requirements-finetune.txt` (heavy deps separate).
- **Split fine-tuning** — local QLoRA, cloud (Together AI), or Colab notebook generation.

## Deployment

**Hugging Face Spaces** (recommended):
1. Create a Space with Docker SDK
2. Push this repo
3. Set `GROQ_API_KEY` and `TOGETHER_API_KEY` in Space secrets
4. App runs at `https://{username}-dataforge.hf.space`

**Not supported**: Vercel (requires WebSocket server process).

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes (Screen 4 & 5) | Groq API key for model generation + chat |
| `TOGETHER_API_KEY` | No | Together AI key for cloud fine-tuning |
| `HF_TOKEN` | No | Hugging Face token for pushing datasets |

Copy `.env.example` to `.env` and fill in.

## Tech Stack

- **Frontend**: Streamlit
- **ML**: scikit-learn, joblib
- **LLM**: Groq API (llama-3.3-70b-versatile), Together AI
- **Visualization**: plotly, matplotlib, seaborn
- **Fine-tuning**: transformers, peft, bitsandbytes, QLoRA
- **Container**: Docker (python:3.10-slim)

## Future Plans

See [`docs/future-features.md`](docs/future-features.md) for the full roadmap — pipeline templates, experiment tracking, model registry, REST API deployment, data versioning, scheduled automation, multi-user support, collaborative labeling, drift monitoring, and more.

## License

MIT
