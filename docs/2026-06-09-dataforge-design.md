# DataForge — Automated Data-to-Fine-Tuning Platform

**Date:** 2026-06-09
**Status:** Design Approved (pending implementation)

---

## 1. Problem

Pre-trained models are commoditized. The bottleneck is turning messy raw data into a high-quality fine-tuning-ready dataset — a process that currently takes data engineers weeks of manual wrangling. DataForge automates this pipeline from raw file upload to a deployable fine-tuned model.

## 2. Scope (MVP)

### What it does
- Accept Excel/CSV file uploads
- Auto-profile data (quality report via ydata-profiling)
- Clean data (dedup, impute, drop high-null columns, parse dates)
- Show before/after comparison in the UI
- Format cleaned data into instruction JSONL and/or chat template JSONL
- Optionally fine-tune a small model locally (QLoRA) or generate a Colab notebook for larger models
- Deliver download links for dataset + model + push-to-Hub option

### Out of scope (for MVP)
- Database connectors (Postgres, etc.)
- Cloud storage ingestion (S3, GCS)
- API-based ingestion (scheduled pulls)
- Embedding-based intelligent filtering
- DVC / LakeFS versioning
- Multi-user auth
- Pipeline orchestration (Prefect / Dagster)

---

## 3. Architecture

### 3.1 Stack

| Layer | Technology |
|-------|-----------|
| Frontend / UI | Streamlit |
| Data processing | pandas, ydata-profiling |
| Fine-tuning (local) | transformers + peft + bitsandbytes (QLoRA) |
| Fine-tuning (remote) | Generated Colab notebook |
| Model hub | Hugging Face Hub |
| Deployment | Docker -> Hugging Face Spaces |

### 3.2 Project Structure

```
dataforge/
├── app.py                  # Main Streamlit entry point (4 screens)
├── pipeline/
│   ├── __init__.py
│   ├── ingest.py           # Read Excel/CSV via pandas
│   ├── profile.py          # ydata-profiling wrapper
│   ├── clean.py            # Dedup, impute, drop high-null, parse dates
│   └── format.py           # Instruction JSONL + Chat JSONL formatters
├── finetune/
│   ├── __init__.py
│   ├── train_local.py      # QLoRA script (TinyLlama / Phi-3)
│   └── colab_notebook.py   # Generates downloadable .ipynb
├── utils/
│   ├── __init__.py
│   └── display.py          # Before/after table + diff helpers
├── output/                 # Timestamped subdirs for datasets + models
├── requirements.txt
├── Dockerfile
└── README.md
```

### 3.3 Data Flow

```
User uploads file
    |
    v
[Ingest] -> pandas DataFrame
    |
    v
[Profile] -> ydata-profiling HTML report (shown in UI)
    |
    v
[Clean] -> togglable steps: drop null cols, dedup, impute, parse dates
    |
    v
[Preview] -> side-by-side before/after table with diff highlights
    |
    v
[Format] -> Instruction JSONL + Chat JSONL (user selects)
    |
    v
[Export] -> download buttons + optional push to HF Hub
    |
    v
[Fine-Tune] -> local QLoRA (small models) OR Colab notebook (large)
```

---

## 4. UI Screens (Streamlit)

### Screen 1 -- Upload & Configure
- File uploader (CSV, XLSX)
- Auto-detect delimiter, encoding, header row
- Preview first 5 rows
- Column tagging: text column, label column (optional), columns to drop

### Screen 2 -- Profile & Clean
- Expandable ydata-profiling report
- Togglable cleaning rules with descriptions
- Side-by-side before/after table (original vs cleaned)
- Row/column count diff displayed prominently

### Screen 3 -- Format & Export
- Format selector: Instruction JSONL / Chat JSONL / Both
- Preview first 3 formatted rows
- Download: clean CSV, formatted JSONL(s)
- Optional: push the formatted dataset to Hugging Face Hub (token + repo name input)

### Screen 4 -- Fine-Tune (Optional)
- Model size selector
- Local: dropdown -> TinyLlama-1.1B / Phi-3-mini; epochs slider (1-5)
- Remote: "Generate Colab Notebook" button -> downloads .ipynb
- Live training log output (streaming)
- Model card + download link after completion

---

## 5. Pipeline Modules

### 5.1 `pipeline/ingest.py`
- `read_file(uploaded_file) -> (DataFrame, metadata)`
- Handles CSV (with encoding detection via `chardet`) and XLSX
- Metadata: filename, shape, dtypes, detected delimiter

### 5.2 `pipeline/profile.py`
- `generate_profile(df) -> HTML string`
- Wraps `ydata-profiling.ProfileReport`
- Returns report as HTML for embedding in Streamlit

### 5.3 `pipeline/clean.py`
- Each function takes `(df, **params) -> (df, changes_dict)`
- `drop_high_null(df, threshold=0.5)` -- drops columns exceeding null threshold
- `deduplicate(df, subset=None)` -- removes duplicated rows
- `impute_missing(df)` -- numeric to median, categorical to mode
- `parse_dates(df)` -- auto-detect and convert date columns
- `changes_dict` tracks: rows removed, cols removed, imputed cells

### 5.4 `pipeline/format.py`
- `to_instruction_jsonl(df, text_col, label_col=None) -> str`
- `to_chat_jsonl(df, text_col, label_col=None) -> str`
- Instruction format (with label): `{"instruction": text, "output": label}`
- Instruction format (no label): `{"instruction": text, "output": ""}`
- Chat format (with label): `{"messages": [{"role": "user", "content": text}, {"role": "assistant", "content": label}]}`
- Chat format (no label): includes only the user message, no assistant turn

### 5.5 `finetune/train_local.py`
- Loads formatted JSONL via `datasets.load_dataset()`
- Applies QLoRA (4-bit quantization, LoRA rank=8)
- Target model: TinyLlama-1.1B-Chat (default) or Phi-3-mini
- Training args: epochs (slider), batch size (auto-computed from memory)
- Saves LoRA adapter + merged model to `output/{timestamp}/model/`
- Writes logs to a temp file; Streamlit polls via `st.empty()` + periodic refresh to render live output

### 5.6 `finetune/colab_notebook.py`
- Generates `.ipynb` using `nbformat`
- Cells: install deps, load formatted dataset from HF Hub, run LoRA training, save fine-tuned model to HF Hub
- User fills HF token + repo name in notebook cells

---

## 6. Deployment

- **Single Docker container** with Streamlit + all dependencies
- Deployed on **Hugging Face Spaces** (Docker runtime, free tier)
- Dockerfile installs pip deps, exposes port 7860 (Streamlit default)
- Model weights for TinyLlama/Phi-3 are downloaded at runtime (not bundled)

---

## 7. Error Handling

- File upload: validate extension, encoding, minimum rows (>=1)
- Cleaning: wrap each step individually so one failure doesn't block others
- Fine-tuning: catch OOM errors (suggest smaller model or Colab path)
- All errors surfaced inline in Streamlit via `st.error()` with actionable messages

---

## 8. Testing

- Unit tests for each `pipeline/` function (pytest)
- Test with synthetic data: nulls, duplicates, mixed types, edge cases
- Format tests: verify JSONL structure matches expected schema
- Fine-tuning tests: run 1 epoch on a 50-row synthetic dataset to verify no crashes

---

## 9. Future Iterations (Post-MVP)

- Database connectors (Postgres, MySQL)
- Embedding-based intelligent filtering (sentence-transformers + FAISS)
- DVC / LakeFS versioning for data lineage
- Prefect orchestration for scheduled pipelines
- Multi-user support with session isolation
- Support for image datasets (folder-based formatting)
