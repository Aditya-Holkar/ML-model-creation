# DataForge

Automated Data-to-Fine-Tuning Pipeline — upload raw Excel/CSV data, profile it, clean it, format it for fine-tuning, and optionally train a model.

## Prerequisites

- Python 3.10+ installed. If not in PATH, find it with:
  ```powershell
  Get-ChildItem -Path "C:\" -Filter "python.exe" -Recurse -ErrorAction SilentlyContinue -Depth 3
  ```
- Common locations: `C:\Users\<username>\AppData\Local\Programs\Python\Python312\python.exe`

## Setup

### 1. Install core dependencies (fast — needed for Upload, Profile, Clean, Format)

```powershell
& "C:\Users\Holkar\AppData\Local\Programs\Python\Python312\python.exe" -m pip install -r requirements-core.txt
```

### 2. (Optional) Install fine-tuning dependencies (large — torch, transformers)

```powershell
& "C:\Users\Holkar\AppData\Local\Programs\Python\Python312\python.exe" -m pip install -r requirements-finetune.txt
```

### 3. Run the app

```powershell
& "C:\Users\Holkar\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run app.py
```

If `python` is in your PATH, you can use shorter commands:

```powershell
pip install -r requirements-core.txt
streamlit run app.py
```

**PowerShell note:** Use `& "path\with spaces\python.exe"` for spaces. `&&` is not supported — run commands separately.

## Pipeline

1. **Upload** — CSV or Excel file
2. **Profile** — auto-generated quality report
3. **Clean** — dedup, impute, drop high-null columns, parse dates
4. **Format** — instruction JSONL or chat template JSONL
5. **Fine-Tune** — local QLoRA (small models) or Colab notebook (larger models)

## Deployment

Deploy on Hugging Face Spaces as a Docker app:

1. Push this repo to a Hugging Face Space (Docker SDK)
2. Space builds automatically
3. App runs at `https://{username}-dataforge.hf.space`
