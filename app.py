import os
import json
import tempfile
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import pandas as pd

from pipeline.ingest import read_file
from pipeline.profile import generate_profile
from pipeline.clean import (
    drop_high_null,
    deduplicate,
    impute_missing,
    parse_dates,
    strip_whitespace,
    lowercase_text,
    remove_empty_rows,
    replace_placeholder_nulls,
    remove_outliers,
    rename_to_snake_case,
)
from pipeline.format import to_instruction_jsonl, to_chat_jsonl
from pipeline.playground import show_playground
from utils.display import show_before_after, show_format_preview

MODEL_SESSION_KEYS = ["gen_model_instance", "gen_model_module", "gen_model_code", "gen_metrics", "gen_dir"]

def clear_model_session_state():
    for k in MODEL_SESSION_KEYS:
        if k in st.session_state:
            del st.session_state[k]


AVAILABLE_MODELS = {
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
}

try:
    from finetune.train_local import train_async, MODEL_REGISTRY
    AVAILABLE_MODELS = MODEL_REGISTRY
    HAS_FINETUNE_DEPS = True
except (ImportError, RuntimeError):
    train_async = None
    HAS_FINETUNE_DEPS = False

try:
    from finetune.colab_notebook import generate_colab_notebook
    HAS_COLAB_DEPS = True
except ImportError:
    generate_colab_notebook = None
    HAS_COLAB_DEPS = False

try:
    from finetune.cloud_tune import (
        upload_file,
        create_fine_tune,
        poll_until_done,
        SUPPORTED_MODELS as CLOUD_MODELS,
    )
    HAS_CLOUD_DEPS = True
except ImportError:
    upload_file = create_fine_tune = poll_until_done = None
    CLOUD_MODELS = []
    HAS_CLOUD_DEPS = False

from finetune.model_config import generate_config as generate_model_config
from finetune.model_generator import generate_model, train_generated_model, suggest_model

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

st.set_page_config(page_title="DataForge", layout="wide")
st.title("DataForge — Automated Data to Fine-Tuning Pipeline")

if "df" not in st.session_state:
    st.session_state.df = None
if "df_original" not in st.session_state:
    st.session_state.df_original = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "cleaned" not in st.session_state:
    st.session_state.cleaned = None
if "profile_html" not in st.session_state:
    st.session_state.profile_html = None
if "train_log" not in st.session_state:
    st.session_state.train_log = []
if "training" not in st.session_state:
    st.session_state.training = False
if "gen_dir" not in st.session_state:
    st.session_state.gen_dir = None
if "train_text_cols" not in st.session_state:
    st.session_state.train_text_cols = None
if "train_label_cols" not in st.session_state:
    st.session_state.train_label_cols = None

screen = st.sidebar.radio("Navigate", ["1. Upload & Configure", "2. Profile & Clean", "3. Format & Export", "4. Fine-Tune", "5. Model Playground"])

# ─────────────────────────────────────────────
# SCREEN 1: UPLOAD & CONFIGURE
# ─────────────────────────────────────────────
if screen == "1. Upload & Configure":
    st.header("Upload & Configure")

    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        with st.spinner("Reading file..."):
            try:
                df, metadata = read_file(uploaded_file)
                st.session_state.df = df
                st.session_state.df_original = df.copy()
                st.session_state.metadata = metadata
                st.session_state.cleaned = None
                st.session_state.profile_html = None
                st.success(f"Loaded {metadata['filename']} — {metadata['shape'][0]} rows x {metadata['shape'][1]} cols")
            except Exception as e:
                st.error(f"Failed to read file: {e}")
                st.stop()

    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("Preview")
        st.dataframe(df.head(5), use_container_width=True)

        st.subheader("Column Configuration")
        cols = list(df.columns)

        use_all = st.checkbox("Use entire dataset (no column config needed)", key="widget_use_all")

        if use_all:
            sel_text_cols = cols
            sel_label_cols = None
            sel_drop_cols = []
            st.info(f"All {len(cols)} columns will be used as input.")
        else:
            sel_text_cols = st.multiselect("Input columns (text)", cols, key="widget_text_cols")
            sel_label_cols = st.multiselect("Output/label columns (optional)", cols, key="widget_label_cols") or None
            sel_drop_cols = st.multiselect("Columns to drop", cols, key="widget_drop_cols")

        if st.button("Apply & Proceed"):
            if not use_all and not sel_text_cols:
                st.error("Select at least one input column.")
                st.stop()
            df = df.drop(columns=sel_drop_cols, errors="ignore")
            st.session_state.df = df
            st.session_state.text_cols = sel_text_cols
            st.session_state.label_cols = sel_label_cols
            st.success("Configuration saved. Go to Profile & Clean.")

# ─────────────────────────────────────────────
# SCREEN 2: PROFILE & CLEAN
# ─────────────────────────────────────────────
elif screen == "2. Profile & Clean":
    st.header("Profile & Clean")

    if st.session_state.df is None:
        st.warning("Upload a file first on the Upload screen.")
        st.stop()

    df = st.session_state.df

    # Profiling
    if st.button("Generate Data Profile"):
        with st.spinner("Generating profile..."):
            try:
                st.session_state.profile_html = generate_profile(df)
            except Exception as e:
                st.error(f"Profiling failed: {e}")

    if st.session_state.profile_html:
        with st.expander("Data Profile Report", expanded=False):
            st.components.v1.html(st.session_state.profile_html, height=600, scrolling=True)

    st.divider()

    # Cleaning
    st.subheader("Cleaning Options")

    with st.expander("Missing & Duplicates", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            do_drop_null = st.checkbox("Drop columns >50% null", value=True)
            do_dedup = st.checkbox("Deduplicate rows", value=True)
            do_remove_empty = st.checkbox("Remove fully empty rows", value=False)
        with c2:
            do_impute = st.checkbox("Impute missing values", value=True)
            do_replace_nulls = st.checkbox("Replace placeholder nulls (N/A, null, -)", value=False)

    with st.expander("Text Standardization", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            do_strip = st.checkbox("Strip whitespace from text", value=True)
            do_lowercase = st.checkbox("Lowercase all text", value=False)
        with c2:
            do_outliers = st.checkbox("Remove numeric outliers (z-score > 3)", value=False)

    with st.expander("Column Cleanup", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            do_snake = st.checkbox("Rename columns to snake_case", value=False)
            do_parse_dates = st.checkbox("Parse date columns", value=True)

    if st.button("Run Cleaning"):
        cleaned = df.copy()
        changes = []

        if do_drop_null:
            cleaned, ch = drop_high_null(cleaned)
            changes.append(ch)

        if do_dedup:
            cleaned, ch = deduplicate(cleaned)
            changes.append(ch)

        if do_remove_empty:
            cleaned, ch = remove_empty_rows(cleaned)
            changes.append(ch)

        if do_replace_nulls:
            cleaned, ch = replace_placeholder_nulls(cleaned)
            changes.append(ch)

        if do_impute:
            cleaned, ch = impute_missing(cleaned)
            changes.append(ch)

        if do_strip:
            cleaned, ch = strip_whitespace(cleaned)
            changes.append(ch)

        if do_lowercase:
            cleaned, ch = lowercase_text(cleaned)
            changes.append(ch)

        if do_parse_dates:
            cleaned, ch = parse_dates(cleaned)
            changes.append(ch)

        if do_outliers:
            cleaned, ch = remove_outliers(cleaned)
            changes.append(ch)

        if do_snake:
            cleaned, ch = rename_to_snake_case(cleaned)
            changes.append(ch)

        st.session_state.cleaned = cleaned

        st.success("Cleaning complete!")
        for ch in changes:
            st.info(ch["reason"])

    if st.session_state.cleaned is not None:
        st.divider()
        st.subheader("Before vs After")
        show_before_after(st.session_state.df_original, st.session_state.cleaned)

# ─────────────────────────────────────────────
# SCREEN 3: FORMAT & EXPORT
# ─────────────────────────────────────────────
elif screen == "3. Format & Export":
    st.header("Format & Export")

    if st.session_state.cleaned is None:
        st.warning("Clean your data first on the Profile & Clean screen.")
        st.stop()

    cleaned = st.session_state.cleaned
    text_cols = st.session_state.get("text_cols")
    label_cols = st.session_state.get("label_cols")

    if not text_cols:
        cols = list(cleaned.columns)
        text_cols = st.multiselect("Input columns", cols, default=cols[:1], key="fmt_text_cols")
        label_cols = st.multiselect("Output/label columns (optional)", cols, key="fmt_label_cols") or None
    else:
        label_info = f"{len(label_cols)} label column(s)" if label_cols else "no label columns"
        st.success(f"Using {len(text_cols)} input column(s) | {label_info}")

    st.subheader("Output Format")
    fmt = st.radio("Format", ["Instruction JSONL", "Chat JSONL", "Both"], horizontal=True)

    if st.button("Generate"):
        instruction_str = None
        chat_str = None

        if fmt in ("Instruction JSONL", "Both"):
            instruction_str = to_instruction_jsonl(cleaned, text_cols, label_cols)
        if fmt in ("Chat JSONL", "Both"):
            chat_str = to_chat_jsonl(cleaned, text_cols, label_cols)

        st.session_state.instruction_str = instruction_str
        st.session_state.chat_str = chat_str
        st.success("Formatted!")

    # Preview
    if "instruction_str" in st.session_state and st.session_state.instruction_str:
        st.subheader("Instruction JSONL Preview")
        show_format_preview(st.session_state.instruction_str)

    if "chat_str" in st.session_state and st.session_state.chat_str:
        st.subheader("Chat JSONL Preview")
        show_format_preview(st.session_state.chat_str)

    # Downloads (in-memory, no server disk writes)
    st.divider()
    st.subheader("Downloads")

    csv_bytes = cleaned.to_csv(index=False).encode()
    st.download_button("Download Clean CSV", csv_bytes, file_name="clean_data.csv")

    if "instruction_str" in st.session_state and st.session_state.instruction_str:
        st.download_button("Download Instruction JSONL", st.session_state.instruction_str.encode(), file_name="instruction_data.jsonl")

    if "chat_str" in st.session_state and st.session_state.chat_str:
        st.download_button("Download Chat JSONL", st.session_state.chat_str.encode(), file_name="chat_data.jsonl")

    # Push to HF Hub (in-memory)
    st.divider()
    st.subheader("Push to Hugging Face Hub")
    hf_token = st.text_input("Hugging Face Token", type="password", key="hf_token_export", help="Get one at https://huggingface.co/settings/tokens")
    hf_repo = st.text_input("Repo name (e.g., username/dataset-name)", key="hf_repo_export")

    if st.button("Push Dataset to Hub") and hf_token and hf_repo:
        import io
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            api.create_repo(hf_repo, repo_type="dataset", exist_ok=True, token=hf_token)
            data_to_push = st.session_state.get("instruction_str") or st.session_state.get("chat_str") or ""
            api.upload_file(
                path_or_fileobj=io.BytesIO(data_to_push.encode()),
                path_in_repo="data.jsonl",
                repo_id=hf_repo,
                repo_type="dataset",
                token=hf_token,
            )
            st.success(f"Pushed to https://huggingface.co/datasets/{hf_repo}")
        except Exception as e:
            st.error(f"Push failed: {e}")

# ─────────────────────────────────────────────
# SCREEN 4: FINE-TUNE
# ─────────────────────────────────────────────
elif screen == "4. Fine-Tune":
    st.header("Fine-Tune (Optional)")

    if st.session_state.cleaned is None:
        st.warning("Clean your data first on the Profile & Clean screen.")
        st.stop()

    cleaned = st.session_state.cleaned
    text_cols = st.session_state.get("text_cols", [])
    label_cols = st.session_state.get("label_cols", [])

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in .env file. Add it and restart.")
        st.stop()

    # ── Step 1: Describe → Generate Code ──
    st.subheader("Step 1: Describe What You Want")
    st.caption("Describe your ML goal. DataForge generates a real Python model for your data.")

    model_desc = st.text_area(
        "What should your model do?",
        value=st.session_state.get("model_desc_input", ""),
        placeholder="e.g. Predict house prices from features\n"
                    "or: Classify customer feedback as positive/negative\n"
                    "or: Cluster customers into segments",
        height=120,
    )
    st.session_state.model_desc_input = model_desc

    df_info = {
        "columns": list(cleaned.columns),
        "dtypes": {c: str(cleaned[c].dtype) for c in cleaned.columns},
        "sample": {c: cleaned[c].dropna().head(5).tolist() for c in cleaned.columns[:5]},
    }

    if st.button("🤖 Suggest Model (AI Advice)") and model_desc:
        with st.spinner("Analyzing your data and goal..."):
            try:
                advice = suggest_model(model_desc, df_info, GROQ_API_KEY)
                st.session_state.model_desc_input = advice
                st.session_state.model_advice = advice
                st.rerun()
            except Exception as e:
                st.error(f"Suggestion failed: {e}")

    if st.session_state.get("model_advice"):
        with st.container(border=True):
            st.markdown("**AI Recommendation — now in the text box above. Review and edit it, then click Generate Model Code.**")
            st.caption(st.session_state.model_advice)

    if st.button("⚡ Generate Model Code", type="primary") and model_desc:
        with st.spinner("Generating custom ML model code..."):
            try:
                result = generate_model(model_desc, df_info, GROQ_API_KEY)
                st.session_state.generated_model = result
                if result["compiles"]:
                    st.success("Model code generated and verified!")
                else:
                    st.error("Generated code has syntax errors. Try a different description.")
            except Exception as e:
                st.error(f"Generation failed: {e}")

    # ── Step 2: Review + Train ──
    if st.session_state.get("generated_model"):
        result = st.session_state.generated_model

        st.subheader("Step 2: Review Generated Model")
        st.caption("This is a real Python ML model generated for your task.")

        with st.expander("View Generated Model Code", expanded=True):
            st.code(result.get("code", "# code not available"), language="python")

        st.divider()
        st.subheader("Step 3: Train This Model on Your Data")

        if st.button("Train Generated Model", type="primary"):
            cols = list(cleaned.columns)
            train_label_cols = label_cols if label_cols else None
            if train_label_cols:
                train_text_cols = text_cols if text_cols else [c for c in cols if c not in train_label_cols]
                train_df = cleaned[train_text_cols + train_label_cols].copy()
            else:
                train_text_cols = text_cols if text_cols else cols
                train_df = cleaned[train_text_cols].copy()

            st.session_state.train_log = [f"Training on {len(cleaned)} rows", f"Features: {train_text_cols}", f"Labels: {train_label_cols}"]

            try:
                import importlib.util, sys, tempfile
                # Write generated code to temp file for importlib
                tmp_dir = tempfile.mkdtemp(prefix="dataforge_")
                tmp_code_path = os.path.join(tmp_dir, "generated_model.py")
                with open(tmp_code_path, "w") as f:
                    f.write(result["code"])

                spec = importlib.util.spec_from_file_location("gen_model", tmp_code_path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules["gen_model"] = mod
                spec.loader.exec_module(mod)

                model_instance = mod.GeneratedModel()
                st.session_state.train_log.append(f"Training on {len(train_df)} rows...")
                model_instance.train(train_df, train_text_cols, train_label_cols)
                st.session_state.train_log.append("Training complete")

                metrics = model_instance.evaluate(train_df, train_text_cols, train_label_cols)
                st.session_state.train_log.append(f"Metrics: {json.dumps(metrics, default=str)}")

                st.session_state.gen_metrics = metrics
                st.session_state.gen_model_instance = model_instance
                st.session_state.gen_model_module = mod
                st.session_state.gen_model_code = result["code"]
                st.session_state.train_text_cols = train_text_cols
                st.session_state.train_label_cols = train_label_cols
                st.session_state.training = False
                st.success("Training complete!")
                st.rerun()
            except Exception as e:
                st.session_state.train_log.append(f"ERROR: {e}")
                st.error(f"Training failed: {e}")

        # Show training log from session state
        if st.session_state.get("train_log"):
            with st.expander("Training Log", expanded=True):
                st.code("\n".join(st.session_state.train_log))

        # Show trained model results
        if st.session_state.get("gen_metrics"):
            st.success("Your generated model has been trained!")
            st.subheader("Model Metrics")
            metrics = st.session_state.gen_metrics
            c1, c2, c3 = st.columns(3)
            i = 0
            for k, v in metrics.items():
                if isinstance(v, float):
                    with [c1, c2, c3][i % 3]:
                        st.metric(k.replace("_", " ").title(), f"{v:.4f}")
                    i += 1
            for k, v in metrics.items():
                if isinstance(v, dict) and k == "classification_report":
                    with st.expander("Classification Report", expanded=False):
                        st.json(v)
                elif not isinstance(v, float):
                    with st.expander(k.replace("_", " ").title(), expanded=False):
                        st.write(v)

            # Model download (in-memory)
            if st.session_state.get("gen_model_instance"):
                import io, joblib
                buf = io.BytesIO()
                joblib.dump(st.session_state.gen_model_instance, buf)
                buf.seek(0)
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("Download Trained Model (.joblib)", buf, file_name="trained_model.joblib", use_container_width=True)
                with col2:
                    if st.button("Delete Model", type="secondary", use_container_width=True):
                        clear_model_session_state()
                        st.rerun()

        st.divider()
        st.subheader("Alternative: Fine-Tune an LLM Instead")
        st.caption("If you want a text-generation model instead, use the options below. Requires formatted JSONL from Screen 3.")

        has_formatted = st.session_state.get("instruction_str") or st.session_state.get("chat_str")
        if has_formatted:
            approach = st.radio(
                "LLM Fine-Tuning Mode",
                ["Local (TinyLlama)", "Cloud (Together AI)", "Colab Notebook"],
                horizontal=True,
            )

            if approach == "Local (TinyLlama)":
                if not HAS_FINETUNE_DEPS:
                    st.warning("Install: pip install -r requirements-finetune.txt")
                    st.stop()
                model_choice = st.selectbox("Model", list(AVAILABLE_MODELS.keys()))
                num_epochs = st.slider("Epochs", 1, 3, 1)
                jsonl_data = st.session_state.get("instruction_str") or st.session_state.get("chat_str")
                tmp_dir = Path(tempfile.mkdtemp(prefix="dataforge_"))
                train_path = tmp_dir / "train_data.jsonl"
                train_path.write_text(jsonl_data)
                log_path = tmp_dir / "train_log.txt"
                done_flag = tmp_dir / "train_done.flag"

                if st.button("Start Fine-Tuning"):
                    log_path.write_text("[0/4] Starting...\n")
                    train_async(
                        jsonl_path=str(train_path),
                        model_name=model_choice,
                        num_epochs=num_epochs,
                        output_dir=str(tmp_dir / "model"),
                        log_file=str(log_path),
                    )
                    st.rerun()

                if log_path.exists():
                    log_text = log_path.read_text()
                    is_done = "__DONE__" in log_text
                    if is_done:
                        done_flag.touch()
                        st.success("Training complete!")
                    else:
                        steps = sum(1 for l in log_text.split("\n") if "[0/4]" in l or "[1/4]" in l or "[2/4]" in l or "[3/4]" in l or "[4/4]" in l)
                        st.progress(min(steps / 4, 0.99), text=f"Step {steps}/4")
                        st.info("Training...")
                        if st.button("🔄 Refresh"):
                            st.rerun()
                    with st.expander("Logs"):
                        st.code(log_text)

            elif approach == "Cloud (Together AI)":
                if not TOGETHER_API_KEY:
                    st.error("TOGETHER_API_KEY not found in .env")
                    st.stop()
                cloud_model = st.selectbox("Base model", CLOUD_MODELS if CLOUD_MODELS else ["meta-llama/Meta-Llama-3.1-8B-Instruct"])
                cloud_epochs = st.slider("Epochs", 1, 3, 1)
                jsonl_data = st.session_state.get("instruction_str") or st.session_state.get("chat_str")
                tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
                tmp_file.write(jsonl_data)
                tmp_file.close()

                cloud_ph = st.empty()
                logs = []
                def clog(m):
                    logs.append(m)
                    cloud_ph.code("\n".join(logs[-20:]))

                if st.button("Start Cloud Fine-Tune"):
                    try:
                        clog("Uploading...")
                        fid = upload_file(tmp_file.name, TOGETHER_API_KEY)
                        clog(f"Uploaded: {fid}")
                        jid = create_fine_tune(fid, cloud_model, TOGETHER_API_KEY, epochs=cloud_epochs)
                        clog(f"Job: {jid}")
                        result = poll_until_done(jid, TOGETHER_API_KEY, log_callback=clog, poll_interval=15)
                        clog(f"Done! Model: {result.get('output', {}).get('model', 'available')}")
                    except Exception as e:
                        clog(f"Error: {e}")
                os.unlink(tmp_file.name)

            else:
                hf_dataset = st.text_input("HF dataset repo", placeholder="username/dataset")
                colab_model = st.selectbox("Model", list(AVAILABLE_MODELS.keys()), key="colab_model")
                if st.button("Generate Colab Notebook") and hf_dataset:
                    nb = generate_colab_notebook(hf_dataset, AVAILABLE_MODELS[colab_model])
                    st.download_button("Download Colab Notebook", nb.encode(), file_name="finetune_colab.ipynb")
        else:
            st.info("Format your data on Screen 3 first to access LLM fine-tuning options.")

# ─────────────────────────────────────────────
# SCREEN 5: MODEL PLAYGROUND
# ─────────────────────────────────────────────
elif screen == "5. Model Playground":
    st.header("Model Playground")

    if st.session_state.cleaned is None:
        st.warning("Clean your data first on the Profile & Clean screen.")
        st.stop()

    model_instance = st.session_state.get("gen_model_instance")
    model_module = st.session_state.get("gen_model_module")
    model_code = st.session_state.get("gen_model_code")

    if not model_instance:
        st.info("No trained model found. Go to **Fine-Tune** screen to generate and train a model first.")
        st.stop()

    cleaned = st.session_state.cleaned
    text_cols = st.session_state.get("train_text_cols") or st.session_state.get("text_cols", [])
    label_cols = st.session_state.get("train_label_cols") or st.session_state.get("label_cols", [])

    show_playground(cleaned, model_instance, model_module, model_code, text_cols, label_cols, GROQ_API_KEY, st.session_state.get("gen_metrics", {}))
