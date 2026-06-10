import os
import re
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

MODEL_SESSION_KEYS = ["gen_model_instance", "gen_model_module", "gen_model_code", "gen_metrics", "gen_dir", "gen_model_ensemble"]
PLAYGROUND_CACHE_KEYS = ["_playground_col_stats", "_playground_data_context", "_playground_model_context"]

def clear_model_session_state():
    for k in MODEL_SESSION_KEYS + PLAYGROUND_CACHE_KEYS:
        if k in st.session_state:
            del st.session_state[k]

_UNIVERSAL_TEMPLATE = '''import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score, classification_report

class GeneratedModel:
    def __init__(self):
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        self.preprocessor = ColumnTransformer([
            ("num", num_pipe, make_column_selector(dtype_include=np.number)),
            ("cat", cat_pipe, make_column_selector(dtype_include=["object","category"])),
        ], remainder="drop")
        self.model = None
        self.feature_names_ = None
        self._data_stats = None

    def _compute_stats(self, df, cols):
        stats = {}
        for c in cols:
            if c not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                stats[c] = {"dtype": "numeric", "median": float(df[c].median()) if not df[c].isna().all() else 0.0}
            else:
                mode_vals = df[c].dropna().mode()
                stats[c] = {"dtype": "categorical", "mode": str(mode_vals.iloc[0]) if len(mode_vals) > 0 else ""}
        return stats

    def _auto_fill(self, X):
        if self._data_stats is None:
            return X
        for c, stat in self._data_stats.items():
            if c not in X.columns:
                X[c] = stat.get("median") if stat["dtype"] == "numeric" else stat.get("mode", "")
            else:
                idx = X[c].isna()
                if idx.any():
                    fill_val = stat.get("median") if stat["dtype"] == "numeric" else stat.get("mode", "")
                    X.loc[idx, c] = fill_val
        return X

    def _prepare_y(self, df, label_cols):
        if not label_cols:
            return None
        y = df[label_cols[0]]
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        y = y.ffill().bfill().fillna(0)
        if y.dtype.kind == "b":
            y = y.astype(int)
        return y

    def train(self, df, text_cols, label_cols):
        feat_cols = [c for c in text_cols if not label_cols or c not in label_cols] or text_cols[:5]
        X = df[feat_cols].copy()
        y = self._prepare_y(df, label_cols)
        self.feature_names_ = X.columns.tolist()
        self._data_stats = self._compute_stats(df, self.feature_names_)
        Xp = self.preprocessor.fit_transform(X)
        if y is not None:
            if y.dtype.kind in ("O",) or y.nunique() < 20:
                self.model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
            else:
                self.model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
            self.model.fit(Xp, y)
        else:
            n = min(3, len(Xp))
            self.model = KMeans(n_clusters=n, random_state=42, n_init="auto")
            self.model.fit(Xp)

    def predict(self, X):
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        X = self._auto_fill(X)
        return self.model.predict(self.preprocessor.transform(X[self.feature_names_]))

    def evaluate(self, df, text_cols, label_cols):
        feat_cols = [c for c in text_cols if not label_cols or c not in label_cols] or text_cols[:5]
        X = df[feat_cols].copy()
        y = self._prepare_y(df, label_cols)
        if y is None:
            Xp = self.preprocessor.transform(X)
            preds = self.model.predict(Xp)
            return {"note": "Clustering", "clusters": int(self.model.n_clusters) if hasattr(self.model, "n_clusters") else 0}
        Xp = self.preprocessor.transform(X)
        preds = self.model.predict(Xp)
        if isinstance(self.model, RandomForestClassifier):
            return {"accuracy": round(accuracy_score(y, preds), 4), "classification_report": classification_report(y, preds, output_dict=True, zero_division=0)}
        return {"r2_score": round(r2_score(y, preds), 4), "predictions_sample": preds[:5].tolist()}
'''


def _assess_training_readiness(df, text_cols, label_cols):
    """Analyze data compatibility and recommend the right model approach."""
    import numpy as np
    result = {"ready": True, "warnings": [], "recommendation": None, "task_type": "clustering"}

    if not text_cols:
        result["ready"] = False
        result["warnings"].append("No feature columns selected.")
        return result

    n_rows = len(df)
    if n_rows == 0:
        result["ready"] = False
        result["warnings"].append("Dataset is empty.")
        return result

    features = [c for c in text_cols if c in df.columns]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if not pd.api.types.is_numeric_dtype(df[c]) and c in df.columns]

    result["row_count"] = n_rows
    result["feature_count"] = len(features)
    result["numeric_features"] = numeric[:10]
    result["categorical_features"] = categorical[:10]

    if len(features) == 0:
        result["ready"] = False
        result["warnings"].append("No valid feature columns found.")
        return result

    if not numeric:
        result["warnings"].append("No numeric features detected. Model will use a synthetic fallback feature.")

    if label_cols:
        label = [c for c in label_cols if c in df.columns]
        if not label:
            result["ready"] = False
            result["warnings"].append("Label column not found in dataframe.")
            return result

        y = df[label[0]].dropna()
        nunique = y.nunique()
        total = len(y)
        result["label_cardinality"] = int(nunique)
        result["label_samples"] = int(total)
        result["label_distribution"] = y.value_counts().head(10).to_dict()

        if total < 10:
            result["warnings"].append(f"Only {total} labeled samples — results may be unreliable.")

        if y.dtype.kind in ("O", "b") or nunique < 20:
            result["task_type"] = "classification"
            if nunique == 2:
                result["recommendation"] = "binary_classification"
            elif nunique <= 10:
                result["recommendation"] = f"multiclass_classification_{nunique}_classes"
            else:
                result["warnings"].append(f"High cardinality label ({nunique} classes). Consider grouping rare classes.")

            if nunique > total * 0.5:
                result["warnings"].append(f"Each class has very few samples ({total}/{nunique} ≈ {total//max(nunique,1):.0f}/class).")

            top_class_pct = (y.value_counts().iloc[0] / total * 100) if total > 0 else 0
            if top_class_pct > 80:
                result["warnings"].append(f"Class imbalance: '{y.value_counts().index[0]}' dominates ({top_class_pct:.0f}%).")
        else:
            result["task_type"] = "regression"
            result["recommendation"] = "regression"
            if nunique < 30 and total > 100:
                result["warnings"].append(f"Only {nunique} unique label values — consider treating as classification.")
    else:
        result["task_type"] = "clustering"
        result["recommendation"] = "clustering"
        if n_rows < 5:
            result["warnings"].append("Too few rows for meaningful clustering.")

    return result


def _model_code_for_assessment(assessment=None):
    return _UNIVERSAL_TEMPLATE

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

try:
    from finetune.model_config import generate_config as generate_model_config
    from finetune.model_generator import generate_model, train_generated_model, suggest_model
    HAS_MODEL_GEN_DEPS = True
except ImportError:
    generate_model_config = generate_model = train_generated_model = suggest_model = None
    HAS_MODEL_GEN_DEPS = False

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
        try:
            st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Preview unavailable: {e}")

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
            try:
                if not use_all and not sel_text_cols:
                    st.error("Select at least one input column.")
                    st.stop()
                df = df.drop(columns=sel_drop_cols, errors="ignore")
                st.session_state.df = df
                st.session_state.text_cols = sel_text_cols
                st.session_state.label_cols = sel_label_cols
                st.success("Configuration saved. Go to Profile & Clean.")
            except Exception as e:
                st.error(f"Apply failed: {e}")
                st.stop()

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
            try:
                st.components.v1.html(st.session_state.profile_html, height=600, scrolling=True)
            except Exception as e:
                st.caption(f"Profile preview unavailable: {e}")

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

    def _reconcile_column_names(stored, original_cols, cleaned_df):
        """Map stored column names to their equivalents in cleaned_df (handles renames & drops)."""
        if not stored:
            return stored
        cleaned_cols = set(cleaned_df.columns)
        result = []
        for col in stored:
            if col in cleaned_cols:
                result.append(col)
            elif col in original_cols:
                snake = col.strip().lower()
                snake = re.sub(r"[^a-z0-9_]", "_", snake)
                snake = re.sub(r"_+", "_", snake).strip("_")
                if snake in cleaned_cols:
                    result.append(snake)
        return result if result else None

    if st.button("Run Cleaning"):
        try:
            cleaned = df.copy()
            changes = []
            original_cols = set(df.columns)

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

            # Reconcile stored column names with cleaned dataframe
            old_text = st.session_state.get("text_cols")
            old_label = st.session_state.get("label_cols")
            st.session_state.text_cols = _reconcile_column_names(old_text, original_cols, cleaned)
            st.session_state.label_cols = _reconcile_column_names(old_label, original_cols, cleaned)
            st.session_state.cleaned = cleaned
            st.success("Cleaning complete!")
            for ch in changes:
                try:
                    st.info(ch["reason"])
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Cleaning failed: {e}")
            st.stop()

    if st.session_state.cleaned is not None:
        st.divider()
        st.subheader("Before vs After")
        try:
            show_before_after(st.session_state.df_original, st.session_state.cleaned)
        except Exception as e:
            st.caption(f"Comparison view unavailable: {e}")

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

    # Validate stored columns against cleaned dataframe (handles renames/drops)
    valid_cols = set(cleaned.columns)
    if text_cols and not all(c in valid_cols for c in text_cols):
        missing = [c for c in text_cols if c not in valid_cols]
        st.warning(f"Stored columns not found after cleaning: {missing}. Please re-select.")
        text_cols = None
    if label_cols and not all(c in valid_cols for c in label_cols):
        label_cols = None

    if not text_cols:
        cols = list(cleaned.columns)
        text_cols = st.multiselect("Input columns", cols, default=cols[:1], key="fmt_text_cols")
        label_cols = st.multiselect("Output/label columns (select just 1 — the column to predict)", cols, default=[cols[-1]] if cols else [], key="fmt_label_cols") or None
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

    try:
        csv_bytes = cleaned.to_csv(index=False).encode()
        st.download_button("Download Clean CSV", csv_bytes, file_name="clean_data.csv")
    except Exception as e:
        st.error(f"CSV export failed: {e}")

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

    # Validate stored columns against cleaned dataframe
    valid_columns = list(cleaned.columns)
    valid_set = set(valid_columns)
    if text_cols and not all(c in valid_set for c in text_cols):
        missing = [c for c in text_cols if c not in valid_set]
        st.warning(f"Stored input columns not found after cleaning: {missing}. Please re-select below.")
        text_cols = None
    if label_cols and not all(c in valid_set for c in label_cols):
        label_cols = None

    # Column pickers (shown when stored columns are missing or invalid)
    with st.expander("Column Configuration for Training", expanded=not text_cols):
        c1, c2 = st.columns(2)
        with c1:
            ft_text = st.multiselect(
                "Input/feature columns",
                valid_columns,
                default=text_cols if text_cols else valid_columns[:min(3, len(valid_columns))],
                key="ft_text_cols",
            )
        with c2:
            label_default = label_cols if label_cols else [valid_columns[-1]] if valid_columns else []
            ft_label = st.multiselect("Output/label columns (select just 1 — the column to predict)", valid_columns, default=label_default, key="ft_label_cols") or None
        if ft_text:
            text_cols = ft_text
            label_cols = ft_label

    has_groq = bool(GROQ_API_KEY)
    if not has_groq:
        with st.warning("GROQ_API_KEY not set — AI generation disabled. Enter model code manually below."):
            st.caption("Set `GROQ_API_KEY` in `.env` to enable AI suggestion & code generation.")

    # ── Data Compatibility Assessment ──
    assessment = None
    if text_cols:
        assessment = _assess_training_readiness(cleaned, text_cols, label_cols)
        if assessment:
            ready = assessment.get("ready", True)
            task_type = assessment.get("task_type", "unknown").title()
            n_feat = assessment.get("feature_count", 0)
            n_rows = assessment.get("row_count", 0)
            warnings = assessment.get("warnings", [])

            if ready:
                st.success(f"Data ready for **{task_type}** ({n_feat} features, {n_rows} rows)")
            else:
                st.error(f"Cannot train: {'; '.join(warnings)}")

            if warnings:
                with st.expander("Data Compatibility Warnings", expanded=False):
                    for w in warnings:
                        st.caption(f"⚠ {w}")

            if assessment.get("label_distribution"):
                with st.expander("Label Distribution (top 10)", expanded=False):
                    dist = assessment["label_distribution"]
                    st.json({str(k): int(v) for k, v in dist.items()})

    # ── Step 1: Describe → Generate Code ──
    st.subheader("Step 1: Describe or Paste Your Model Code")

    if has_groq and HAS_MODEL_GEN_DEPS:
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
            "assessment": assessment,
        }

        if st.button("🤖 Suggest Model (AI Advice)", disabled=not model_desc) and model_desc:
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

        gen_btn = st.button("⚡ Generate Model Code", type="primary", disabled=not model_desc)
    else:
        if has_groq and not HAS_MODEL_GEN_DEPS:
            with st.warning("GROQ_API_KEY is set but model generation dependencies are missing."):
                st.caption("Enter model code manually below.")
        else:
            st.caption("Paste your sklearn model code below, or use the quick template.")
        model_desc = ""
        gen_btn = False

    # Manual code editor (shown when no API key, missing deps, or after generation)
    if not has_groq or (has_groq and not HAS_MODEL_GEN_DEPS):
        task_hint = assessment.get("task_type", "model").title() if assessment else "Model"
        DEFAULT_TEMPLATE = f"""# Paste your sklearn model code here, or click one of the buttons below.
# It must define a class with train() and evaluate() methods.
#
# class GeneratedModel:
#     def train(self, df, text_cols, label_cols):
#         ...
#     def evaluate(self, df, text_cols, label_cols):
#         ...
"""
        manual_code = st.text_area("Model Code (Python)", value=st.session_state.get("manual_code", DEFAULT_TEMPLATE), height=300)
        st.session_state.manual_code = manual_code

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            apply_btn = st.button("Use This Code", type="primary")
        with col_b:
            if st.button("Use Universal Template"):
                st.session_state.manual_code = _UNIVERSAL_TEMPLATE
                st.rerun()
        with col_c:
            if st.button("Use Universal Template"):
                st.session_state.manual_code = _UNIVERSAL_TEMPLATE
                st.rerun()

        if apply_btn and manual_code.strip():
            try:
                compile(manual_code, "<manual>", "exec")
                result = {"model_path": None, "code": manual_code, "compiles": True}
                st.session_state.generated_model = result
                st.success("Code verified and applied!")
            except SyntaxError as e:
                st.error(f"Syntax error: {e}")

    if gen_btn and model_desc:
        with st.spinner("Generating custom ML model code..."):
            try:
                result = generate_model(model_desc, df_info, GROQ_API_KEY)
                if not result.get("compiles"):
                    st.warning("AI-generated code has syntax errors. Using pre-built model for your data type instead.")
                    result = {"model_path": None, "code": _UNIVERSAL_TEMPLATE, "compiles": True}
            except Exception as e:
                st.warning(f"AI generation failed ({e}). Using pre-built model for your data.")
                result = {"model_path": None, "code": _UNIVERSAL_TEMPLATE, "compiles": True}
            st.session_state.generated_model = result
            if result["compiles"]:
                st.success("Model ready! Review and train below.")

    # ── Step 2: Review + Train ──
    if st.session_state.get("generated_model"):
        result = st.session_state.generated_model

        st.subheader("Step 2: Review Generated Model")
        st.caption("This is a real Python ML model generated for your task.")

        with st.expander("View Generated Model Code", expanded=True):
            st.code(result.get("code", "# code not available"), language="python")

        st.divider()
        st.subheader("Step 3: Train This Model on Your Data")

        # ── Retrain-on-full-data trigger (must check before the button to consume the flag) ──
        _retrain_requested = st.session_state.pop("_retrain_full", False)

        if st.button("Train Generated Model", type="primary") or _retrain_requested:
            # Deduplicate column names (e.g. after snake_case rename)
            if cleaned.columns.duplicated().any():
                cleaned = cleaned.loc[:, ~cleaned.columns.duplicated(keep="first")]
            cols = list(cleaned.columns)
            train_label_cols = [c for c in (label_cols or []) if c in cols] if label_cols else None
            train_text_cols = [c for c in (text_cols or []) if c in cols] if text_cols else None
            if not train_text_cols:
                train_text_cols = [c for c in cols if not train_label_cols or c not in train_label_cols][:5]
            if not train_text_cols:
                st.error("No feature columns remaining after excluding labels.")
                st.stop()
            # Build target list
            if train_label_cols and len(train_label_cols) >= len(cols) - 1:
                target_columns = train_label_cols[:]
            elif train_label_cols:
                target_columns = train_label_cols[:1]
            else:
                target_columns = []

            # ── Auto-sampling ──
            SAMPLE_THRESHOLD = 5000
            using_sample = len(cleaned) > SAMPLE_THRESHOLD and not _retrain_requested
            train_df = cleaned.sample(n=SAMPLE_THRESHOLD, random_state=42) if using_sample else cleaned
            feature_cols = [c for c in train_text_cols if c not in target_columns] or train_text_cols[:10]

            st.session_state.train_log = [f"Training on {len(train_df)} rows" + (" (5K sample)" if using_sample else "")]
            train_log = []

            def _log(msg):
                train_log.append(msg)
                st.session_state.train_log = train_log[:]

            def _compute_metrics(y_true, y_pred, model_name):
                from sklearn.metrics import accuracy_score, r2_score, classification_report, mean_squared_error
                import numpy as np
                name = model_name.lower()
                if "classifier" in name or "svc" in name:
                    return {"accuracy": round(accuracy_score(y_true, y_pred), 4), "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)}
                if "regressor" in name or "svr" in name or "regression" in name:
                    mse = mean_squared_error(y_true, y_pred)
                    return {"r2_score": round(r2_score(y_true, y_pred), 4), "rmse": round(float(np.sqrt(mse)), 2)}
                return {}

            def _ensure_n_jobs(model):
                try:
                    model.set_params(n_jobs=-1)
                except (ValueError, TypeError):
                    pass
                return model

            def _train_ensemble(code_template):
                _log("Compiling model code...")
                import sys, types
                _mod_name = "dataforge_gen_model"
                if _mod_name in sys.modules:
                    del sys.modules[_mod_name]
                _mod = types.ModuleType(_mod_name)
                exec(compile(code_template, _mod_name, "exec"), _mod.__dict__)
                sys.modules[_mod_name] = _mod
                ModelClass = _mod.GeneratedModel

                proto = ModelClass()
                X_all = train_df[feature_cols]
                _log("Fitting preprocessor...")
                proto.preprocessor.fit(X_all)

                ensemble = {}
                all_metrics = {}
                final_code = code_template

                for i, target in enumerate(target_columns):
                    _log(f"Training model {i+1}/{len(target_columns)}: {target}")
                    inst = ModelClass()
                    inst.preprocessor = proto.preprocessor
                    Xp = inst.preprocessor.transform(X_all)
                    y = train_df[target].ffill().bfill().fillna(0)
                    if y.dtype.kind == "b":
                        y = y.astype(int)
                    _ensure_n_jobs(inst.model)
                    inst.model.fit(Xp, y)
                    y_pred = inst.model.predict(Xp)
                    metrics = _compute_metrics(y, y_pred, type(inst.model).__name__)
                    all_metrics[target] = metrics
                    inst.feature_names_ = feature_cols
                    ensemble[target] = {"model": inst, "features": feature_cols, "metrics": metrics}
                    _log(f"  {target} done")

                return ensemble, all_metrics, final_code

            used_code = result["code"]
            try:
                ensemble, all_metrics, used_code = _train_ensemble(used_code)
            except Exception as e:
                _log(f"Generated model failed: {e}")
                _log("Falling back to universal template...")
                try:
                    ensemble, all_metrics, used_code = _train_ensemble(_UNIVERSAL_TEMPLATE)
                    st.warning("AI model had issues — universal template trained instead.")
                except Exception as e2:
                    _log(f"Fallback also failed: {e2}")
                    st.error(f"Training failed: {e2}")
                    st.stop()

            for _ck in PLAYGROUND_CACHE_KEYS:
                st.session_state.pop(_ck, None)
            st.session_state.gen_model_ensemble = ensemble
            st.session_state.gen_model_instance = list(ensemble.values())[0]["model"]
            st.session_state.gen_model_module = None
            st.session_state.gen_model_code = used_code
            st.session_state.gen_metrics = all_metrics
            st.session_state.train_text_cols = feature_cols
            st.session_state.train_label_cols = target_columns
            st.session_state._sampled_training = using_sample
            st.session_state.training = False
            st.success(f"Trained {len(ensemble)} model{'s' if len(ensemble) != 1 else ''}! Use the Playground to ask about any column.")
            st.rerun()

        # ── Retrain-on-full-data button (shown after sampled training) ──
        if st.session_state.get("_sampled_training") and not st.session_state.get("_retrain_full"):
            st.info("Trained on a 5K-row sample for speed.")
            if st.button("Train on Full Dataset (more accurate)", type="primary"):
                st.session_state._retrain_full = True
                st.rerun()

        # Show training log from session state
        if st.session_state.get("train_log"):
            with st.expander("Training Log", expanded=True):
                try:
                    st.code("\n".join(str(x) for x in st.session_state.train_log))
                except Exception as e:
                    st.caption(f"Log display unavailable: {e}")

        # Show trained model results
        if st.session_state.get("gen_metrics"):
            st.success("Your model ensemble has been trained!")
            st.subheader("Model Metrics")
            try:
                all_metrics = st.session_state.gen_metrics
                ensemble = st.session_state.get("gen_model_ensemble", {})
                targets = list(all_metrics.keys()) if isinstance(all_metrics, dict) and not any(isinstance(v, float) for v in all_metrics.values()) else ["model"]
                for target in targets:
                    with st.expander(f"{target} Performance", expanded=(target == targets[0])):
                        m = all_metrics[target] if isinstance(all_metrics, dict) else all_metrics
                        c1, c2, c3 = st.columns(3)
                        i = 0
                        for k, v in m.items():
                            if isinstance(v, float):
                                with [c1, c2, c3][i % 3]:
                                    st.metric(k.replace("_", " ").title(), f"{v:.4f}")
                                i += 1
                        for k, v in m.items():
                            if isinstance(v, dict) and k == "classification_report":
                                st.json(v)
                            elif not isinstance(v, float):
                                st.write(f"{k}: {v}")
            except Exception as e:
                st.caption(f"Metrics display error: {e}")

            # Model download (in-memory)
            if st.session_state.get("gen_model_instance"):
                try:
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
                except Exception as e:
                    st.error(f"Model download failed: {e}")

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
                if not jsonl_data:
                    st.error("No formatted data found. Go to Screen 3 and generate JSONL first.")
                    st.stop()
                tmp_dir = Path(tempfile.mkdtemp(prefix="dataforge_"))
                train_path = tmp_dir / "train_data.jsonl"
                train_path.write_text(jsonl_data)
                log_path = tmp_dir / "train_log.txt"
                done_flag = tmp_dir / "train_done.flag"

                if st.button("Start Fine-Tuning"):
                    try:
                        log_path.write_text("[0/4] Starting...\n")
                        train_async(
                            jsonl_path=str(train_path),
                            model_name=model_choice,
                            num_epochs=num_epochs,
                            output_dir=str(tmp_dir / "model"),
                            log_file=str(log_path),
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Fine-tuning failed to start: {e}")

                if log_path.exists():
                    try:
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
                    except Exception as e:
                        st.caption(f"Log read error: {e}")

            elif approach == "Cloud (Together AI)":
                if not TOGETHER_API_KEY:
                    st.error("TOGETHER_API_KEY not found in .env")
                    st.stop()
                cloud_model = st.selectbox("Base model", CLOUD_MODELS if CLOUD_MODELS else ["meta-llama/Meta-Llama-3.1-8B-Instruct"])
                cloud_epochs = st.slider("Epochs", 1, 3, 1)
                jsonl_data = st.session_state.get("instruction_str") or st.session_state.get("chat_str")
                if not jsonl_data:
                    st.error("No formatted data found. Go to Screen 3 and generate JSONL first.")
                    st.stop()
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
                try:
                    os.unlink(tmp_file.name)
                except Exception:
                    pass

            else:
                hf_dataset = st.text_input("HF dataset repo", placeholder="username/dataset")
                colab_model = st.selectbox("Model", list(AVAILABLE_MODELS.keys()), key="colab_model")
                if st.button("Generate Colab Notebook") and hf_dataset:
                    try:
                        nb = generate_colab_notebook(hf_dataset, AVAILABLE_MODELS[colab_model])
                        st.download_button("Download Colab Notebook", nb.encode(), file_name="finetune_colab.ipynb")
                    except Exception as e:
                        st.error(f"Colab notebook generation failed: {e}")
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

    model_ensemble = st.session_state.get("gen_model_ensemble", {})
    model_instance = st.session_state.get("gen_model_instance")
    model_module = st.session_state.get("gen_model_module")
    model_code = st.session_state.get("gen_model_code")

    if not model_instance and not model_ensemble:
        st.info("No trained model found. Go to **Fine-Tune** screen to generate and train a model first.")
        st.stop()

    cleaned = st.session_state.cleaned
    text_cols = st.session_state.get("train_text_cols") or st.session_state.get("text_cols", [])
    label_cols = st.session_state.get("train_label_cols") or st.session_state.get("label_cols", [])

    try:
        show_playground(cleaned, model_instance or list(model_ensemble.values())[0]["model"] if model_ensemble else None, model_module, model_code, text_cols, label_cols, GROQ_API_KEY, st.session_state.get("gen_metrics", {}), model_ensemble)
    except Exception as e:
        st.error(f"Playground error: {e}")
