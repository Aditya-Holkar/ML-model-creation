import os
import json
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def get_model_info(gen_dir: str) -> dict:
    gen_dir = Path(gen_dir)
    model_path = gen_dir / "model.joblib"
    code_path = gen_dir / "generated_model.py"
    log_path = gen_dir / "train_log.txt"
    info = {"model_path": str(model_path), "code_path": str(code_path), "exists": False}
    if model_path.exists() and code_path.exists():
        info["exists"] = True
        info["log"] = log_path.read_text() if log_path.exists() else ""
    return info


def load_model_and_module(gen_dir: str):
    gen_dir = Path(gen_dir)
    model_path = gen_dir / "model.joblib"
    code_path = gen_dir / "generated_model.py"
    model = joblib.load(str(model_path))
    mod_name = f"gen_model_{gen_dir.name}"
    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
    else:
        spec = importlib.util.spec_from_file_location(mod_name, str(code_path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return model, mod


def detect_task_type(module, label_cols: list, model=None) -> str:
    if not label_cols:
        return "clustering"
    if model is not None:
        if hasattr(model, "predict_proba") or type(model).__name__ in ("RandomForestClassifier", "LogisticRegression", "SVC"):
            return "classification"
        if type(model).__name__ in ("RandomForestRegressor", "LinearRegression", "SVR"):
            return "regression"
        if type(model).__name__ in ("KMeans", "DBSCAN"):
            return "clustering"
    if module is not None:
        try:
            src = inspect.getsource(module.GeneratedModel.__init__)
            if "RandomForestClassifier" in src or "LogisticRegression" in src:
                return "classification"
            if "RandomForestRegressor" in src or "LinearRegression" in src:
                return "regression"
            if "KMeans" in src or "DBSCAN" in src:
                return "clustering"
        except Exception:
            pass
    return "classification" if label_cols else "clustering"


def get_feature_names(module, text_cols: list, label_cols: list) -> list:
    if label_cols:
        return [c for c in text_cols if c not in label_cols]
    return text_cols


def extract_feature_importance(model, feature_names: list) -> pd.DataFrame | None:
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        else:
            return None
        return pd.DataFrame({"feature": feature_names, "importance": np.abs(importances)}).sort_values("importance", ascending=False)
    except Exception:
        return None


def build_prediction_inputs(df: pd.DataFrame, feature_names: list, label_cols: list):
    inputs = {}
    st.subheader("Enter Feature Values")
    with st.container(border=True):
        cols_per_row = st.columns(3)
        for i, col in enumerate(feature_names):
            with cols_per_row[i % 3]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    cmin = float(df[col].min())
                    cmax = float(df[col].max())
                    cmid = float(df[col].median())
                    inputs[col] = st.slider(col, min_value=cmin, max_value=cmax, value=cmid, format="%.2f")
                else:
                    unique = df[col].dropna().unique().tolist()
                    inputs[col] = st.selectbox(col, unique)
    return inputs


def predict_from_inputs(model, inputs: dict, feature_names: list, module, label_cols: list):
    row = pd.DataFrame([inputs])
    try:
        pred = model.predict(row)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(row)
        except Exception:
            pass
    return pred, probs


def _build_data_context(df: pd.DataFrame, feature_names: list, label_cols: list, task_type: str) -> str:
    parts = []
    parts.append(f"Task: {task_type}")
    feature_names = [c for c in feature_names if c in df.columns]
    parts.append(f"Features ({len(feature_names)}): {', '.join(feature_names)}")
    if label_cols:
        label_cols = [c for c in label_cols if c in df.columns]
        parts.append(f"Target: {', '.join(label_cols)}")
    shape = df.shape
    parts.append(f"Rows: {shape[0]}")
    numeric_cols = df[feature_names].select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        stats = df[numeric_cols].describe().to_dict()
        parts.append(f"Numeric stats: {json.dumps(stats, default=str)}")
    cat_cols = [c for c in feature_names if not pd.api.types.is_numeric_dtype(df[c])]
    if cat_cols:
        val_counts = {}
        for c in cat_cols:
            try:
                val_counts[c] = df[c].value_counts().head(5).to_dict()
            except Exception:
                pass
        if val_counts:
            parts.append(f"Categorical values: {json.dumps(val_counts, default=str)}")
    cols_for_sample = feature_names + (label_cols or [])
    cols_for_sample = [c for c in cols_for_sample if c in df.columns]
    if cols_for_sample:
        sample = df[cols_for_sample].head(5).to_dict(orient="records")
        parts.append(f"Sample rows: {json.dumps(sample, default=str)}")
    return "\n".join(parts)


def _build_model_context(model, feature_names: list, has_labels: bool, task_type: str, gen_metrics: dict = None) -> str:
    parts = []
    parts.append(f"Algorithm: {type(model).__name__}")
    if hasattr(model, "get_params"):
        parts.append(f"Params: {json.dumps(model.get_params(), default=str)}")
    fi = extract_feature_importance(model, feature_names)
    if fi is not None:
        top5 = fi.head(5).to_dict(orient="records")
        parts.append(f"Top features: {json.dumps(top5, default=str)}")
    parts.append(f"Predict mode: {'predict_proba available' if has_labels and hasattr(model, 'predict_proba') else 'predict only'}")
    if gen_metrics:
        flat = {}
        is_ensemble = isinstance(gen_metrics, dict) and not any(isinstance(v, float) for v in gen_metrics.values())
        if is_ensemble:
            for t, m in gen_metrics.items():
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        flat[f"{t}_{k}"] = round(v, 4)
        else:
            for k, v in gen_metrics.items():
                if isinstance(v, (int, float)):
                    flat[k] = round(v, 4)
        if flat:
            parts.append(f"Metrics: {json.dumps(flat)}")
    return "\n".join(parts)


def _query_llm(system_prompt: str, user_msg: str, api_key: str, chat_history: list) -> str:
    import requests
    messages = [{"role": "system", "content": system_prompt}]
    for h in chat_history:
        messages.append(h)
    messages.append({"role": "user", "content": user_msg})
    resp = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0.4},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _is_prediction_question(question: str) -> bool:
    q = question.lower()
    keywords = ["predict", "forecast", "future", "projection", "will be", "would be",
                 "what if", "estimate", "expected", "next", "upcoming", "trend",
                 "outlook", "guess", "what would", "what will", "how much"]
    return any(kw in q for kw in keywords)


def _detect_target_column(question: str, all_columns: list, trained_label: str = None) -> str | None:
    q = question.lower()
    for col in all_columns:
        col_lower = col.lower()
        patterns = [
            rf"predict\s+{re.escape(col_lower)}",
            rf"forecast\s+{re.escape(col_lower)}",
            rf"what (?:is|will be|would be)\s+(?:the\s+)?{re.escape(col_lower)}",
            rf"{re.escape(col_lower)}\s+(?:of|for|when|if)",
            rf"estimate\s+{re.escape(col_lower)}",
        ]
        if any(re.search(p, q) for p in patterns):
            return col
    return trained_label


def _run_bulk_predictions(model, df: pd.DataFrame, feature_names: list, has_labels: bool, max_rows: int = 50) -> str:
    try:
        X = df[feature_names].head(max_rows)
        preds = model.predict(X)
        lines = [f"Predictions on {len(X)} rows (last {max_rows} of data):"]
        lines.append(f"  Min: {float(preds.min()):.4f}")
        lines.append(f"  Max: {float(preds.max()):.4f}")
        lines.append(f"  Mean: {float(preds.mean()):.4f}")
        lines.append(f"  Std: {float(preds.std()):.4f}")
        lines.append(f"  Recent predictions (last 5): {[float(p) for p in preds[-5:]]}")
        if has_labels and hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            lines.append(f"  Confidence range: {float(probs.max(axis=1).min()):.3f} - {float(probs.max(axis=1).max()):.3f}")
        return "\n".join(lines)
    except Exception as e:
        return f"Bulk prediction failed: {e}"


def _suggested_questions(df: pd.DataFrame, feature_names: list, label_cols: list, task_type: str, gen_metrics: dict = None, api_key: str = None, model_ensemble: dict = None) -> list:
    data_summary = _build_data_context(df, feature_names, label_cols, task_type)

    predict_targets = list(model_ensemble.keys()) if model_ensemble else ([label_cols[0]] if label_cols else ["target"])
    targets_str = ", ".join(predict_targets)
    if api_key:
        prompt = f"""You are a data analyst. Based on this dataset summary, generate exactly 3 concise prediction questions that can be answered from the data.

Dataset:
{data_summary}

The ML model can predict these columns: {targets_str}.
Ask about DIFFERENT prediction targets in each question.

Rules:
- Each question must reference actual column names from the dataset
- Each question must ask to predict a DIFFERENT column from the list: {targets_str}
- Focus on predictions, what-if scenarios
- Do NOT ask about clustering, correlations, or distributions
- Do NOT include numbering or prefixes
- Return one question per line, no extra text

Examples of good questions:
- What will the predicted price be for a house with 3 beds and 2000 sqft?
- Will the customer default if their income is below $50k?
- What happens to sales when the marketing spend doubles?"""
        try:
            import requests
            resp = requests.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3},
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            qs = [line.strip().strip('"').strip("'") for line in text.strip().split("\n") if line.strip() and not line.strip().startswith(("- ", "•", "1.", "2.", "3."))]
            if len(qs) >= 3:
                return qs[:6]
        except Exception:
            pass

    # Fallback: data-driven template questions
    questions = []
    num_feats = [c for c in feature_names if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    cat_feats = [c for c in feature_names if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]

    def _val(col, stat):
        try: return round(float(getattr(df[col], stat)()), 1) if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) else None
        except: return None

    if model_ensemble:
        for label in list(model_ensemble.keys())[:5]:
            feats = model_ensemble[label].get("features", [])
            num_f = [c for c in feats if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
            cat_f = [c for c in feats if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
            if num_f:
                f = num_f[0]
                lo, hi = _val(f, "min"), _val(f, "max")
                if lo is not None:
                    questions.append(f"Predict {label} when {f} = {(lo + hi) / 2:.1f}")
            if cat_f:
                vals = df[cat_f[0]].dropna().unique()[:3]
                if len(vals) > 0:
                    questions.append(f"What is the {label} for {cat_f[0]} = {str(vals[0])}?")
            if len(questions) >= 6:
                break
    else:
        label = label_cols[0] if label_cols else "target"
        if task_type == "classification":
            if num_feats:
                f = num_feats[0]
                lo, hi = _val(f, "min"), _val(f, "max")
                if lo is not None:
                    questions.append(f"Will {label} be positive if {f} is high (near {hi})?")
                    questions.append(f"Predict {label} when {f} = {(lo + hi) / 2:.1f}")
            if len(num_feats) >= 2:
                f1, f2 = num_feats[0], num_feats[1]
                lo1, hi1 = _val(f1, "min"), _val(f1, "max")
                lo2, hi2 = _val(f2, "min"), _val(f2, "max")
                if all(v is not None for v in [lo1, hi1, lo2, hi2]):
                    questions.append(f"What happens to {label} when {f1} = {hi1} and {f2} = {lo2}?")
            if cat_feats:
                vals = df[cat_feats[0]].dropna().unique()[:3]
                if len(vals) > 0:
                    questions.append(f"Predict {label} for {cat_feats[0]} = {str(vals[0])}")
            if gen_metrics and gen_metrics.get("accuracy"):
                questions.append(f"The model is {gen_metrics['accuracy']*100:.0f}% accurate — predict {label} for the next row")
        elif task_type == "regression":
            if num_feats:
                f = num_feats[0]
                lo, hi, med = _val(f, "min"), _val(f, "max"), _val(f, "median")
                if lo is not None and hi is not None and med is not None:
                    questions.append(f"Forecast {label} if {f} rises to {hi}")
                    questions.append(f"Predict {label} when {f} is at typical level ({med})")
            if len(num_feats) >= 2:
                f1, f2 = num_feats[0], num_feats[1]
                hi1, hi2 = _val(f1, "max"), _val(f2, "max")
                if hi1 is not None and hi2 is not None:
                    questions.append(f"What will {label} be if {f1} = {hi1} and {f2} = {hi2}?")
            if gen_metrics and gen_metrics.get("r2_score"):
                questions.append(f"Predict {label} for a new row and explain the result")
        else:
            if num_feats:
                f = num_feats[0]
                lo, hi = _val(f, "min"), _val(f, "max")
                if lo is not None and hi is not None:
                    questions.append(f"Predict the outcome when {f} is between {lo} and {hi}")
        questions.append(f"Make a prediction using the latest data and explain the result")

    seen = set()
    unique = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:6]


def _try_extract_prediction_inputs(question: str, feature_names: list, df: pd.DataFrame) -> dict | None:
    inputs = {}
    found_any = False
    q_lower = question.lower()
    for col in feature_names:
        col_lower = col.lower()
        col_variants = [col_lower, col_lower.replace("_", " "), col_lower.replace("_", ""), col_lower.replace(" ", "_")]
        col_variants = list(set(col_variants))
        escaped = [re.escape(v) for v in col_variants]
        is_numeric = col in df.columns and pd.api.types.is_numeric_dtype(df[col])

        if is_numeric:
            for e in escaped:
                patterns = [
                    rf"{e}\s*[=:]\s*([\d\.]+)",
                    rf"{e}\s+is\s+([\d\.]+)",
                    rf"{e}\s+(?:of|at)\s+([\d\.]+)",
                    rf"([\d\.]+)\s+{e}",
                ]
                for pat in patterns:
                    m = re.search(pat, q_lower)
                    if m:
                        val = float(m.group(1))
                        inputs[col] = val
                        found_any = True
                        break
                if col in inputs:
                    break
        if col in inputs:
            continue

        if not is_numeric and col in df.columns:
            cat_vals = df[col].dropna().unique().tolist()
            # Sort longest first to match multi-word values before substrings
            cat_vals_sorted = sorted(cat_vals, key=lambda x: -len(str(x)))
            for cv in cat_vals_sorted:
                cv_str = str(cv).lower()
                if len(cv_str) < 2:
                    continue
                for e in escaped:
                    # 1. Match "col is value", "col = value" etc.
                    for sep in ["is", "=", ":", "of", "as"]:
                        pat = rf"{e}\s*{sep}\s*{re.escape(cv_str)}"
                        if re.search(pat, q_lower):
                            inputs[col] = cv
                            found_any = True
                            break
                    if col in inputs:
                        break
                    # 2. Match "col value" (no separator) — finds column name followed
                    #    by category value within a short window (allowing prepositions)
                    for gap in ["", " ", " for ", " with ", " and ", " where ", " having "]:
                        pat = rf"{re.escape(e)}\s*{gap}\s*{re.escape(cv_str)}"
                        if re.search(pat, q_lower):
                            inputs[col] = cv
                            found_any = True
                            break
                    if col in inputs:
                        break
                    # 3. Match "value col" (value before column name)
                    pat = rf"{re.escape(cv_str)}\s+(?:of|in|for|at|as)\s+{re.escape(e)}"
                    if re.search(pat, q_lower):
                        inputs[col] = cv
                        found_any = True
                        break
                if col in inputs:
                    break
    return inputs if found_any else None








def show_chat_tab(api_key: str, df: pd.DataFrame, model, module, feature_names: list, label_cols: list, task_type: str, has_labels: bool, gen_metrics: dict = None, model_ensemble: dict = None):
    data_context = _build_data_context(df, feature_names, label_cols, task_type)
    model_context = _build_model_context(model, feature_names, has_labels, task_type, gen_metrics)

    ensemble_targets = list(model_ensemble.keys()) if model_ensemble else ([label_cols[0]] if label_cols else [])
    targets_str = ", ".join(ensemble_targets)
    system_prompt = f"""You are a data scientist assistant. You have access to a trained ML model ensemble and the full dataset.

DATA CONTEXT:
{data_context}

MODEL CONTEXT:
{model_context}

The ML ensemble can predict these columns: **{targets_str}**.
When the user asks about predicting ANY of these columns, the system will AUTOMATICALLY run the correct model and provide the prediction result below. Reference that result in your answer.

Your job:
1. Answer questions about the data — trends, patterns, distributions, correlations.
2. Answer questions about the model — how it works, what features matter most, how to interpret predictions.
3. When the user asks a "predict" or "what if" question, the system runs the matching model and provides the prediction below. Always reference that result.
4. **Highlight key numbers in bold.** Example: "The model predicts **₹52,340** for Rate_per_SqFt."
5. If the model has performance metrics (accuracy, R², MSE), reference them: "The model achieves R² of **0.87** on training data."
6. Give clear, actionable insights with numbers from the data.
7. If you cannot answer from the data or model, say so honestly."""

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    st.subheader("Chat with Your Model & Data")

    with st.container(height=420):
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    def _answer_question(question):
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                pred_info = ""
                target_col = _detect_target_column(question, list(df.columns), label_cols[0] if label_cols else None)
                used_model = model
                used_features = feature_names
                used_label = label_cols[0] if label_cols else None
                if model_ensemble and target_col and target_col in model_ensemble:
                    entry = model_ensemble[target_col]
                    used_model = entry["model"]
                    used_features = entry["features"]
                    used_label = target_col
                inp = _try_extract_prediction_inputs(question, used_features, df)
                if inp:
                    try:
                        row = pd.DataFrame([inp])
                        pred = used_model.predict(row)
                        n_filled = len(used_features) - len(inp)
                        fill_note = f" ({n_filled} feature{'s' if n_filled != 1 else ''} auto-filled from data averages)" if n_filled > 0 else ""
                        pred_info = f"\n\n[MODEL RUN] Inputs: {inp} -> Predicted {used_label}: {pred[0]}{fill_note}"
                        if hasattr(used_model, "predict_proba"):
                            try:
                                probs = used_model.predict_proba(row)
                                prob_str = "; ".join([f"{c}: {p:.3f}" for c, p in zip(used_model.classes_, probs[0])])
                                pred_info += f"\nProbabilities: {prob_str}"
                            except Exception:
                                pass
                        st.session_state._last_pred = pred_info
                    except Exception as e:
                        fallback_lines = [f"\n\n[MODEL UNAVAILABLE — {e}]"]
                        if used_label and used_label in df.columns:
                            vals = df[used_label].dropna()
                            if pd.api.types.is_numeric_dtype(vals):
                                fallback_lines.append(f"[DATA] {used_label} — Mean: {vals.mean():.4f}, Median: {vals.median():.4f}, "
                                    f"Min: {vals.min():.4f}, Max: {vals.max():.4f}, Count: {len(vals)}")
                            else:
                                top = vals.value_counts().head(5)
                                fallback_lines.append(f"[DATA] {used_label} — Most common values: "
                                    + ", ".join(f"{k} ({v})" for k, v in top.items()))
                            if inp:
                                matched_filters = []
                                for k, v in inp.items():
                                    if k in df.columns:
                                        matched_filters.append(f"{k} == {v!r}")
                                if matched_filters:
                                    filtered = df
                                    for k, v in inp.items():
                                        if k in df.columns:
                                            filtered = filtered[filtered[k] == v]
                                    fvals = filtered[used_label].dropna()
                                    if len(fvals) >= 3:
                                        if pd.api.types.is_numeric_dtype(fvals):
                                            fallback_lines.append(f"[DATA] Filtered ({'; '.join(matched_filters)}) — "
                                                f"Mean: {fvals.mean():.4f}, Median: {fvals.median():.4f}, "
                                                f"Count: {len(fvals)}")
                                        else:
                                            fallback_lines.append(f"[DATA] Filtered ({'; '.join(matched_filters)}) — "
                                                f"Most common: {fvals.value_counts().head(3).to_dict()}")
                        pred_info = "\n".join(fallback_lines)
                        st.session_state._last_pred = pred_info
                elif _is_prediction_question(question):
                    pred_info = "\n\n[MODEL RUN on data] " + _run_bulk_predictions(used_model, df, used_features, used_label is not None)
                    st.session_state._last_pred = pred_info

                user_msg = question + pred_info
                try:
                    reply = _query_llm(system_prompt, user_msg, api_key, st.session_state.chat_messages[:])
                except Exception as e:
                    reply = f"Error contacting LLM: {e}"

                st.markdown(reply)

                last_pred = st.session_state.get("_last_pred")
                if last_pred:
                    with st.expander("Key Metrics", expanded=False):
                        st.code(last_pred.replace("[MODEL RUN] ", "").replace("[MODEL RUN on data] ", ""))
                        if gen_metrics:
                            is_ensemble = isinstance(gen_metrics, dict) and not any(isinstance(v, float) for v in gen_metrics.values())
                            if is_ensemble:
                                per_target = {}
                                for t, m in gen_metrics.items():
                                    flat_m = {f"{t}_{k}": round(v, 4) for k, v in m.items() if isinstance(v, (int, float))}
                                    per_target.update(flat_m)
                                if per_target:
                                    st.caption("Model Performance")
                                    st.code(json.dumps(per_target, indent=2))
                            else:
                                flat = {k: round(v, 4) for k, v in gen_metrics.items() if isinstance(v, (int, float))}
                                if flat:
                                    st.caption("Model Performance")
                                    st.code(json.dumps(flat, indent=2))
                    st.session_state._last_pred = None
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})

    pending = st.session_state.pop("_pending_suggestion", None)
    if pending:
        st.session_state.chat_messages.append({"role": "user", "content": pending})
        with st.chat_message("user"):
            st.markdown(pending)
        _answer_question(pending)
        cache = st.session_state.get("_suggestions_cache", [])
        if pending in cache:
            cache.remove(pending)
        if len(cache) < 2:
            new_qs = _suggested_questions(df, feature_names, label_cols, task_type, gen_metrics, api_key, model_ensemble)
            for q in new_qs:
                if q not in cache and q != pending:
                    cache.append(q)
                    if len(cache) >= 3:
                        break

    if "_suggestions_cache" not in st.session_state:
        st.session_state._suggestions_cache = _suggested_questions(df, feature_names, label_cols, task_type, gen_metrics, api_key, model_ensemble)
    suggestions = st.session_state._suggestions_cache
    if suggestions:
        st.caption("Try asking:")
        ncols = min(3, len(suggestions))
        for i in range(0, len(suggestions), ncols):
            row = suggestions[i:i + ncols]
            cols = st.columns(ncols)
            for j, sq in enumerate(row):
                if cols[j].button(sq, key=f"sq_{i + j}", use_container_width=True):
                    st.session_state._pending_suggestion = sq
                    st.rerun()

    question = st.chat_input("Ask about your data or model...")
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        _answer_question(question)

    if st.button("Clear Chat History"):
        st.session_state.chat_messages = []
        st.session_state.pop("_suggestions_cache", None)
        st.rerun()


def show_playground(df: pd.DataFrame, model_instance, model_module, model_code: str, text_cols: list, label_cols: list, env_api_key: str = "", gen_metrics: dict = None, model_ensemble: dict = None):
    info = {"exists": model_instance is not None}
    if not info["exists"]:
        st.warning("No trained model found. Train a model on the Fine-Tune screen first.")
        return

    # Delete model option (clears from browser session)
    with st.sidebar:
        st.markdown("---")
        if st.button("Delete Trained Model", type="primary", use_container_width=True):
            for k in ["gen_model_instance", "gen_model_module", "gen_model_code", "gen_metrics", "gen_dir", "gen_model_ensemble"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.success("Model deleted from browser memory.")
            st.stop()

    model = model_instance
    module = model_module
    feature_names = get_feature_names(module, text_cols, label_cols)
    if not feature_names and model_ensemble:
        feature_names = list(next(iter(model_ensemble.values()))["features"])
    feature_names = [c for c in feature_names if c in df.columns]
    task_type = detect_task_type(module, label_cols, model)
    has_labels = bool(label_cols)

    st.subheader("Model Info")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Task Type", task_type.title())
    with c2:
        st.metric("Features", len(feature_names))
    with c3:
        st.metric("Samples", len(df))

    if gen_metrics:
        st.subheader("Performance Metrics")
        is_ensemble = isinstance(gen_metrics, dict) and not any(isinstance(v, float) for v in gen_metrics.values())
        if is_ensemble:
            targets_to_show = list(gen_metrics.keys())
            for target in targets_to_show[:5]:
                m = gen_metrics[target]
                cols = st.columns(3)
                i = 0
                for k, v in m.items():
                    if isinstance(v, float):
                        with cols[i % 3]:
                            st.metric(f"{target} {k}".replace("_", " ").title(), f"{v:.4f}")
                        i += 1
        else:
            cols = st.columns(3)
            i = 0
            for k, v in gen_metrics.items():
                if isinstance(v, float):
                    with cols[i % 3]:
                        st.metric(k.replace("_", " ").title(), f"{v:.4f}")
                    i += 1
            for k, v in gen_metrics.items():
                if isinstance(v, dict) and k == "classification_report":
                    with st.expander("Classification Report", expanded=False):
                        st.json(v)
                elif not isinstance(v, float) and not isinstance(v, str):
                    with st.expander(k.replace("_", " ").title(), expanded=False):
                        st.write(v)
        st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Data Explorer", "Feature Analysis", "Chat"])

    with tab1:
        if has_labels:
            inp = build_prediction_inputs(df, feature_names, label_cols)
            if st.button("Predict", type="primary", use_container_width=True):
                pred, probs = predict_from_inputs(model, inp, feature_names, module, label_cols)
                st.success(f"Prediction: **{pred[0]}**")
                if probs is not None:
                    st.subheader("Probabilities")
                    prob_df = pd.DataFrame(probs, columns=model.classes_ if hasattr(model, "classes_") else [f"class_{i}" for i in range(probs.shape[1])])
                    st.dataframe(prob_df.round(4), use_container_width=True)
        else:
            st.info("Unsupervised model (clustering). Use the Data Explorer tab to visualize clusters.")

        st.divider()
        st.subheader("Bulk Predict")
        bulk_file = st.file_uploader("Upload CSV to predict", type=["csv"], key="bulk_pred")
        if bulk_file and st.button("Run Bulk Prediction"):
            bulk_df = pd.read_csv(bulk_file)
            available = [c for c in feature_names if c in bulk_df.columns]
            missing = [c for c in feature_names if c not in bulk_df.columns]
            if missing:
                st.warning(f"Missing columns (will be auto-filled from data): {missing}")
            preds = model.predict(bulk_df[available])
            out = bulk_df.copy()
            out["prediction"] = preds
            if has_labels and hasattr(model, "predict_proba"):
                probs = model.predict_proba(bulk_df[available])
                for i, cl in enumerate(model.classes_ if hasattr(model, "classes_") else range(probs.shape[1])):
                    out[f"prob_{cl}"] = probs[:, i]
            st.dataframe(out.head(20), use_container_width=True)
            csv = out.to_csv(index=False).encode()
            st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")

    with tab2:
        st.subheader("Data Distribution")
        plot_cols = st.multiselect("Select columns to visualize", df.columns, default=df.select_dtypes(include=np.number).columns[:4].tolist())
        if plot_cols:
            import plotly.express as px
            if has_labels and label_cols[0] in df.columns:
                for col in plot_cols[:4]:
                    fig = px.histogram(df, x=col, color=label_cols[0], title=col, marginal="box")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                for col in plot_cols[:4]:
                    fig = px.histogram(df, x=col, title=col, marginal="box")
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Correlation Heatmap")
        num_df = df.select_dtypes(include=np.number)
        if num_df.shape[1] >= 2:
            import plotly.express as px
            fig = px.imshow(num_df.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Scatter Matrix")
        scatter_cols = st.multiselect("Columns for scatter matrix", num_df.columns, default=num_df.columns[:4].tolist(), key="scatter_cols")
        if len(scatter_cols) >= 2:
            import plotly.express as px
            color_col = None
            if has_labels and label_cols[0] in df.columns:
                color_col = label_cols[0]
            fig = px.scatter_matrix(df, dimensions=scatter_cols, color=color_col, opacity=0.6)
            fig.update_traces(diagonal_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        if not has_labels:
            st.divider()
            st.subheader("Cluster Analysis")
            cluster_features = [c for c in feature_names if c in df.columns]
            if not cluster_features:
                st.info("No valid feature columns for cluster analysis.")
            else:
                try:
                    preds = model.predict(df[cluster_features])
                    cluster_df = df.copy()
                    cluster_df["cluster"] = preds
                    st.dataframe(cluster_df.groupby("cluster").agg(["count", "mean"]).round(2), use_container_width=True)
                    if len(cluster_features) >= 2:
                        from sklearn.decomposition import PCA
                        import plotly.express as px
                        pca = PCA(n_components=2)
                        coords = pca.fit_transform(df[cluster_features])
                        viz_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": preds})
                        fig = px.scatter(viz_df, x="PC1", y="PC2", color="cluster", title="PCA Cluster Visualization")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.caption(f"Cluster viz skipped: {e}")

    with tab3:
        st.subheader("Feature Importance")
        fi = extract_feature_importance(model, feature_names)
        if fi is not None:
            import plotly.express as px
            fig = px.bar(fi.head(15), x="importance", y="feature", orientation="h", title="Top 15 Features by Importance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

        st.divider()
        st.subheader("Data Summary")
        st.dataframe(df.describe(), use_container_width=True)
        st.divider()
        st.subheader("Raw Data Sample")
        st.dataframe(df.head(100), use_container_width=True)
        csv_all = df.to_csv(index=False).encode()
        st.download_button("Download Full Data CSV", csv_all, "full_data.csv", "text/csv")

    with tab4:
        api_key = env_api_key or st.text_input("Groq API Key", type="password", key="chat_api_key",
                                help="Enter your Groq API key to enable chat. Get one at console.groq.com")
        if not api_key:
            st.info("Enter a Groq API key above to start chatting with your model and data.")
        else:
            show_chat_tab(api_key, df, model, module, feature_names, label_cols, task_type, has_labels, gen_metrics, model_ensemble)


