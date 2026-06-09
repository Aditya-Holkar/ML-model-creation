import os
import json
import inspect
import re
import importlib.util
import sys
import tempfile
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


def is_supervised(module) -> bool:
    code = inspect.getsource(module.GeneratedModel.train)
    return "label_cols" in code and "y = df[label_cols]" in code


def detect_task_type(module, label_cols: list) -> str:
    if not label_cols:
        return "clustering"
    src = inspect.getsource(module.GeneratedModel.__init__)
    if "RandomForestClassifier" in src or "LogisticRegression" in src:
        return "classification"
    if "RandomForestRegressor" in src or "LinearRegression" in src:
        return "regression"
    if "KMeans" in src or "DBSCAN" in src:
        return "clustering"
    return "unknown"


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
                dtype = str(df[col].dtype)
                cmin, cmax = float(df[col].min()), float(df[col].max())
                cmid = float(df[col].median())
                if "float" in dtype or "int" in dtype:
                    inputs[col] = st.slider(col, min_value=cmin, max_value=cmax, value=cmid, format="%.2f")
                else:
                    unique = df[col].dropna().unique().tolist()
                    inputs[col] = st.selectbox(col, unique)
    return inputs


def predict_from_inputs(model, inputs: dict, feature_names: list, module, label_cols: list):
    row = pd.DataFrame([inputs])[feature_names]
    pred = model.predict(row)
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
    parts.append(f"Features ({len(feature_names)}): {', '.join(feature_names)}")
    if label_cols:
        parts.append(f"Target: {', '.join(label_cols)}")
    shape = df.shape
    parts.append(f"Rows: {shape[0]}")
    numeric_cols = df[feature_names].select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        stats = df[numeric_cols].describe().to_dict()
        parts.append(f"Numeric stats: {json.dumps(stats, default=str)}")
    cat_cols = [c for c in feature_names if df[c].dtype in ("object", "category")]
    if cat_cols:
        val_counts = {}
        for c in cat_cols:
            val_counts[c] = df[c].value_counts().head(5).to_dict()
        parts.append(f"Categorical values: {json.dumps(val_counts, default=str)}")
    sample = df[feature_names + (label_cols or [])].head(5).to_dict(orient="records")
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
        for k, v in gen_metrics.items():
            if isinstance(v, (int, float)):
                flat[k] = round(v, 4)
        if flat:
            parts.append(f"Metrics: {json.dumps(flat)}")
    return "\n".join(parts)


def _try_extract_prediction_inputs(question: str, feature_names: list, df: pd.DataFrame) -> dict | None:
    inputs = {}
    found_any = False
    for col in feature_names:
        patterns = [
            rf"{re.escape(col)}\s*[=:]\s*([\d\.]+)",
            rf"{re.escape(col)}\s+is\s+([\d\.]+)",
            rf"{re.escape(col)}\s+(?:of|at)\s+([\d\.]+)",
        ]
        for pat in patterns:
            m = re.search(pat, question, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                inputs[col] = val
                found_any = True
                break
    return inputs if found_any else None


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


def show_chat_tab(api_key: str, df: pd.DataFrame, model, module, feature_names: list, label_cols: list, task_type: str, has_labels: bool, gen_metrics: dict = None):
    data_context = _build_data_context(df, feature_names, label_cols, task_type)
    model_context = _build_model_context(model, feature_names, has_labels, task_type, gen_metrics)

    system_prompt = f"""You are a data scientist assistant. You have access to a trained ML model and its dataset.

DATA CONTEXT:
{data_context}

MODEL CONTEXT:
{model_context}

Your job:
1. Answer questions about the data — trends, patterns, distributions, correlations.
2. Answer questions about the model — how it works, what features matter most, how to interpret predictions.
3. When the user asks "what if" or "predict" questions, the system will AUTOMATICALLY run the model and provide the prediction result below. Reference that result in your answer.
4. **Highlight key numbers and insights in bold.** For example: "The model predicts **$52,340** with **87%** confidence."
5. If the model has performance metrics (accuracy, R², MSE), reference them when relevant: "The model achieves **94%** accuracy on training data."
6. Give clear, actionable insights. Use numbers and cite specific values from the data.
7. If the user asks something you cannot answer from the data or model, say so honestly."""

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    st.subheader("Chat with Your Model & Data")

    with st.container(border=True, height=400):
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    question = st.chat_input("Ask about your data or model...")
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                pred_info = ""
                inp = _try_extract_prediction_inputs(question, feature_names, df)
                if inp:
                    try:
                        row = pd.DataFrame([inp])[feature_names]
                        pred = model.predict(row)
                        pred_info = f"\n\n[MODEL RUN] Inputs: {inp} -> Prediction: {pred[0]}"
                        if has_labels and hasattr(model, "predict_proba"):
                            probs = model.predict_proba(row)
                            prob_str = "; ".join([f"{c}: {p:.3f}" for c, p in zip(model.classes_, probs[0])])
                            pred_info += f"\nProbabilities: {prob_str}"
                        st.session_state._last_pred = pred_info
                    except Exception as e:
                        pred_info = f"\n\n[Model run attempted but failed: {e}]"
                elif _is_prediction_question(question):
                    pred_info = "\n\n[MODEL RUN on data] " + _run_bulk_predictions(model, df, feature_names, has_labels)
                    st.session_state._last_pred = pred_info

                user_msg = question + pred_info
                try:
                    reply = _query_llm(system_prompt, user_msg, api_key, st.session_state.chat_messages[:-1])
                except Exception as e:
                    reply = f"Error contacting LLM: {e}"

                st.markdown(reply)

                if st.session_state.get("_last_pred"):
                    with st.expander("Key Metrics", expanded=False):
                        st.code(st.session_state._last_pred.replace("[MODEL RUN] ", "").replace("[MODEL RUN on data] ", ""))
                        if gen_metrics:
                            flat = {k: round(v, 4) for k, v in gen_metrics.items() if isinstance(v, (int, float))}
                            if flat:
                                st.caption("Model Performance")
                                st.code(json.dumps(flat, indent=2))
                    st.session_state._last_pred = None
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})

    if st.button("Clear Chat History"):
        st.session_state.chat_messages = []
        st.rerun()


def show_playground(df: pd.DataFrame, model_instance, model_module, model_code: str, text_cols: list, label_cols: list, env_api_key: str = "", gen_metrics: dict = None):
    info = {"exists": model_instance is not None}
    if not info["exists"]:
        st.warning("No trained model found. Train a model on the Fine-Tune screen first.")
        return

    # Delete model option (clears from browser session)
    with st.sidebar:
        st.markdown("---")
        if st.button("Delete Trained Model", type="primary", use_container_width=True):
            for k in ["gen_model_instance", "gen_model_module", "gen_model_code", "gen_metrics", "gen_dir"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.success("Model deleted from browser memory.")
            st.stop()

    model = model_instance
    module = model_module
    feature_names = get_feature_names(module, text_cols, label_cols)
    task_type = detect_task_type(module, label_cols)
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
                st.error(f"Missing columns: {missing}")
            else:
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
            try:
                preds = model.predict(df[feature_names])
                cluster_df = df.copy()
                cluster_df["cluster"] = preds
                st.dataframe(cluster_df.groupby("cluster").agg(["count", "mean"]).round(2), use_container_width=True)
                if len(feature_names) >= 2:
                    from sklearn.decomposition import PCA
                    import plotly.express as px
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(df[feature_names])
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
            show_chat_tab(api_key, df, model, module, feature_names, label_cols, task_type, has_labels, gen_metrics)
