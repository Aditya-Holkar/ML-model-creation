import json
import os
import re
import requests
import importlib.util
import sys


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SUGGEST_MODEL_PROMPT = """You are an expert ML consultant. Given a user's description and their actual data, suggest the best ML approach.

DATA COLUMNS AND TYPES:
{data_info}

USER GOAL:
{description}

Return a short, practical recommendation with:
- Problem type (classification/regression/clustering/other)
- Best algorithm fit (e.g. RandomForest, LinearRegression, KMeans)
- Why this approach works for their data
- How many rows/features they have
- Any preprocessing steps needed
- Expected training time (seconds)

Keep it concise, practical, and focused on THEIR data."""

GENERATE_MODEL_SNIPPET_PROMPT = \
'You are an ML engineer. Given data info and a task, generate ONLY the model-specific Python code snippet.\n' + \
'\n' + \
'DATA:\n' + \
'{data_info}\n' + \
'TASK: {description}\n' + \
'\n' + \
'Reply with TWO lines wrapped in ```python...```:\n' + \
'Line 1: MODEL = <sklearn model instantiation with hyperparameters>\n' + \
'Line 2: METRICS = <metrics dict with key "type" = "classify", "regress", or "cluster">\n' + \
'\n' + \
'Examples:\n' + \
'  For classification: MODEL = RandomForestClassifier(n_estimators=100, random_state=42)\n' + \
'                       METRICS = {{"type": "classify"}}\n' + \
'  For regression:     MODEL = RandomForestRegressor(n_estimators=100, random_state=42)\n' + \
'                       METRICS = {{"type": "regress"}}\n' + \
'  For clustering:     MODEL = KMeans(n_clusters=3, random_state=42, n_init=10)\n' + \
'                       METRICS = {{"type": "cluster"}}\n' + \
'\n' + \
'Rules:\n' + \
'- sklearn classes only (RandomForestClassifier, LogisticRegression, LinearRegression, RandomForestRegressor, KMeans, etc.)\n' + \
'- Set random_state=42 for reproducibility\n' + \
'- Do NOT generate any other code or explanation\n' + \
'- Only the two lines in ```python ... ```'

_HARDCODED_TEMPLATE = '''import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
import json, joblib

class GeneratedModel:
    def __init__(self):
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        self.preprocessor = ColumnTransformer([
            ("num", num_pipe, make_column_selector(dtype_include=np.number)),
            ("cat", cat_pipe, make_column_selector(dtype_include=["object","category"])),
        ], remainder="drop")
        self.model = __MODEL_PLACEHOLDER__
        self.metric_type = "__METRIC_TYPE__"
        self.feature_names_ = None

    def train(self, df, text_cols, label_cols):
        if label_cols and len(label_cols) > 0:
            feature_cols = [c for c in text_cols if c not in label_cols] or text_cols
            X = df[feature_cols].copy(); y = df[label_cols].values.ravel()
        else:
            X = df[text_cols].copy() if text_cols else df.copy(); y = None
        self.feature_names_ = X.columns.tolist()
        Xp = self.preprocessor.fit_transform(X)
        if y is not None:
            self.model.fit(Xp, y)
        else:
            try:
                self.model.fit(Xp)
            except TypeError:
                fallback_col = self.feature_names_.pop()
                y_fb = df[fallback_col].values.ravel()
                X_fb = df[self.feature_names_].copy()
                Xp_fb = self.preprocessor.fit_transform(X_fb)
                self.model.fit(Xp_fb, y_fb)

    def predict(self, X):
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        return self.model.predict(self.preprocessor.transform(X[self.feature_names_]))

    def evaluate(self, df, text_cols, label_cols):
        if not label_cols or len(label_cols) == 0:
            return {}
        feature_cols = [c for c in text_cols if c not in label_cols] or text_cols
        X = df[feature_cols].copy(); y = df[label_cols].values.ravel()
        Xp = self.preprocessor.transform(X)
        y_pred = self.model.predict(Xp)
        if self.metric_type == "classify":
            return {"accuracy": float(accuracy_score(y, y_pred)), "classification_report": classification_report(y, y_pred, output_dict=True)}
        elif self.metric_type == "regress":
            return {"r2_score": float(r2_score(y, y_pred)), "mse": float(mean_squared_error(y, y_pred))}
        else:
            return {}

    def save(self, path): joblib.dump(self, path)

    @staticmethod
    def load(path): return joblib.load(path)
'''

_CLASSIFY_METRICS = '''{"accuracy": float(accuracy_score(y, y_pred)), "classification_report": classification_report(y, y_pred, output_dict=True)}'''

_REGRESS_METRICS = '''{"r2_score": float(r2_score(y, y_pred)), "mse": float(mean_squared_error(y, y_pred))}'''


def _get_data_info(df_info: dict) -> str:
    cols = df_info.get("columns", [])
    dtypes = df_info.get("dtypes", {})
    sample = df_info.get("sample", {})
    lines = []
    for c in cols:
        t = dtypes.get(c, "unknown")
        ex = sample.get(c, [])
        examples = ", ".join(str(v) for v in ex[:3])
        lines.append(f"  - {c} ({t}): [{examples}]")
    return "\n".join(lines)


_DEFAULT_MODELS = {
    "classify": "RandomForestClassifier(n_estimators=100, random_state=42)",
    "regress": "LinearRegression()",
    "cluster": "KMeans(n_clusters=3, random_state=42, n_init=10)",
}


def _validate_model_def(model_def: str, metric_type: str) -> str:
    import ast
    try:
        tree = ast.parse(model_def.strip(), mode="eval")
        if isinstance(tree.body, ast.Call) and isinstance(tree.body.func, ast.Name):
            name = tree.body.func.id
            if name not in _SKLEARN_SHIM:
                print(f"WARN: Unknown model '{name}', falling back to default for '{metric_type}'")
                return _DEFAULT_MODELS.get(metric_type, "RandomForestClassifier(n_estimators=100, random_state=42)")
    except Exception:
        pass
    return model_def


def _strip_bad_kwargs(model_def: str) -> str:
    import ast
    try:
        tree = ast.parse(model_def.strip(), mode="eval")
        if isinstance(tree.body, ast.Call):
            # Collect known-good kwargs for common sklearn models
            model_name = tree.body.func.id if isinstance(tree.body.func, ast.Name) else ""
            safe_kwargs = _SAFE_KWARGS.get(model_name, None)
            if safe_kwargs is not None:
                ok_kwargs = []
                for kw in tree.body.keywords:
                    if kw.arg in safe_kwargs:
                        ok_kwargs.append(kw)
                if len(ok_kwargs) != len(tree.body.keywords):
                    args = [ast.copy_location(ast.arg(a.arg, a.annotation), a) for a in tree.body.args] if hasattr(tree.body, 'args') else tree.body.args
                    new_call = ast.Call(func=tree.body.func, args=tree.body.args, keywords=ok_kwargs)
                    ast.fix_missing_locations(new_call)
                    model_def = ast.unparse(new_call)
                    return model_def
    except Exception:
        pass

    # Fallback: try instantiating, strip unknown kwargs one by one
    import re
    for _ in range(5):
        try:
            eval(model_def, {"__builtins__": {}}, _SKLEARN_SHIM)
            break
        except (TypeError, NameError) as e:
            m = re.search(r"'(\w+)'", str(e))
            if m:
                kw = m.group(1)
                model_def = re.sub(r',?\s*' + re.escape(kw) + r'\s*=\s*[^,)]+', '', model_def)
            else:
                break
    return model_def


_SKLEARN_SHIM = {
    "RandomForestClassifier": __import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier,
    "RandomForestRegressor": __import__("sklearn.ensemble", fromlist=["RandomForestRegressor"]).RandomForestRegressor,
    "LogisticRegression": __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression,
    "LinearRegression": __import__("sklearn.linear_model", fromlist=["LinearRegression"]).LinearRegression,
    "KMeans": __import__("sklearn.cluster", fromlist=["KMeans"]).KMeans,
    "DecisionTreeClassifier": __import__("sklearn.tree", fromlist=["DecisionTreeClassifier"]).DecisionTreeClassifier,
    "DecisionTreeRegressor": __import__("sklearn.tree", fromlist=["DecisionTreeRegressor"]).DecisionTreeRegressor,
    "SVC": __import__("sklearn.svm", fromlist=["SVC"]).SVC,
    "SVR": __import__("sklearn.svm", fromlist=["SVR"]).SVR,
}

_SAFE_KWARGS = {
    "RandomForestClassifier": {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "random_state", "class_weight", "n_jobs"},
    "RandomForestRegressor": {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "random_state", "n_jobs"},
    "LogisticRegression": {"C", "penalty", "solver", "max_iter", "random_state", "class_weight"},
    "LinearRegression": {"fit_intercept", "copy_X", "n_jobs", "positive"},
    "KMeans": {"n_clusters", "init", "n_init", "max_iter", "random_state", "algorithm"},
    "DecisionTreeClassifier": {"max_depth", "min_samples_split", "min_samples_leaf", "random_state", "class_weight"},
    "DecisionTreeRegressor": {"max_depth", "min_samples_split", "min_samples_leaf", "random_state"},
    "SVC": {"C", "kernel", "degree", "gamma", "random_state", "class_weight", "probability"},
    "SVR": {"C", "kernel", "degree", "gamma", "epsilon"},
}


def suggest_model(description: str, df_info: dict, api_key: str) -> str:
    data_info_str = _get_data_info(df_info)
    prompt = SUGGEST_MODEL_PROMPT.format(data_info=data_info_str, description=description)
    resp = requests.post(GROQ_API_URL, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "system", "content": "You are a concise ML consultant. Give practical advice."}, {"role": "user", "content": prompt}], "temperature": 0.3}, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def generate_model(description: str, df_info: dict, api_key: str, output_dir: str = None) -> dict:
    data_info_str = _get_data_info(df_info)
    prompt = GENERATE_MODEL_SNIPPET_PROMPT.format(data_info=data_info_str, description=description)
    resp = requests.post(GROQ_API_URL, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "system", "content": "You generate sklearn model snippets. Two lines only. No extra text."}, {"role": "user", "content": prompt}], "temperature": 0.1}, timeout=30)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()

    if "```python" in content:
        content = content.split("```python", 1)[1]
    if "```" in content:
        content = content.rsplit("```", 1)[0]
    content = content.strip()

    model_def = "RandomForestClassifier(n_estimators=100, random_state=42)"
    metric_type = "classify"

    for line in content.split("\n"):
        s = line.strip()
        if s.startswith("MODEL ="):
            model_def = s[len("MODEL ="):].strip()
        elif s.startswith("METRICS"):
            import ast
            try:
                d = ast.literal_eval(s.split("=", 1)[1].strip())
                mt = d.get("type", "classify")
                if mt in ("classify", "regress", "cluster"):
                    metric_type = mt
            except Exception:
                pass

    model_def = _validate_model_def(model_def, metric_type)
    import ast as _ast
    model_def = _strip_bad_kwargs(model_def)

    code = _HARDCODED_TEMPLATE.replace("__MODEL_PLACEHOLDER__", model_def)
    code = code.replace('__METRIC_TYPE__', metric_type)

    if output_dir:
        model_path = os.path.join(output_dir, "generated_model.py")
        os.makedirs(output_dir, exist_ok=True)
    else:
        import tempfile
        model_path = os.path.join(tempfile.mkdtemp(prefix="dataforge_"), "generated_model.py")

    with open(model_path, "w") as f:
        f.write(code)

    try:
        compile(code, model_path, "exec")
        compiles = True
    except SyntaxError:
        compiles = False
        code = None

    return {"model_path": model_path, "code": code, "compiles": compiles}


def train_generated_model(model_path: str, df, text_cols: list, label_cols: list, output_dir: str, log_callback=None):
    def _log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    spec = importlib.util.spec_from_file_location("generated_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generated_model"] = mod
    spec.loader.exec_module(mod)

    _log("Generated model loaded successfully")
    model_instance = mod.GeneratedModel()
    _log(f"Training on {len(df)} rows...")
    model_instance.train(df, text_cols, label_cols)
    _log("Training complete")

    metrics = model_instance.evaluate(df, text_cols, label_cols)
    _log(f"Metrics: {json.dumps(metrics, indent=2)}")

    save_path = os.path.join(output_dir, "model.joblib")
    model_instance.save(save_path)
    _log(f"Model saved to {save_path}")

    return metrics
