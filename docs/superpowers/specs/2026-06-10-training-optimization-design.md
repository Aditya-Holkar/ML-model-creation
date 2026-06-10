# Training Speed Optimization Design

**Date:** 2026-06-10
**Project:** DataForge
**Status:** Draft

## Problem

Training the generated ML model ensemble on datasets of 10K-50K rows with 6+ target columns takes too long (~18s observed for 50K rows). Users need faster iteration, especially during model selection and parameter tuning.

## Root Cause Analysis

Three bottlenecks dominate training time:

1. **Redundant preprocessing (dominant):** Each of the 6+ ensemble models independently calls `preprocessor.fit_transform(X)` on the full dataset. The `ColumnTransformer` (imputation + scaling + one-hot encoding) runs N times on the same data.

2. **CPU oversubscription:** `ThreadPoolExecutor(max_workers=8)` launches 6+ single-threaded RandomForest fits in parallel. Each fit holds the GIL during pandas operations and competes for CPU during numpy operations. Context switching adds overhead without throughput gain.

3. **Repeated code compilation:** `exec(compile(code, ...))` runs once per target column, adding ~100ms × N overhead.

## Solution Overview

Three-pronged approach combining preprocessing reuse, efficient parallelism, and automatic sampling:

1. **Preprocessor reuse** — fit `ColumnTransformer` once, reuse across all targets
2. **Sequential + all-core training** — replace `ThreadPoolExecutor` with sequential loop, each model uses `n_jobs=-1`
3. **Automatic sampling** — silently train on 5K-row sample when data exceeds threshold, offer full retrain

## Detailed Design

### 1. Preprocessor Reuse

**Current behavior:**
```python
# app.py:797 called N times, each call does:
inst = GeneratedModel()
inst.train(tdf, feats, [target_col])  # fit_transform inside
```

**New behavior:**
- At the training entry point, extract the `ColumnTransformer` from a fresh `GeneratedModel()` instance
- Call `preprocessor.fit_transform(X)` once with all feature columns
- Pass the precomputed `Xp` (numpy array) and fitted preprocessor to each target model
- Each model calls `model.fit(Xp, y)` directly, skipping preprocessing

```python
# New flow in app.py
preprocessor = GeneratedModel().preprocessor
X = df[feature_cols]
Xp = preprocessor.fit_transform(X)  # Once

for target in target_columns:
    y = _prepare_y(df, [target])
    model = RandomForestClassifier(n_jobs=-1, n_estimators=100)
    model.fit(Xp, y)
    ensemble[target] = model
```

**Template impact:** The `GeneratedModel` class in `model_generator.py` needs a path for "preprocess then train" — either a new method or a flag to skip preprocessing. Since the template is auto-generated, the cleanest approach is to train standalone models (not using the template's `train()` method) and wrap them for the ensemble.

### 2. Sequential Training with n_jobs=-1

**Current:**
```python
with ThreadPoolExecutor(max_workers=8) as ex:
    fut_to_target = {
        ex.submit(_train_single_target, code, t, cols, log): t
        for t in target_columns
    }
```

**New:**
```python
ns = {}
exec(compile(result["code"], "<model>", "exec"), ns)
GeneratedModel = ns["GeneratedModel"]
preprocessor = GeneratedModel().preprocessor
Xp = preprocessor.fit_transform(feature_df)

for target in target_columns:
    y = _prepare_y(df, [target])
    model = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=42)
    model.fit(Xp, y)
    metrics = _compute_metrics(model, Xp, y)
    ensemble[target] = {"model": model, "metrics": metrics, "features": feature_cols}
```

Each `RandomForest` with `n_jobs=-1` parallelizes tree building across all available CPU cores (e.g., 8 trees simultaneously on 8 cores). Sequential execution avoids resource contention.

### 3. Automatic Sampling

- Threshold: `len(df) > 5000`
- When exceeded, silently `df.sample(n=5000, random_state=42)` before training
- Show badge: `⚠ Trained on 5K-row sample`
- Show button: `🔄 Train on Full Dataset (more accurate)` — re-runs on full data
- Metrics from sampled training include a note about sampling
- Small datasets (< 5000 rows): skip sampling entirely

### 4. Model Code Cache

- `exec(compile(...))` runs once at the start of training
- The class is imported into `sys.modules` under a unique name
- Subsequent calls look up the module instead of recompiling

### 5. Model Defaults Update

- `n_estimators=50` → `n_estimators=100` (better accuracy, same throughput with `n_jobs=-1`)
- Add `class_weight="balanced"` for classification targets
- Keep `max_depth=10` to prevent overfitting

## Files to Modify

| File | Changes |
|------|---------|
| `app.py` (lines 769-855) | Rewrite training loop: preprocessor reuse, sequential n_jobs=-1, sampling, progress display |
| `finetune/model_generator.py` | Update template defaults, add code cache utility function |
| `pipeline/playground.py` | No changes needed if models remain `joblib.dump`-compatible |

## Expected Performance

Measured on 8-core machine, 50K rows, 10 features, 6 targets:

| Scenario | Before | After | Speedup |
|----------|--------|-------|---------|
| Full training (50K rows) | ~18s | ~5s | 3.6× |
| Quick training (5K sample) | — | ~0.5s | 36× |
| 2 targets only (50K rows) | ~6s | ~2s | 3× |

## Edge Cases

- **Only 1 target:** Sequential loop runs once, no parallelism overhead. Still benefits from preprocessing reuse and n_jobs=-1.
- **Fewer features than targets:** Training is already fast (small Xp). Sampling not critical but harmless.
- **Categorical features with high cardinality:** Preprocessor reuse is even more impactful since OneHotEncoder is the most expensive step.
- **User cancels training mid-way:** Streamlit reruns the script. The long `fit_transform` can't be checkpointed — but with the new sequential loop, progress is shown per-target so user sees incremental completion.
