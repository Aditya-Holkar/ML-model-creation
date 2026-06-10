# DataForge — Model Ensemble Rework

**Date:** 2026-06-10
**Status:** Design Approved
**Based on brainstorming session**

---

## 1. Problem

The model training and prediction pipeline has three critical issues:

1. **Fallback models drop categorical features silently** — templates like `_safe_model_code()` use `df[text_cols].select_dtypes(include=[np.number])` which silently drops all categorical/text columns (Society, Location, Property_Type), training on 0 real features.
2. **Ensemble training is slow** — N models trained sequentially, each creating temp files, writing Python files, and loading modules via `importlib`, resulting in ~3-5s overhead per model unrelated to actual sklearn training.
3. **Chat predictions fail with incomplete inputs** — when user asks "predict Rate_per_SqFt when Carpet_Area is 1500", the model receives a DataFrame with only 1-2 columns populated and the rest NaN, causing preprocessing errors.

Additionally, the user wants both **training speed** and **prediction speed**.

## 2. Solution: Unified ColumnTransformer Pipeline + Parallel Ensemble

### 2.1 Core Architecture

Replace all 5 fallback model templates with one robust `GeneratedModel` class that uses sklearn's `ColumnTransformer` for proper mixed-type feature handling:

| Component | Implementation |
|-----------|---------------|
| Numeric pipeline | SimpleImputer(strategy="median") → StandardScaler |
| Categorical pipeline | SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore") |
| Column selection | `make_column_selector(dtype_include=np.number)` for numeric, `make_column_selector(dtype_include=["object","category"])` for categorical |
| Model | RandomForestRegressor/Classifier with n_estimators=50, max_depth=10 (down from 100 for 2x speed) |
| Combined via | `ColumnTransformer` with remainder="drop" |

### 2.2 Data Stats for Auto-Fill

During `train()`, the model computes and stores per-column statistics:
- Numeric columns: `median`
- Categorical columns: `mode` (most frequent value)
- Also stores: `min`, `max`, `dtype`

Stored as `self._data_stats: dict[str, dict]` for use during prediction.

### 2.3 Auto-Fill in Predict

`predict(self, X)` accepts a dict or DataFrame which may have only a subset of feature columns populated:

```python
def predict(self, X):
    if isinstance(X, dict):
        X = pd.DataFrame([X])
    X = self._auto_fill(X)  # fill missing columns with data stats
    return self.model.predict(
        self.preprocessor.transform(X[self.feature_names_])
    )
```

`_auto_fill()` fills each missing column:
- Numeric → column median
- Categorical → column mode (most frequent value)

This guarantees predict NEVER fails with missing column errors.

### 2.4 Parallel Training

Replace the sequential per-target training loop with `concurrent.futures.ThreadPoolExecutor`:

- Max workers: `min(8, len(target_columns))` (respects CPU core count)
- Each thread trains one GeneratedModel instance in memory
- No temp files, no file writes, no module loading per model
- Progress reported via ThreadPoolExecutor.as_completed()

Expected speedup: ~4-5x on 4-8 core machines (15 targets × 2s = 30s sequential → ~6-8s parallel)

## 3. Files Changed

### 3.1 `finetune/model_generator.py`

**`_HARDCODED_TEMPLATE` modifications:**
- Reduce `n_estimators` from 100 → 50 in RandomForest lines
- Add `max_depth=10` parameter
- Add `_auto_fill(self, X)` method
- Add `_compute_stats(self, df, cols)` method  
- Store stats as `self._data_stats` in `train()`

### 3.2 `app.py`

- **Remove:** `_quick_demo_code()` function (simple RandomForestClassifier demo — replaced by universal template)
- **Remove:** `_safe_model_code()` function (auto-detection template — replaced by ColumnTransformer version)
- **Modify:** `_model_code_for_assessment()` — replace the 3 internal templates (regression/clustering/classification) with ONE ColumnTransformer-based universal template. Keep the function itself for the manual code editor path.
- **Remove:** The 5 inline fallback model class strings
- **Modify:** `_train_single_target()` — use in-memory GeneratedModel, no file writes, no temp dirs
- **Modify:** Training loop — wrap with `ThreadPoolExecutor`
- **Modify:** Fallback path — use the new ColumnTransformer universal template instead of old numeric-only ones
- **Keep:** `_assess_training_readiness()` — still useful for data profiling UI

### 3.3 `pipeline/playground.py`

- **Modify `_try_extract_prediction_inputs()`:**
  - Add categorical regex: `rf"{col} is (\w+)"`, `rf"{col} = (\w+)"`, `rf"(\w+) {col}"`
  - Numeric regex already works: `rf"{col} (?:is|=|:)\s*([\d\.]+)"`
  - Flexible column matching: match `Carpet_Area` against `"carpet area"`, `"carpert area"` (fuzzy prefix), `Rate_per_SqFt` against `"rate per sqft"`, `"rate/sqft"`
  - Return partial dict with any features found (not all required)
- **Modify `_answer_question()`:**
  - Pass information about which features were user-provided vs auto-filled
  - Show auto-filled feature count in the `[MODEL RUN]` output

## 4. Prediction Flow (Chat)

Complete flow from user question to answer:

```
User types: "predict Rate_per_SqFt when Carpet_Area is 1500 and 3 bathrooms"
  ↓
_detect_target_column() → "Rate_per_SqFt" (regex match on question text)
  ↓
model = ensemble["Rate_per_SqFt"]  (correct model from ensemble)
  ↓
_try_extract_prediction_inputs():
  → regex extracts: {Carpet_Area: 1500.0, Bathrooms: 3.0}
  → {"Carpet_Area": 1500.0, "Bathrooms": 3.0}
  ↓
model.predict(partial_inputs):
  → _auto_fill() fills: Society=[mode], Price=[median], ...
  → preprocessor.transform(all_features)
  → model.predict(transformed)
  → returns 52340.0
  ↓
LLM assistant function receives:
  [MODEL RUN] Inputs: {Carpet_Area:1500, Bathrooms:3}
  → Predicted Rate_per_SqFt: 52340.0
  → Feature source: 2 user-provided, 8 auto-filled from data averages
  ↓
LLM responds: "Based on a flat with **1,500 sqft** Carpet_Area and **3 bathrooms**,
  the predicted Rate_per_SqFt is **₹52,340**. The model achieves R² of **0.87**
  on training data and the most important feature is Carpet_Area."
```

### 4.1 Manual Predict Tab

The existing "Predict" tab (Screen 5) already works with sliders (numeric) + dropdowns (categorical). No changes needed — it always provides all feature values.

### 4.2 Bulk Predict

The existing bulk CSV upload already calls `model.predict(bulk_df[available])`. With auto-fill, missing columns in the uploaded CSV are also auto-filled. No changes needed.

## 5. Speed Analysis

| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| Train 15 targets (sequential) | 30-45s | 6-10s | ~4-5x |
| Train 15 targets (parallel) | N/A (sequential only) | 6-10s on 4 cores | New capability |
| Predict single row | ~5ms | ~5ms (same) | No change |
| Predict bulk (1000 rows) | ~50ms | ~50ms | No change |
| Model template generation | ~3s (Groq API call) | ~3s (same) | No change |

## 6. Error Handling

- **Column missing in predict:** `_auto_fill()` catches and fills with data stats — never crash
- **Model training fails for one target:** caught in `as_completed()` loop, logged, other targets continue
- **All models fail:** user sees error message explaining the issue (empty dataframe, no numeric columns, etc.)
- **ColumnTransformer transform error:** rare (only if feature_names_ has columns not in DataFrame), handled by `_auto_fill()` which ensures all features are present

## 7. What Gets Removed

- `app.py`: Lines 39-141 (`_quick_demo_code`, `_safe_model_code`, `_model_code_for_assessment` with all 5 templates) — ~150 lines of dead fallback templates
- `app.py`: Fallback path in training loop that switches between generated code and template code

## 8. Out of Scope

- Database/persistent model storage (models remain in session_state)
- Multi-user support
- Deep learning models (LLM fine-tuning is already separate)
- Automated feature engineering (beyond what ColumnTransformer provides)
- Model versioning
