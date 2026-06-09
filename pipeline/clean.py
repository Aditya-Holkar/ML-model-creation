import re
import pandas as pd
import numpy as np


def drop_high_null(df: pd.DataFrame, threshold: float = 0.5) -> tuple:
    before = df.shape[1]
    df = df.loc[:, df.isnull().mean() < threshold]
    removed = before - df.shape[1]
    return df, {"cols_removed": removed, "reason": f"dropped columns with >{threshold*100:.0f}% nulls"}


def deduplicate(df: pd.DataFrame, subset: list = None) -> tuple:
    before = df.shape[0]
    df = df.drop_duplicates(subset=subset)
    removed = before - df.shape[0]
    return df, {"rows_removed": removed, "reason": "removed duplicate rows"}


def impute_missing(df: pd.DataFrame) -> tuple:
    changes = {"imputed_cells": 0, "reason": ""}
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")
        changes["imputed_cells"] += null_count
    changes["reason"] = f"imputed {changes['imputed_cells']} missing values"
    return df, changes


def parse_dates(df: pd.DataFrame) -> tuple:
    from pandas.api.types import is_object_dtype

    parsed_cols = []
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                converted = pd.to_datetime(df[col], infer_datetime_format=True)
                if converted.notna().sum() > 0.5 * len(df):
                    df[col] = converted
                    parsed_cols.append(col)
            except (ValueError, TypeError):
                pass
    return df, {"parsed_columns": parsed_cols, "reason": f"parsed {len(parsed_cols)} date columns"}


def strip_whitespace(df: pd.DataFrame) -> tuple:
    trimmed = 0
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            before_null = df[col].isnull().sum()
            df[col] = df[col].str.strip()
            trimmed += (df[col] != df[col]).sum() - before_null
    return df, {"reason": "stripped whitespace from all text columns"}


def lowercase_text(df: pd.DataFrame) -> tuple:
    affected = 0
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            lowered = df[col].str.lower()
            affected += lowered.notna().sum()
            df[col] = lowered
    return df, {"reason": f"lowercased text in {affected} cells"}


def remove_empty_rows(df: pd.DataFrame, subset: list = None) -> tuple:
    before = df.shape[0]
    df = df.dropna(how="all", subset=subset)
    removed = before - df.shape[0]
    return df, {"rows_removed": removed, "reason": "removed fully empty rows"}


def replace_placeholder_nulls(df: pd.DataFrame) -> tuple:
    placeholders = ["n/a", "na", "null", "none", "-", "--", "?"]
    replaced = 0
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            mask = df[col].str.strip().str.lower().isin(placeholders)
            df.loc[mask, col] = None
            replaced += mask.sum()
    return df, {"nulled_cells": replaced, "reason": f"replaced {replaced} placeholder nulls (N/A, null, -, etc.)"}


def remove_outliers(df: pd.DataFrame, z_threshold: float = 3.0) -> tuple:
    removed_total = 0
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 5:
            z = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z > z_threshold
            df = df[~outliers]
            removed_total += outliers.sum()
    return df, {"rows_removed": removed_total, "reason": f"removed {removed_total} outlier rows (z > {z_threshold})"}


def rename_to_snake_case(df: pd.DataFrame) -> tuple:
    renamed = []
    mapping = {}
    for col in df.columns:
        new = col.strip().lower()
        new = re.sub(r"[^a-z0-9_]", "_", new)
        new = re.sub(r"_+", "_", new).strip("_")
        if new != col:
            mapping[col] = new
            renamed.append(f"{col} -> {new}")
    if mapping:
        df = df.rename(columns=mapping)
    return df, {"reason": f"renamed {len(renamed)} columns to snake_case" if renamed else "no columns needed renaming"}
