import io
import base64
import pandas as pd
import numpy as np


def generate_profile(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    buf.write("<html><head><style>")
    buf.write("body{font-family:sans-serif;margin:20px;color:#333}")
    buf.write("table{border-collapse:collapse;width:100%;margin:10px 0}")
    buf.write("th,td{border:1px solid #ddd;padding:8px;text-align:left}")
    buf.write("th{background:#f5f5f5}")
    buf.write("h2{color:#555;border-bottom:2px solid #eee;padding-bottom:5px}")
    buf.write(".card{display:inline-block;background:#f9f9f9;border:1px solid #ddd;border-radius:6px;padding:12px 20px;margin:5px;text-align:center}")
    buf.write(".card-num{font-size:24px;font-weight:bold;color:#333}")
    buf.write(".card-label{font-size:12px;color:#888}")
    buf.write(".warn{background:#fff3cd;color:#856404;padding:10px;border-radius:4px}")
    buf.write("</style></head><body>")
    buf.write("<h1>Data Profile</h1>")

    n, c = df.shape
    mem = df.memory_usage(deep=True).sum()
    for b in ["bytes", "KB", "MB", "GB"]:
        if mem < 1024:
            mem_str = f"{mem:.1f} {b}"
            break
        mem /= 1024
    buf.write(f"<div class='card'><div class='card-num'>{n:,}</div><div class='card-label'>Rows</div></div>")
    buf.write(f"<div class='card'><div class='card-num'>{c}</div><div class='card-label'>Columns</div></div>")
    buf.write(f"<div class='card'><div class='card-num'>{mem_str}</div><div class='card-label'>Memory</div></div>")
    buf.write(f"<div class='card'><div class='card-num'>{df.duplicated().sum():,}</div><div class='card-label'>Duplicates</div></div>")

    buf.write("<h2>Missing Values</h2>")
    missing = df.isnull().sum()
    missing_pct = (missing / n * 100).round(1)
    mv = pd.DataFrame({"Column": missing.index, "Missing": missing.values, "Percent": missing_pct.values})
    mv = mv[mv["Missing"] > 0].sort_values("Missing", ascending=False)
    if len(mv):
        buf.write(mv.to_html(index=False))
    else:
        buf.write("<p>No missing values found.</p>")

    buf.write("<h2>Column Summary</h2>")
    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        nulls = df[col].isnull().sum()
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            extra = f"min={desc['min']:.2f}, max={desc['max']:.2f}, mean={desc['mean']:.2f}"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            extra = f"{df[col].min()} → {df[col].max()}"
        else:
            top = df[col].value_counts().index[0] if nunique > 0 else ""
            extra = f"top: {str(top)[:40]}, freq: {df[col].value_counts().iloc[0] if nunique > 0 else 0}"
        rows.append({"Column": col, "Type": dtype, "Unique": nunique, "Nulls": nulls, "Details": extra})
    buf.write(pd.DataFrame(rows).to_html(index=False))

    buf.write("<h2>Numeric Correlations (top 20 pairs)</h2>")
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] > 1:
        corr = numeric.corr().abs().unstack().sort_values(ascending=False)
        corr = corr[corr < 1].drop_duplicates().head(20)
        pairs = pd.DataFrame({"Pair": [str(k) for k in corr.index], "Correlation": corr.values.round(3)})
        buf.write(pairs.to_html(index=False))
    else:
        buf.write("<p>Not enough numeric columns for correlation.</p>")

    buf.write("</body></html>")
    return buf.getvalue()
