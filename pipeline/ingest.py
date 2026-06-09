import pandas as pd
import chardet
from pathlib import Path


def read_file(uploaded_file):
    ext = Path(uploaded_file.name).suffix.lower()

    if ext == ".csv":
        raw = uploaded_file.read()
        encoding = chardet.detect(raw)["encoding"] or "utf-8"
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding=encoding, engine="python")
    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    metadata = {
        "filename": uploaded_file.name,
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "extension": ext,
    }

    return df, metadata
