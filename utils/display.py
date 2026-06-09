import pandas as pd
import streamlit as st


def show_before_after(original: pd.DataFrame, cleaned: pd.DataFrame):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Original Rows", original.shape[0])
        st.metric("Original Columns", original.shape[1])
        st.dataframe(original.head(10), use_container_width=True)
    with col2:
        st.metric("Cleaned Rows", cleaned.shape[0], delta=cleaned.shape[0] - original.shape[0])
        st.metric("Cleaned Columns", cleaned.shape[1], delta=cleaned.shape[1] - original.shape[1])
        st.dataframe(cleaned.head(10), use_container_width=True)

    if st.checkbox("Show rows removed"):
        removed = original[~original.index.isin(cleaned.index)]
        if len(removed):
            st.dataframe(removed.head(20), use_container_width=True)
        else:
            st.info("No rows were removed.")


def show_format_preview(jsonl_str: str, n: int = 3):
    lines = jsonl_str.strip().split("\n")
    preview = "\n".join(lines[:n])
    st.code(preview, language="json")
    st.caption(f"Showing {min(n, len(lines))} of {len(lines)} rows")
