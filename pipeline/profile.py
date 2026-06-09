import pandas as pd
from ydata_profiling import ProfileReport


def generate_profile(df: pd.DataFrame) -> str:
    report = ProfileReport(df, title="Data Profile", minimal=True, html={"style": {"full_width": True}})
    return report.to_html()
