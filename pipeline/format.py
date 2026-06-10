import json
import pandas as pd


def _concat_cols(row, cols, sep="\n"):
    if not cols:
        return ""
    parts = [str(row[c]) for c in cols if c in row.index]
    return sep.join(parts)


def to_instruction_jsonl(df: pd.DataFrame, text_cols: list, label_cols: list = None) -> str:
    lines = []
    for _, row in df.iterrows():
        instruction = _concat_cols(row, text_cols)
        if label_cols:
            output = _concat_cols(row, label_cols)
        else:
            output = ""
        lines.append(json.dumps({"instruction": instruction, "output": output}))
    return "\n".join(lines)


def to_chat_jsonl(df: pd.DataFrame, text_cols: list, label_cols: list = None) -> str:
    lines = []
    for _, row in df.iterrows():
        content = _concat_cols(row, text_cols)
        messages = [{"role": "user", "content": content}]
        if label_cols:
            messages.append({"role": "assistant", "content": _concat_cols(row, label_cols)})
        lines.append(json.dumps({"messages": messages}))
    return "\n".join(lines)
