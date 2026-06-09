import json
import os
import time
import requests
from pathlib import Path


TOGETHER_API_BASE = "https://api.together.xyz/v1"


def upload_file(file_path: str, api_key: str) -> str:
    url = f"{TOGETHER_API_BASE}/files"
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(file_path, "rb") as f:
        resp = requests.post(url, headers=headers, files={"file": f})
    resp.raise_for_status()
    return resp.json()["id"]


def create_fine_tune(
    file_id: str,
    model: str,
    api_key: str,
    suffix: str = None,
    epochs: int = 3,
    learning_rate: float = 1e-5,
) -> str:
    url = f"{TOGETHER_API_BASE}/fine-tunes"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "training_file": file_id,
        "model": model,
        "n_epochs": epochs,
        "learning_rate": learning_rate,
    }
    if suffix:
        payload["suffix"] = suffix
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["id"]


def get_fine_tune_status(fine_tune_id: str, api_key: str) -> dict:
    url = f"{TOGETHER_API_BASE}/fine-tunes/{fine_tune_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def poll_until_done(fine_tune_id: str, api_key: str, log_callback=None, poll_interval: int = 30):
    def _log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    _log(f"Fine-tune job {fine_tune_id} started. Polling every {poll_interval}s...")
    while True:
        status = get_fine_tune_status(fine_tune_id, api_key)
        state = status.get("status", "unknown")
        _log(f"Status: {state}")
        if state in ("completed", "succeeded"):
            _log(f"Model ready: {status.get('output', {}).get('model', 'unknown')}")
            return status
        elif state in ("failed", "cancelled"):
            _log(f"Job {state}: {status.get('error', 'unknown error')}")
            raise RuntimeError(f"Fine-tune {state}: {status.get('error')}")
        time.sleep(poll_interval)


SUPPORTED_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-2b-it",
]
