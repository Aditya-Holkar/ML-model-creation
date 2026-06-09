import json
import os
import threading
from pathlib import Path


MODEL_REGISTRY = {
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
}


def train(
    jsonl_path: str,
    model_name: str = "TinyLlama-1.1B",
    num_epochs: int = 1,
    output_dir: str = None,
    log_file: str = None,
):
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from accelerate import Accelerator

    def _log(msg):
        print(msg)
        if log_file:
            with open(log_file, "a") as f:
                f.write(msg + "\n")

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}")

    hf_model = MODEL_REGISTRY[model_name]
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "output", "model")
    os.makedirs(output_dir, exist_ok=True)

    _log(f"[0/4] Loading model: {hf_model}")

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=jsonl_path, split="train")

    def tokenize_fn(examples):
        texts = []
        for row in examples.get("text", examples.get(examples.keys()[0], [])):
            texts.append(str(row))
        return tokenizer(texts, truncation=True, padding="max_length", max_length=256)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    _log(f"[1/4] Dataset loaded: {len(tokenized)} examples")

    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    _log("[2/4] Model loaded with QLoRA")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    _log("[3/4] LoRA adapters attached")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_drop_last=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    _log(f"[4/4] Training started ({num_epochs} epoch(s))")
    trainer.train()
    _log("Training complete")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    _log(f"Model saved to {output_dir}")
    _log("__DONE__")


def train_async(jsonl_path, model_name, num_epochs, output_dir, log_file):
    thread = threading.Thread(
        target=train,
        args=(jsonl_path, model_name, num_epochs, output_dir, log_file),
        daemon=True,
    )
    thread.start()
    return thread
