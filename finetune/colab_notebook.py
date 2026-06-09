import nbformat as nbf


def generate_colab_notebook(hf_dataset_repo: str, model_name: str = "TinyLlama-1.1B") -> str:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
    }

    cells = []

    cells.append(nbf.v4.new_code_cell(
        "!pip install -q transformers peft bitsandbytes datasets accelerate torch"
    ))

    cells.append(nbf.v4.new_code_cell(
        'from huggingface_hub import notebook_login\nnotebook_login()'
    ))

    cells.append(nbf.v4.new_code_cell(
        f'HF_DATASET = "{hf_dataset_repo}"\n'
        f'MODEL_NAME = "{model_name}"\n'
        f'OUTPUT_REPO = "your-username/your-finetuned-model"  # change this'
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "### Load Dataset from Hugging Face Hub"
    ))

    cells.append(nbf.v4.new_code_cell(
        "from datasets import load_dataset\n\n"
        "dataset = load_dataset(HF_DATASET, split='train')\n"
        "print(f'Loaded {len(dataset)} examples')"
    ))

    cells.append(nbf.v4.new_markdown_cell("### Load Model with QLoRA"))

    cells.append(nbf.v4.new_code_cell(
        "import torch\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer\n"
        "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n\n"
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n"
        "tokenizer.pad_token = tokenizer.eos_token\n\n"
        "model = AutoModelForCausalLM.from_pretrained(\n"
        "    MODEL_NAME,\n"
        "    device_map='auto',\n"
        "    load_in_4bit=True,\n"
        "    torch_dtype=torch.float16,\n"
        ")\n\n"
        "model = prepare_model_for_kbit_training(model)\n\n"
        "lora_config = LoraConfig(\n"
        "    r=8, lora_alpha=32,\n"
        "    target_modules=['q_proj', 'v_proj'],\n"
        "    lora_dropout=0.05, bias='none',\n"
        "    task_type='CAUSAL_LM',\n"
        ")\n"
        "model = get_peft_model(model, lora_config)"
    ))

    cells.append(nbf.v4.new_markdown_cell("### Tokenize and Train"))

    cells.append(nbf.v4.new_code_cell(
        "def tokenize_fn(examples):\n"
        "    texts = [str(e) for e in examples['text']]\n"
        "    return tokenizer(texts, truncation=True, padding='max_length', max_length=512)\n\n"
        "tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)\n\n"
        "training_args = TrainingArguments(\n"
        "    output_dir='./model-output',\n"
        "    num_train_epochs=3,\n"
        "    per_device_train_batch_size=2,\n"
        "    gradient_accumulation_steps=4,\n"
        "    save_strategy='epoch',\n"
        "    logging_steps=10,\n"
        "    report_to='none',\n"
        "    fp16=True,\n"
        ")\n\n"
        "trainer = Trainer(\n"
        "    model=model,\n"
        "    args=training_args,\n"
        "    train_dataset=tokenized,\n"
        ")\n\n"
        "trainer.train()"
    ))

    cells.append(nbf.v4.new_markdown_cell("### Push to Hugging Face Hub"))

    cells.append(nbf.v4.new_code_cell(
        "trainer.save_model('./model-output')\n"
        "tokenizer.save_pretrained('./model-output')\n\n"
        "from huggingface_hub import HfApi\n"
        "api = HfApi()\n"
        "api.upload_folder(\n"
        "    folder_path='./model-output',\n"
        "    repo_id=OUTPUT_REPO,\n"
        "    repo_type='model',\n"
        ")"
    ))

    nb.cells = cells

    return nbf.writes(nb)
