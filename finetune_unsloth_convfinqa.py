# finetune_unsloth_convfinqa.py

from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch, json

MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"

# Load Unsloth model
max_seq_length = 2048
use_flash_attn = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    dtype = torch.float16,
    load_in_4bit = True,
    use_flash_attention_2 = use_flash_attn
)

# Load and preprocess ConvFinQA
def load_and_format(path):
    with open(path, "r") as f:
        raw = json.load(f)

    samples = []
    for ex in raw:
        question = " ".join(ex["history"] + [ex["question"]])
        paragraphs = "\n".join(ex["paragraphs"])
        table = "\n".join([f"{row['name']}: {row['value']}" for row in ex["table"]])

        context = f"Context:\n{paragraphs}\n\nTable:\n{table}"
        instruction = f"Question: {question}\n{context}\n\nGenerate the program to compute the answer."

        samples.append({
            "input": instruction.strip(),
            "output": " ".join(ex["program"]).strip()
        })

    return Dataset.from_list(samples)

dataset = load_and_format("data/train_small.json")

# Format for SFT
def format_instruction(example):
    return f"<|user|>\n{example['input']}\n<|assistant|>\n{example['output']}"

dataset = dataset.map(lambda x: {"text": format_instruction(x)})

# Tokenize
tokenized = dataset.map(lambda x: tokenizer(x["text"]), remove_columns=["text"])

# Prepare for SFT
model = FastLanguageModel.prepare_model_for_training(model)

# Trainer config
trainer = Trainer(
    model = model,
    train_dataset = tokenized,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-5,
        fp16 = True,
        logging_steps = 10,
        output_dir = "checkpoints/unsloth-convfinqa",
        save_strategy = "epoch",
        report_to = "none"
    ),
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
