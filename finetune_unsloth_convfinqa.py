# finetune_unsloth_convfinqa_hf.py

from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch

MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"

# Load Unsloth-compatible LLaMA 3 model
max_seq_length = 2048
use_flash_attn = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    dtype = torch.float16,
    load_in_4bit = True,
    use_flash_attention_2 = use_flash_attn,
)

# Load dataset from Hugging Face Hub
raw_dataset = load_dataset("TheFinAI/CONVFINQA_train", split="train")

# Reformat into prompt-completion examples
def format_example(example):
    history = example.get("history", [])
    question = example.get("question", "")
    paragraphs = example.get("paragraphs", [])
    table = example.get("table", [])
    program = example.get("program", [])

    # Join context
    history_block = " ".join(history)
    context_block = "\n".join(paragraphs)
    table_block = "\n".join([f"{row['name']}: {row['value']}" for row in table])
    full_question = f"{history_block} {question}".strip()

    # Final prompt
    prompt = f"""<|user|>
Question: {full_question}
Context:
{context_block}

Table:
{table_block}

Generate the program to compute the answer.
<|assistant|>"""

    program_text = " ".join(program)
    return {"text": prompt + " " + program_text}

# Apply formatting
dataset = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)

# Tokenize
tokenized = dataset.map(lambda x: tokenizer(x["text"]), remove_columns=["text"])

# Prepare model for SFT
model = FastLanguageModel.prepare_model_for_training(model)

# Trainer setup
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
        output_dir = "checkpoints/unsloth-convfinqa-hf",
        save_strategy = "epoch",
        report_to = "none",
    ),
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
