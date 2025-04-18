import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig
import wandb

# --- Config ---
model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
max_seq_length = 2048
batch_size = 2
gradient_accumulation_steps = 4
total_steps = 200
eval_steps = 50
save_steps = 50
logging_steps = 1
output_dir = "outputs"
project_name = "tomoro"

# --- Format as conversation ---
def format_conversation(example):
    dialogue = ""
    if "entries" in example and isinstance(example["entries"], list):
        for turn in example["entries"]:
            role = turn.get("role", "")
            content = turn.get("content", "").strip()
            if role == "human":
                dialogue += f"<|user|>\n{content}\n"
            elif role == "agent":
                dialogue += f"<|assistant|>\n{content}\n"
    return {"text": dialogue.strip()}

# --- Tokenize ---
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors=None,
        return_special_tokens_mask=True,
    )

# --- Load Dataset ---
dataset = load_dataset("TheFinAI/CONVFINQA_train", split="train")

# format entries -> text
dataset = dataset.map(format_conversation)

# remove unused columns (entries is gone already)
columns_to_remove = [col for col in ["id", "entries"] if col in dataset.column_names]
if columns_to_remove:
    dataset = dataset.remove_columns(columns_to_remove)

# --- Tokenize Dataset ---
dataset = dataset.map(tokenize, batched=True, batch_size=8)

# --- Train/Val Split ---
split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# --- Load Model & Tokenizer with LoRA already integrated ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=True,
    attn_implementation="flash_attention_2"
)

# --- Verify model is ready for training ---
model.print_trainable_parameters()
assert any(p.requires_grad for p in model.parameters()), "No parameters require gradients!"
model.train()

# --- Data Collator ---
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- W&B Login ---
wandb.init(project=project_name)

# --- Trainer ---
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    args=SFTConfig(
        dataset_text_field="text",
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=10,
        max_steps=total_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        optim="adamw_8bit",
        seed=3407,
        report_to="wandb",
        run_name="llama-convfinqa-sft",
        fp16=True,
        bf16=False,
    ),
)

# --- Train ---
trainer.train()

# --- Evaluation ---
print("__ Evaluating on validation set...")
eval_results = trainer.evaluate()
print("__ Eval Loss:", eval_results.get("eval_loss"))

# --- Inference Sample ---
sample = eval_dataset[0]
inputs = tokenizer(sample["text"], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print("__ Sample Output:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
