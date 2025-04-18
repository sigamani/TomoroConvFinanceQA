import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
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

# --- Load Dataset ---
dataset = load_dataset("TheFinAI/CONVFINQA_train", split="train")

def format_conversation(example):
    dialogue = ""
    for turn in example["entries"]:
        role = turn["role"]
        content = turn["content"].strip()
        if role == "human":
            dialogue += f"<|user|>\n{content}\n"
        elif role == "agent":
            dialogue += f"<|assistant|>\n{content}\n"
    return {"text": dialogue.strip()}

dataset = dataset.map(format_conversation)
dataset = dataset.remove_columns(["id", "entries"])

# --- Load Model & Tokenizer ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=True,
    attn_implementation="flash_attention_2"
)

# --- Inject LoRA Adapters ---
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

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

dataset = dataset.map(tokenize, remove_columns=["text"], batched=True, batch_size=8)

# --- Split Dataset ---
split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

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
