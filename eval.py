import torch
import json
import re
import wandb
from datasets import load_dataset
from unsloth import FastLanguageModel

# --- Safe casting for W&B + JSON ---
def safe(val):
    try:
        if val is None:
            return None
        elif isinstance(val, (int, float, bool)):
            return val
        elif isinstance(val, str):
            val_strip = val.strip().replace(",", "")
            if val_strip.replace(".", "", 1).isdigit():
                return float(val_strip) if "." in val_strip else int(val_strip)
            return val
        elif isinstance(val, list):
            return " ".join(map(str, val))
        return str(val)
    except Exception:
        return None

# --- Load model & tokenizer ---
model_path = "outputs/checkpoint-200"  
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=False,
    attn_implementation="flash_attention_2",
)

# --- Load dev subset ---
eval_set = load_dataset("TheFinAI/CONVFINQA_dev", split="train[:50]")

# --- Init Weights & Biases ---
wandb.init(project="finqa-eval", name="llama3-checkpoint-200")
table = wandb.Table(columns=[
    "question", "gold_answer", "pred_program", "pred_answer", "exec_result", "exec_match"
])

# --- Utility functions ---
def build_prompt(example):
    dialogue = ""
    for turn in example["entries"]:
        if turn["role"] == "human":
            dialogue += f"<|user|>\n{turn['content'].strip()}\n"
        elif turn["role"] == "agent":
            dialogue += f"<|assistant|>\n{turn['content'].strip()}\n"
    dialogue += "<|user|>\nPlease generate the output in the following format:\nProgram: ...\nAnswer: ..."
    return dialogue.strip()

def execute_program(tokens):
    try:
        return eval("".join(tokens))
    except:
        return None

def parse_answer(text):
    numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
    if numbers:
        return float(numbers[0])
    return None

# --- Evaluation loop ---
results = []
for example in eval_set:
    question = example["entries"][-1]["content"]
    prompt = build_prompt(example)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.0, do_sample=False)
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract structured outputs
    prog_match = re.search(r"Program:\s*(.*?)\n", completion)
    ans_match = re.search(r"Answer:\s*(.*?)\n?", completion)

    pred_program = prog_match.group(1).strip().split() if prog_match else []
    pred_answer = ans_match.group(1).strip() if ans_match else ""

    exec_result = execute_program(pred_program)
    gold_answer = parse_answer(question)

    if isinstance(exec_result, float) and isinstance(gold_answer, float):
        exec_match = abs(exec_result - gold_answer) < 1e-3
    else:
        exec_match = False

    # Save to list + W&B table
    row = {
        "question": safe(question),
        "gold_answer": safe(gold_answer),
        "pred_program": safe(pred_program),
        "pred_answer": safe(pred_answer),
        "exec_result": safe(exec_result),
        "exec_match": exec_match,
    }
    results.append(row)

    table.add_data(
        row["question"],
        row["gold_answer"],
        row["pred_program"],
        row["pred_answer"],
        row["exec_result"],
        row["exec_match"]
    )

# --- Final logging ---
wandb.log({"execution_eval_table": table})
wandb.finish()

# --- Optional: Save locally ---
with open("eval_results.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print("✅ Evaluation complete. Logged to W&B and saved to eval_results.jsonl")
