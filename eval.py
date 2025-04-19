import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import json, re
import wandb

# --- Load Model & Tokenizer ---
model_path = "outputs"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# --- Load Dev Subset ---
eval_set = load_dataset("TheFinAI/CONVFINQA_dev", split="train[:50]")

# --- Execution Logic ---
def execute_program(tokens):
    try:
        return eval("".join(tokens))
    except:
        return None

# --- Prompt Builder ---
def build_prompt(example):
    dialogue = ""
    for turn in example["entries"]:
        if turn["role"] == "human":
            dialogue += f"<|user|>\n{turn['content'].strip()}\n"
        elif turn["role"] == "agent":
            dialogue += f"<|assistant|>\n{turn['content'].strip()}\n"
    dialogue += "<|user|>\nPlease generate the output in the following format:\nProgram: ...\nAnswer: ..."
    return dialogue.strip()

# --- W&B Init ---
wandb.init(project="finqa-eval", name="llama3-exec-accuracy")
table = wandb.Table(columns=["question", "gold_answer", "pred_program", "pred_answer", "exec_result", "exec_match"])

results = []
for example in eval_set:
    prompt = build_prompt(example)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.0, do_sample=False)
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract structured outputs
    prog_match = re.search(r"Program:\s*(.*?)\n", completion)
    ans_match = re.search(r"Answer:\s*(.*?)\n?", completion)

    pred_program = prog_match.group(1).strip().split() if prog_match else []
    pred_answer = ans_match.group(1).strip() if ans_match else ""

    try:
        gold_answer = example["entries"][-1]["content"].strip()
        gold_answer = float(re.sub(r"[^\d.\-]", "", gold_answer))
    except:
        gold_answer = None

    exec_result = execute_program(pred_program)
    exec_match = exec_result is not None and abs(exec_result - gold_answer) < 1e-3

    results.append({
        "question": example["entries"][-1]["content"],
        "gold_answer": gold_answer,
        "pred_program": pred_program,
        "pred_answer": pred_answer,
        "exec_result": exec_result,
        "exec_match": exec_match,
    })

    table.add_data(
        example["entries"][-1]["content"],
        gold_answer,
        " ".join(pred_program),
        pred_answer,
        exec_result,
        exec_match
    )

# --- W&B Log ---
wandb.log({"execution_eval_table": table})
wandb.finish()

# --- Save Local Results ---
with open("eval_results.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print("✅ Saved results to eval_results.jsonl")
