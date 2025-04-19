import json
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# --- Init ---
run = wandb.init(project="finqa-eval", name="execution-benchmark")

# --- Config ---
model_path = "outputs/CHECKOOINT-200/"  # Your finetuned model
dataset_path = "TheFinAI/CONVFINQA_train"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
model.eval()

# --- Dataset ---
dataset = load_dataset(dataset_path, split="train[:50]")  # Fast loop

# --- Utils ---
def safe(val):
    if isinstance(val, (int, float, str)):
        return val
    if val is None or val is Ellipsis:
        return ""
    return str(val)

def extract_question(example):
    return example["entries"][-1]["content"]

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def parse_program_and_answer(completion):
    # Naive split for demo; replace with regex or structured logic
    if "Program:" in completion and "Answer:" in completion:
        try:
            prog = completion.split("Program:")[1].split("Answer:")[0].strip().split()
            ans = float(completion.split("Answer:")[1].strip().replace("%", ""))
            return prog, ans
        except Exception:
            return [], None
    return [], None

def execute_program(prog):
    try:
        return eval("".join(prog)) if prog else None
    except:
        return None

# --- Main loop ---
results = []
table = wandb.Table(columns=["question", "gold_answer", "pred_program", "pred_answer", "exec_result", "exec_match"])

for example in dataset:
    gold_answer = float(example["entries"][-1]["content"]) if example["entries"][-1]["content"].replace('.', '', 1).isdigit() else None
    prompt = "<|user|>\n" + extract_question(example) + "\n<|assistant|>\nPlease answer with:\nProgram:...\nAnswer:... %."

    completion = generate_answer(prompt)
    pred_program, pred_answer = parse_program_and_answer(completion)
    exec_result = execute_program(pred_program)
    exec_match = exec_result is not None and gold_answer is not None and abs(exec_result - gold_answer) < 1e-3

    results.append({
        "question": safe(extract_question(example)),
        "gold_answer": safe(gold_answer),
        "pred_program": safe(pred_program),
        "pred_answer": safe(pred_answer),
        "exec_result": safe(exec_result),
        "exec_match": exec_match,
    })

    table.add_data(
        safe(extract_question(example)),
        safe(gold_answer),
        safe(" ".join(pred_program)),
        safe(pred_answer),
        safe(exec_result),
        exec_match,
    )

# --- Log everything ---
wandb.log({"execution_eval_table": table})
wandb.finish()

# Optional local dump
with open("eval_results.jsonl", "w") as f:
    for r in results:
        json.dump({k: safe(v) for k, v in r.items()}, f)
        f.write("\n")
