"""
Standalone GSM8K pass@k evaluation for a saved checkpoint.

Saves an eval log JSON in the same format as chat_rl.py training eval logs.

Example:
    torchrun --nproc_per_node=4 -m scripts.gsm8k_eval \
        --source rl --model-tag nanochat-d20 --step 115 \
        --run-name rl-gsm8k-nanochat-d20 --device-batch-size 8 --eval-examples 400
"""
import argparse
import json
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="rl", choices=["sft", "rl"])
parser.add_argument("--model-tag", type=str, required=True)
parser.add_argument("--step", type=int, required=True)
parser.add_argument("--run-name", type=str, required=True, help="Eval log folder name under chatrl_eval_logs/")
parser.add_argument("--device-batch-size", type=int, default=8, help="Samples per question (= k for pass@k)")
parser.add_argument("--eval-examples", type=int, default=400)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top-k", type=int, default=50)
args = parser.parse_args()

from nanochat.common import compute_init, compute_cleanup, get_dist_info, autodetect_device_type, print0
from nanochat.checkpoint_manager import load_model, get_base_dir
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

model, tokenizer, _ = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
engine = Engine(model, tokenizer)

task = GSM8K(subset="main", split="test")
max_examples = min(args.eval_examples, len(task))

model.eval()
records = []
for idx in range(ddp_rank, max_examples, ddp_world_size):
    conversation = task[idx]
    tokens = tokenizer.render_for_completion(conversation)
    prefix_length = len(tokens)
    generated_token_sequences, _ = engine.generate_batch(
        tokens,
        num_samples=args.device_batch_size,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    question = conversation["messages"][0]["content"]
    ref_parts = conversation["messages"][-1]["content"]
    ref_text = "".join(p["text"] for p in ref_parts if p["type"] == "text")
    outcomes = []
    for sample_tokens in generated_token_sequences:
        generated_text = tokenizer.decode(sample_tokens[prefix_length:])
        outcomes.append({
            "is_correct": task.evaluate(conversation, generated_text),
            "generated_text": generated_text,
        })
    records.append({"idx": idx, "question": question, "reference": ref_text, "outcomes": outcomes})
    total_so_far = len(records)
    passed = sum(any(o["is_correct"] for o in r["outcomes"][:1]) for r in records)
    print(f"\r\033[KRank {ddp_rank} | pass@1 so far: {passed}/{total_so_far}", end="", flush=True)

print()

# Gather records from all ranks onto rank 0
if ddp:
    import torch.distributed as dist
    gathered = [None] * ddp_world_size
    dist.all_gather_object(gathered, records)
    if master_process:
        records = [r for rank_records in gathered for r in rank_records]
        records.sort(key=lambda r: r["idx"])

if master_process:
    k_max = args.device_batch_size
    passk = {}
    for k in range(1, k_max + 1):
        passk[f"pass@{k}"] = sum(
            any(o["is_correct"] for o in r["outcomes"][:k]) for r in records
        ) / len(records)
    print0("pass@k: " + ", ".join(f"{key}={val:.4f}" for key, val in passk.items()))

    base_dir = get_base_dir()
    eval_log_dir = os.path.join(base_dir, "chatrl_eval_logs", args.run_name)
    os.makedirs(eval_log_dir, exist_ok=True)
    eval_log_path = os.path.join(eval_log_dir, f"eval_step_{args.step:06d}.json")
    with open(eval_log_path, "w") as f:
        json.dump({"step": args.step, "pass@k": passk, "records": records}, f, indent=2)
    print(f"Saved eval log: {eval_log_path}")

compute_cleanup()
