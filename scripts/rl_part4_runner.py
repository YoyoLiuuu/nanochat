"""
Utility to generate (and optionally execute) standardized Part 4 RL runs.
(LOCAL ONLY, use nanochat_modal.py to run on modal)

Examples:
    # Print commands only
    python -m scripts.rl_part4_runner --owner J --print-only

    # Run all reward systems on 1 GPU
    python -m scripts.rl_part4_runner --owner K

    # Run all reward systems with torchrun on 4 GPUs
    python -m scripts.rl_part4_runner --owner J --launcher torchrun --nproc-per-node 4
"""

from __future__ import annotations

import argparse
import shlex
import subprocess


def build_commands(args: argparse.Namespace):
    reward_systems = [
        name.strip() for name in args.reward_systems.split(",") if name.strip()
    ]
    commands = []

    for reward_system in reward_systems:
        run_name = f"{args.run_prefix}-{args.owner.lower()}-{reward_system}"
        core = [
            "-m",
            "scripts.chat_rl",
            f"--run={run_name}",
            f"--reward-system={reward_system}",
            f"--num-epochs={args.num_epochs}",
            f"--device-batch-size={args.device_batch_size}",
            f"--examples-per-step={args.examples_per_step}",
            f"--num-samples={args.num_samples}",
            f"--max-new-tokens={args.max_new_tokens}",
            f"--eval-every={args.eval_every}",
            f"--eval-examples={args.eval_examples}",
            f"--save-every={args.save_every}",
        ]

        if args.model_tag:
            core.append(f"--model-tag={args.model_tag}")
        if args.model_step is not None:
            core.append(f"--model-step={args.model_step}")

        if args.launcher == "torchrun":
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={args.nproc_per_node}",
                *core,
            ]
        else:
            cmd = ["python", *core]

        commands.append((reward_system, run_name, cmd))

    return commands


def main():
    parser = argparse.ArgumentParser(description="Part 4 standardized RL runner")
    parser.add_argument(
        "--owner",
        type=str,
        required=True,
        help="Owner tag used in run names, e.g. J or K",
    )
    parser.add_argument("--run-prefix", type=str, default="rl-gsm8k-part4-teammate")
    parser.add_argument(
        "--reward-systems",
        type=str,
        default="baseline,numeric_distance,calc_consistency",
    )
    parser.add_argument(
        "--launcher", type=str, default="python", choices=["python", "torchrun"]
    )
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--model-tag", type=str, default="sft-teammate")
    parser.add_argument("--model-step", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--device-batch-size", type=int, default=8)
    parser.add_argument("--examples-per-step", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=60)
    parser.add_argument("--eval-examples", type=int, default=400)
    parser.add_argument("--save-every", type=int, default=60)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    commands = build_commands(args)
    for reward_system, run_name, cmd in commands:
        print(f"\n[{reward_system}] run={run_name}")
        print(" ".join(shlex.quote(part) for part in cmd))

    if args.print_only:
        return

    for reward_system, run_name, cmd in commands:
        print(f"\n=== Starting {reward_system}: {run_name} ===")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
