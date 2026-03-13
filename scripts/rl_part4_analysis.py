"""
Compare Part 4 RL runs from chatrl_eval_logs JSON files.

This script summarizes:
1) pass@k curves over steps
2) final-step mistake type distribution
3) final-step summary table per run

Usage:
    python -m scripts.rl_part4_analysis \
    --runs rl-gsm8k-part4-teammate-j-baseline,rl-gsm8k-part4-teammate-j-numeric_distance,rl-gsm8k-part4-teammate-j-calc_consistency
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def classify_error_type(generated_text: str, reference: str) -> str:
    gen = (generated_text or "").strip()
    if "####" not in gen:
        return "missing_marker"

    pred_match = re.search(r"####\s*([\-0-9\.,]+)", gen)
    ref_match = re.search(r"####\s*([\-0-9\.,]+)", reference or "")

    if pred_match is None:
        return "malformed_marker"

    pred = pred_match.group(1).replace(",", "")
    ref = ref_match.group(1).replace(",", "") if ref_match else None

    try:
        pred_val = float(pred)
        ref_val = float(ref) if ref is not None else None
    except Exception:
        return "non_numeric_answer"

    if ref_val is not None and abs(pred_val - ref_val) <= 1.0:
        return "off_by_one_or_rounding"

    if "<<" in gen and ">>" in gen:
        return "calc_reasoning_error"
    return "wrong_final_answer"


def load_run_logs(base_dir: str, run_name: str):
    run_dir = os.path.join(base_dir, "chatrl_eval_logs", run_name)
    files = sorted(glob.glob(os.path.join(run_dir, "eval_step_*.json")))
    if not files:
        raise FileNotFoundError(f"No eval logs found for run '{run_name}' in {run_dir}")

    passk_rows = []
    outcome_rows = []
    for path in files:
        with open(path, "r") as f:
            data = json.load(f)
        step = data["step"]
        passk = data.get("pass@k", {})
        passk_rows.append({"run": run_name, "step": step, **passk})
        for rec in data.get("records", []):
            for outcome in rec.get("outcomes", []):
                outcome_rows.append(
                    {
                        "run": run_name,
                        "step": step,
                        "idx": rec.get("idx"),
                        "question": rec.get("question", ""),
                        "reference": rec.get("reference", ""),
                        "generated_text": outcome.get("generated_text", ""),
                        "is_correct": bool(outcome.get("is_correct", False)),
                    }
                )
    return pd.DataFrame(passk_rows), pd.DataFrame(outcome_rows)


def plot_passk(passk_df: pd.DataFrame, out_dir: str, columns: Iterable[str]):
    os.makedirs(out_dir, exist_ok=True)
    for column in columns:
        if column not in passk_df.columns:
            continue
        plt.figure(figsize=(8, 4.5))
        for run_name, group in passk_df.groupby("run"):
            group = group.sort_values("step")
            plt.plot(group["step"], group[column], marker="o", label=run_name)
        plt.title(f"{column} over training")
        plt.xlabel("step")
        plt.ylabel(column)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{column.replace('@', 'at')}_curve.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()


def build_final_summary(passk_df: pd.DataFrame, outcomes_df: pd.DataFrame):
    rows = []
    error_rows = []
    for run_name in sorted(passk_df["run"].unique()):
        passk_run = passk_df[passk_df["run"] == run_name].sort_values("step")
        final_step = int(passk_run["step"].max())
        final_passk = passk_run[passk_run["step"] == final_step].iloc[0].to_dict()

        out_run = outcomes_df[
            (outcomes_df["run"] == run_name) & (outcomes_df["step"] == final_step)
        ].copy()
        if len(out_run) > 0:
            accuracy = out_run["is_correct"].mean()
            incorrect = out_run[~out_run["is_correct"]].copy()
            if len(incorrect) > 0:
                incorrect["error_type"] = incorrect.apply(
                    lambda row: classify_error_type(
                        row["generated_text"], row["reference"]
                    ),
                    axis=1,
                )
                for error_type, count in incorrect["error_type"].value_counts().items():
                    error_rows.append(
                        {
                            "run": run_name,
                            "final_step": final_step,
                            "error_type": error_type,
                            "count": int(count),
                            "fraction": float(count / len(incorrect)),
                        }
                    )
        else:
            accuracy = float("nan")

        summary = {
            "run": run_name,
            "final_step": final_step,
            "sample_accuracy_final": accuracy,
        }
        for key, value in final_passk.items():
            if key.startswith("pass@"):
                summary[key] = value
        rows.append(summary)

    return pd.DataFrame(rows), pd.DataFrame(error_rows)


def plot_error_types(error_df: pd.DataFrame, out_dir: str):
    if error_df.empty:
        return
    os.makedirs(out_dir, exist_ok=True)
    pivot = error_df.pivot(index="error_type", columns="run", values="fraction").fillna(
        0.0
    )
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.title("Final-step error type fractions")
    plt.ylabel("fraction")
    plt.xlabel("error_type")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "error_type_fractions.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Part 4 RL comparison from eval logs")
    parser.add_argument(
        "--runs", type=str, required=True, help="Comma-separated run names"
    )
    parser.add_argument(
        "--base-dir", type=str, default=os.path.expanduser("~/.cache/nanochat")
    )
    parser.add_argument("--out-dir", type=str, default="dev/part4_outputs")
    args = parser.parse_args()

    run_names = [run.strip() for run in args.runs.split(",") if run.strip()]
    passk_frames = []
    outcome_frames = []
    for run_name in run_names:
        passk_df, outcomes_df = load_run_logs(args.base_dir, run_name)
        passk_frames.append(passk_df)
        outcome_frames.append(outcomes_df)

    all_passk = pd.concat(passk_frames, ignore_index=True)
    all_outcomes = pd.concat(outcome_frames, ignore_index=True)

    os.makedirs(args.out_dir, exist_ok=True)
    plot_passk(all_passk, args.out_dir, columns=["pass@1", "pass@4", "pass@8"])

    summary_df, error_df = build_final_summary(all_passk, all_outcomes)
    summary_path = os.path.join(args.out_dir, "summary_table.csv")
    summary_df.to_csv(summary_path, index=False)

    error_path = os.path.join(args.out_dir, "error_type_fractions.csv")
    error_df.to_csv(error_path, index=False)
    plot_error_types(error_df, args.out_dir)

    print("Saved outputs:")
    print(f"- {summary_path}")
    print(f"- {error_path}")
    print(f"- {os.path.join(args.out_dir, 'passat1_curve.png')} (if pass@1 exists)")
    print(f"- {os.path.join(args.out_dir, 'passat8_curve.png')} (if pass@8 exists)")
    print(
        f"- {os.path.join(args.out_dir, 'error_type_fractions.png')} (if errors exist)"
    )


if __name__ == "__main__":
    main()
