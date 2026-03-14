# Analysis Notebook — Setup Instructions

Hi! This is the setup guide for running `analysis.ipynb`, which generates all the RL training analysis plots and commentary for the report.

---

## Prerequisites

You need:
- Access to the Modal volume `nanochat-vol` (app `nanochat-ablation`)
- Python 3.10+ with the repo's venv

Install dependencies if you haven't already:
```bash
pip install uv
uv sync
uv run modal setup   # follow browser login if prompted
```

Install notebook dependencies (if not already in the venv):
```bash
uv run pip install notebook scikit-learn seaborn
```

---

## Step 1 — Download eval logs from Modal

Run this from the repo root:
```bash
uv run modal run nanochat_modal.py::download_all_d20_eval_logs
```

This downloads eval logs for all four reward-system runs to `./eval_logs/`:
```
eval_logs/
  rl-gsm8k-nanochat-d20/               ← baseline
  rl-gsm8k-nanochat-d20-numeric_distance/
  rl-gsm8k-nanochat-d20-completion_brevity/
  rl-gsm8k-nanochat-d20-combined/      ← if it exists
```

Each folder contains `eval_step_XXXXXX.json` files — one per evaluation checkpoint.

If a run folder is missing or empty, the notebook will skip it gracefully and print a warning.

---

## Step 2 — Add Karpathy reference curve values

Open `analysis.ipynb` and find **cell 4** (titled "Karpathy digitised reference curves"). It looks like:

```python
KARPATHY_STEPS  = [0, 60, 120, ...]
KARPATHY_PASS1  = [0.04, 0.08, ...]
KARPATHY_PASS8  = [0.12, 0.22, ...]
```

Replace the placeholder values with the actual digitised pass@1 and pass@8 values from the Karpathy nanochat-d32 RL training figure. Steps should match the x-axis of that figure.

---

## Step 3 — Run the notebook

```bash
uv run jupyter notebook analysis.ipynb
```

Run all cells top to bottom (**Kernel → Restart & Run All**).

The notebook will generate and save these plots to the repo root:
- `passk_1_curve.png` — pass@1 over training steps (all runs + Karpathy reference)
- `passk_8_curve.png` — pass@8 over training steps
- `problem_type_dist.png` — distribution of the 7 problem categories
- `error_x_problem_heatmap.png` — error type × problem type heatmap (one panel per run)
- `cluster_<run_name>.png` — TF-IDF K-Means PCA scatter of incorrect problems
- `difficulty_length.png` — pass@1 accuracy by question length bin
- `length_vs_correctness_scatter.png` — scatter of length vs correctness
- `error_evolution.png` — stacked bar of error types over training steps
- `accuracy_by_problem_type.png` — grouped bar comparing reward systems per problem type

Section 10 contains the written commentary — no interaction needed, it's already written.

---

## Troubleshooting

**`ModuleNotFoundError: sklearn` / `seaborn`** — run `uv run pip install scikit-learn seaborn notebook` and restart the kernel.

**"No eval logs found"** — the `./eval_logs/` folder is missing or empty. Re-run Step 1.

**A run shows 0 eval snapshots** — that run either didn't finish or used a different run name. Check `ls eval_logs/` and compare against the `RUNS` dict in cell 1 of the notebook.

**Karpathy reference line looks wrong** — the placeholder values in cell 4 are approximate. Replace them with the actual digitised values (Step 2 above).
