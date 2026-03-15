# D20 Pipeline Instructions

Runs the full pipeline: download → SFT → push → RL → push.
Base model: `sdobson/nanochat` (d20, 65K vocab, step 650).

---

## Prerequisites

You need access to:
- This GitHub repo (clone it)
- The shared Modal workspace (`nanochat-ablation` app, `nanochat-vol` volume)
- A HuggingFace repo to push results to (create one at huggingface.co)

Install dependencies:
```bash
pip install uv
uv sync
```

Authenticate:
```bash
uv run modal setup          # follow browser login
uv run wandb login          # paste your W&B API key
```

---

## Step 1 — Set your HuggingFace destination repo

Open `nanochat_modal.py` and find the `D20_HF_PUSH_REPO` constant (around line 1220):

```python
D20_HF_PUSH_REPO = "YOUR_HF_USERNAME/nanochat-d20-finetuned"
```

Replace with your actual HuggingFace repo (it will be created automatically if it doesn't exist).

---

## Step 2 — Launch the pipeline (detached)

```bash
uv run modal run --detach nanochat_modal.py::run_d20_pipeline
```

This runs entirely on Modal servers — safe to close your terminal. It will:
1. Download `sdobson/nanochat` base checkpoint + tokenizer to the Modal volume
2. SFT on 4x H100 (~6–8h)
3. Push SFT checkpoint to `YOUR_HF_REPO/sft-nanochat-d20/`
4. RL (GRPO on GSM8K) on 4x H100 (~1h)
5. Push RL checkpoint to `YOUR_HF_REPO/rl-gsm8k-nanochat-d20/`

**Monitor progress:**
- W&B: https://wandb.ai — project `nanochat-sft` (SFT) and `nanochat-rl` (RL)
- Modal: https://modal.com/apps

---

## Re-running individual stages

If a stage fails, you can re-run it standalone without restarting the whole pipeline:

```bash
# Re-run just the download (idempotent — skips already-cached files)
uv run modal run nanochat_modal.py::stage_download_base_checkpoint

# Re-run just SFT
uv run modal run nanochat_modal.py::stage_sft_d20

# Push SFT checkpoint to HuggingFace
uv run modal run nanochat_modal.py::stage_push_checkpoint_to_hf \
  --source sft \
  --model-tag nanochat-d20 \
  --hf-repo YOUR_HF_REPO \
  --hf-folder sft-nanochat-d20 \
  --source-base-dir /vol/nanochat_d20_cache

# Re-run just RL
uv run modal run nanochat_modal.py::stage_rl_d20

# Push RL checkpoint to HuggingFace
uv run modal run nanochat_modal.py::stage_push_checkpoint_to_hf \
  --source rl \
  --model-tag nanochat-d20 \
  --hf-repo YOUR_HF_REPO \
  --hf-folder rl-gsm8k-nanochat-d20 \
  --source-base-dir /vol/nanochat_d20_cache
```

Check what checkpoints are saved on the volume:
```bash
uv run modal run nanochat_modal.py::ls_checkpoints
```

---

## What gets logged

### SFT (W&B project: `nanochat-sft`)
| Metric | Description |
|---|---|
| `val/bpb` | Validation bits-per-byte (lower = better) |
| `train/loss` | Training loss |
| `lrm` | Learning rate multiplier |

Checkpoints saved every 500 steps to `/vol/nanochat_d20_cache/chatsft_checkpoints/nanochat-d20/`.

### RL (W&B project: `nanochat-rl`)
| Metric | Description |
|---|---|
| `reward` | Mean reward per step (fraction of correct answers) |
| `pass@1` … `pass@8` | Fraction of GSM8K problems solved with k attempts |
| `sequence_length` | Mean generated sequence length |
| `examples` | W&B Table of sample questions + model outputs |

Checkpoints saved every 60 steps to `/vol/nanochat_d20_cache/chatrl_checkpoints/nanochat-d20/`.

---

## Troubleshooting

**`ModuleNotFoundError: huggingface_hub`** — Image needs a rebuild. Just re-run, Modal will rebuild automatically.

**`No checkpoints found`** — The push stage can't find `.pt` files. Run `ls_checkpoints` to verify they exist on the volume.

**OOM during SFT** — The d20 model uses 65K vocab. `device-batch-size=8` is already set; if still OOM try reducing further in `stage_sft_d20`.

**Job disappeared from Modal** — Timeout or preemption. Check https://modal.com/apps for the exit reason. Re-run the individual stage that failed.

**All RL rewards are 0.0** — Model isn't outputting `#### <number>` format. SFT may not have trained long enough or the GSM8K data wasn't included. Check a few examples:
```bash
uv run modal run nanochat_modal.py::peek_eval
```
