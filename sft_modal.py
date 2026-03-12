"""
SFT data mixture optimization: baseline vs enhanced mixture.

Runs SFT on a d12 pretrained checkpoint (nanochat-swiglu-rope500k-seed0)
with (a) the original data mixture and (b) an enhanced mixture adding
MetaMathQA, Orca-Math, and UltraChat-200k.

Setup (one-time):
    modal setup
    modal secret create nanochat-secrets \\
        WANDB_API_KEY=<your_key> \\
        HF_TOKEN=hf_<your_token> \\
        WANDB_ENTITY=<your_wandb_username>

Run full pipeline (baseline SFT + enhanced SFT + evals):
    modal run sft_modal.py

Run individual stages:
    modal run sft_modal.py::run_baseline_sft
    modal run sft_modal.py::run_enhanced_sft
    modal run sft_modal.py::run_sft_evals
"""

import os
import subprocess
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION
# =============================================================================

# Pretrained base model on volume (picochat d8, step 5000 from ctx extension)
BASE_MODEL_TAG = "picochat-ctx-s1"
BASE_MODEL_STEP = 5000

# SFT output tags
BASELINE_SFT_TAG = "sft-baseline"
ENHANCED_SFT_TAG = "sft-enhanced"

# Enhanced mixture sizes
METAMATHQA_SIZE = 50000
ORCAMATH_SIZE = 50000
ULTRACHAT_SIZE = 100000

GPU = "H100:1"
DEVICE_BATCH_SIZE = 4  # seq_len=2048 with total_batch_size=16384 → batch*seq must divide total
TOTAL_BATCH_SIZE = 16384

VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"

TRAIN_TIMEOUT_SEC = 60 * 60 * 6   # 6h max
EVAL_TIMEOUT_SEC = 60 * 60 * 2    # 2h max

# =============================================================================
# MODAL PRIMITIVES
# =============================================================================

app = App("nanochat-sft-experiment")

volume = Volume.from_name("nanochat-vol", create_if_missing=True)
secret = Secret.from_name("nanochat-secrets")

image = (
    ModalImage.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "build-essential", "curl", "wget", "unzip")
    .add_local_dir(
        local_path=".",
        remote_path="/root/nanochat",
        copy=True,
        ignore=[".venv", "__pycache__", "*.pyc", ".git", "rustbpe/target", "runs"],
    )
    .workdir("/root/nanochat")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> $HOME/.bashrc",
    )
    .pip_install("uv")
    .run_commands("uv sync --extra gpu --no-install-project")
    .env({
        "OMP_NUM_THREADS": "1",
        "NANOCHAT_BASE_DIR": NANOCHAT_CACHE,
        "HF_HOME": f"{VOLUME_MOUNT}/hf_cache",
    })
)

# =============================================================================
# HELPERS
# =============================================================================

def _python(module: str, args: list | None = None) -> None:
    args = args or []
    cmd = f"cd /root/nanochat && uv run python -m {module} {' '.join(args)}"
    _run(cmd)


def _torchrun(module: str, args: list | None = None, *, nproc: int = 1) -> None:
    args = args or []
    args_str = (" -- " + " ".join(args)) if args else ""
    cmd = (
        f"cd /root/nanochat && "
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m {module}{args_str}"
    )
    _run(cmd)


def _run(cmd: str) -> None:
    print(f"\n>>>  {cmd}\n", flush=True)
    result = subprocess.run(["bash", "-c", cmd], check=False,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}):\n  {cmd}\n{result.stdout[-2000:] if result.stdout else ''}")


def _setup_cache() -> None:
    os.makedirs(NANOCHAT_CACHE, exist_ok=True)


def _download_identity_conversations() -> None:
    """Download identity conversations JSONL if not already present."""
    filepath = os.path.join(NANOCHAT_CACHE, "identity_conversations.jsonl")
    if not os.path.exists(filepath):
        print("Downloading identity_conversations.jsonl...")
        _run(f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")
    else:
        print("identity_conversations.jsonl already exists.")


# =============================================================================
# DIAGNOSTICS
# =============================================================================

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    cpu=1, timeout=60 * 5,
)
def list_checkpoints() -> None:
    """List all checkpoints on the volume."""
    _setup_cache()
    import json as _json
    import shutil as _shutil

    # Move SFT checkpoints into a4p2_checkpoints if they exist
    sft_dir = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints")
    a4p2_dir = os.path.join(NANOCHAT_CACHE, "a4p2_checkpoints")
    if os.path.exists(sft_dir):
        for tag in ["sft-baseline", "sft-enhanced"]:
            src = os.path.join(sft_dir, tag)
            dst = os.path.join(a4p2_dir, tag)
            if os.path.exists(src) and not os.path.exists(dst):
                os.makedirs(a4p2_dir, exist_ok=True)
                _shutil.move(src, dst)
                print(f"Moved {src} -> {dst}")
        # Clean up lora checkpoints
        for tag in ["sft-lora-baseline", "sft-lora-enhanced"]:
            lora_path = os.path.join(sft_dir, tag)
            if os.path.exists(lora_path):
                _shutil.rmtree(lora_path)
                print(f"Deleted {lora_path}")
        volume.commit()

    for subdir in ["base_checkpoints", "chatsft_checkpoints", "a4p2_checkpoints"]:
        base_dir = os.path.join(NANOCHAT_CACHE, subdir)
        print(f"\n{subdir}/")
        if not os.path.exists(base_dir):
            print("  (not found)")
            continue
        for entry in sorted(os.listdir(base_dir)):
            full = os.path.join(base_dir, entry)
            if os.path.isdir(full):
                files = sorted(os.listdir(full))
                print(f"  {entry}/ ({len(files)} files): {files[:5]}")
                # Print metadata from meta files
                for f in files:
                    if f.startswith("meta_") and f.endswith(".json"):
                        meta_path = os.path.join(full, f)
                        with open(meta_path) as mf:
                            meta = _json.load(mf)
                        print(f"    {f}: step={meta.get('step')}, val_bpb={meta.get('val_bpb')}")
            else:
                print(f"  {entry}")


# =============================================================================
# SFT STAGES
# =============================================================================

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu=GPU, timeout=TRAIN_TIMEOUT_SEC,
)
def stage_sft(
    run_name: str,
    model_tag: str,
    sft_output_tag: str,
    model_step: int | None = None,
    extra_args: list | None = None,
) -> None:
    """Run SFT on a pretrained checkpoint."""
    _setup_cache()
    _download_identity_conversations()
    print(f"\n{'='*60}")
    print(f"SFT: {run_name}")
    print(f"  base model: {model_tag} step={model_step}")
    print(f"  output tag: {sft_output_tag}")
    print(f"{'='*60}\n")

    args = [
        f"--model-tag={model_tag}",
        f"--device-batch-size={DEVICE_BATCH_SIZE}",
        f"--total-batch-size={TOTAL_BATCH_SIZE}",
        f"--run={run_name}",
        f"--chatcore-every=-1",
        f"--eval-every=-1",
    ]
    if model_step is not None:
        args.append(f"--model-step={model_step}")
    if extra_args:
        args.extend(extra_args)

    _python("scripts.chat_sft", args)

    # Rename the SFT checkpoint directory to the desired output tag
    src = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", model_tag)
    dst = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", sft_output_tag)
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)
        print(f"Renamed checkpoint: {model_tag} -> {sft_output_tag}")

    volume.commit()
    print(f"Done: {run_name}")


@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu=GPU, timeout=EVAL_TIMEOUT_SEC,
)
def stage_chat_eval(
    run_name: str,
    model_tag: str,
) -> None:
    """Run chat evaluation on an SFT checkpoint."""
    _setup_cache()
    print(f"\n{'='*60}")
    print(f"Chat Eval: {run_name} ({model_tag})")
    print(f"{'='*60}\n")

    _python("scripts.chat_eval", [
        "-i", "sft",
        f"--model-tag={model_tag}",
    ])
    volume.commit()
    print(f"Done: {run_name}")


# =============================================================================
# ENTRYPOINTS
# =============================================================================

@app.local_entrypoint()
def run_baseline_sft() -> None:
    """Run SFT with the original data mixture (baseline)."""
    stage_sft.remote(
        run_name="sft-baseline",
        model_tag=BASE_MODEL_TAG,
        sft_output_tag=BASELINE_SFT_TAG,
        model_step=BASE_MODEL_STEP,
    )


@app.local_entrypoint()
def run_enhanced_sft() -> None:
    """Run SFT with the enhanced data mixture (original + MetaMathQA + Orca-Math + UltraChat)."""
    stage_sft.remote(
        run_name="sft-enhanced",
        model_tag=BASE_MODEL_TAG,
        sft_output_tag=ENHANCED_SFT_TAG,
        model_step=BASE_MODEL_STEP,
        extra_args=[
            f"--metamathqa-size={METAMATHQA_SIZE}",
            f"--orcamath-size={ORCAMATH_SIZE}",
            f"--ultrachat-size={ULTRACHAT_SIZE}",
        ],
    )


@app.local_entrypoint()
def run_sft_evals() -> None:
    """Evaluate both SFT checkpoints."""
    stage_chat_eval.remote(
        run_name="eval-sft-baseline",
        model_tag=BASELINE_SFT_TAG,
    )
    stage_chat_eval.remote(
        run_name="eval-sft-enhanced",
        model_tag=ENHANCED_SFT_TAG,
    )


@app.local_entrypoint()
def main() -> None:
    """
    Launch both SFT runs in parallel. Use with: modal run --detach sft_modal.py::main
    """
    print("Spawning 2 SFT runs in parallel on H100s...")

    h1 = stage_sft.spawn(
        run_name="sft-baseline",
        model_tag=BASE_MODEL_TAG,
        sft_output_tag=BASELINE_SFT_TAG,
        model_step=BASE_MODEL_STEP,
    )
    print("  [1/2] Baseline FFT spawned")

    h2 = stage_sft.spawn(
        run_name="sft-enhanced",
        model_tag=BASE_MODEL_TAG,
        sft_output_tag=ENHANCED_SFT_TAG,
        model_step=BASE_MODEL_STEP,
        extra_args=[
            f"--metamathqa-size={METAMATHQA_SIZE}",
            f"--orcamath-size={ORCAMATH_SIZE}",
            f"--ultrachat-size={ULTRACHAT_SIZE}",
        ],
    )
    print("  [2/2] Enhanced FFT spawned")

    for i, h in enumerate([h1, h2], 1):
        h.get()
        print(f"  [{i}/2] complete")

    print("\nAll training done! Running evals...")
    for tag in [BASELINE_SFT_TAG, ENHANCED_SFT_TAG]:
        stage_chat_eval.remote(run_name=f"eval-{tag}", model_tag=tag)
        print(f"  Eval {tag} complete")

    print("\nAll done! Check W&B: https://wandb.ai/alvinay73-university-of-toronto/nanochat-sft")
