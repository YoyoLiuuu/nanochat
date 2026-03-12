"""
A4P2: Full pipeline — pretrain nanochat-baseline (d12) then SFT with baseline
and enhanced data mixtures.

Setup (one-time):
    modal setup
    modal secret create nanochat-secrets \\
        WANDB_API_KEY=<your_key> \\
        HF_TOKEN=hf_<your_token> \\
        WANDB_ENTITY=<your_wandb_username>

Run full pipeline (pretrain + SFT baseline + SFT enhanced + evals):
    modal run --detach sft_modal.py::main

Run individual stages:
    modal run --detach sft_modal.py::run_pretrain
    modal run --detach sft_modal.py::run_baseline_sft
    modal run --detach sft_modal.py::run_enhanced_sft
    modal run --detach sft_modal.py::run_sft_evals
"""

import os
import subprocess
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION
# =============================================================================

# Pretrain config: nanochat-baseline (d12, relu2, RoPE 10K)
DEPTH = 12
MAX_SEQ_LEN = 2048
PRETRAIN_MODEL_TAG = "nanochat-baseline"
NUM_DATA_SHARDS = 150  # ~7B tokens for GPT-2 capability

# SFT config
BASELINE_SFT_TAG = "sft-baseline-d12"
ENHANCED_SFT_TAG = "sft-enhanced-d12"
SFT_DEVICE_BATCH_SIZE = 16

# Enhanced mixture sizes
METAMATHQA_SIZE = 50000
ORCAMATH_SIZE = 50000
ULTRACHAT_SIZE = 100000

# GPU: 4x H200 for pretrain, 1x H200 for SFT/eval
PRETRAIN_GPU = "H100:4"
SFT_GPU = "H100:1"

VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"

PRETRAIN_TIMEOUT_SEC = 60 * 60 * 6
SFT_TIMEOUT_SEC = 60 * 60 * 4
EVAL_TIMEOUT_SEC = 60 * 60 * 2

# =============================================================================
# MODAL PRIMITIVES
# =============================================================================

app = App("nanochat-a4p2")

volume = Volume.from_name("nanochat-vol", create_if_missing=True)
secret = Secret.from_name("nanochat-secrets")

image = (
    ModalImage.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "build-essential", "curl", "wget", "unzip")
    .add_local_dir(
        local_path=".",
        remote_path="/root/nanochat",
        copy=True,
        ignore=[".venv", "__pycache__", "*.pyc", ".git", "rustbpe/target", "runs", "a4p2_checkpoints"],
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
    filepath = os.path.join(NANOCHAT_CACHE, "identity_conversations.jsonl")
    if not os.path.exists(filepath):
        print("Downloading identity_conversations.jsonl...")
        _run(f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")


# =============================================================================
# STAGE 0: DATA + TOKENIZER
# =============================================================================

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    cpu=8, memory=16384, timeout=60 * 60,
)
def stage_data() -> None:
    """Download data shards and train tokenizer."""
    _setup_cache()
    print(f"Downloading {NUM_DATA_SHARDS} data shards...")
    _python("nanochat.dataset", [f"-n {NUM_DATA_SHARDS}"])
    volume.commit()


@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu="A10G:1", timeout=60 * 30,
)
def stage_tokenizer() -> None:
    """Train BPE tokenizer (skip if exists)."""
    _setup_cache()
    tokenizer_path = os.path.join(NANOCHAT_CACHE, "tokenizer.model")
    if os.path.exists(tokenizer_path):
        print("Tokenizer already exists, skipping.")
    else:
        print("Training tokenizer...")
        _python("scripts.tok_train", ["--max-chars=2000000000"])
        volume.commit()


# =============================================================================
# STAGE 1: PRETRAIN nanochat-baseline (d12)
# =============================================================================

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu=PRETRAIN_GPU, timeout=PRETRAIN_TIMEOUT_SEC,
)
def stage_pretrain() -> None:
    """Pretrain nanochat-baseline: d12, relu2, RoPE 10K."""
    _setup_cache()
    print(f"\n{'='*60}")
    print(f"Pretraining nanochat-baseline (d12)")
    print(f"  GPU: {PRETRAIN_GPU}")
    print(f"{'='*60}\n")

    _python("nanochat.report", ["reset"])

    nproc = int(PRETRAIN_GPU.split(":")[1])
    _torchrun(
        "scripts.base_train",
        [
            f"--depth={DEPTH}",
            f"--max-seq-len={MAX_SEQ_LEN}",
            f"--device-batch-size=16",
            f"--window-pattern=SSSL",
            f"--mlp-type=relu2",
            f"--rope-base=10000",
            f"--model-tag={PRETRAIN_MODEL_TAG}",
            f"--run=a4p2-pretrain-baseline",
            "--save-every=500",
            "--eval-every=250",
            "--core-metric-every=-1",
            "--sample-every=-1",
        ],
        nproc=nproc,
    )
    volume.commit()
    print("Pretraining complete!")


# =============================================================================
# STAGE 2: SFT
# =============================================================================

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu=SFT_GPU, timeout=SFT_TIMEOUT_SEC,
)
def stage_sft(
    run_name: str,
    model_tag: str,
    sft_output_tag: str,
    extra_args: list | None = None,
) -> None:
    """Run SFT on a pretrained checkpoint."""
    _setup_cache()
    _download_identity_conversations()
    print(f"\n{'='*60}")
    print(f"SFT: {run_name}")
    print(f"  base model: {model_tag}")
    print(f"  output tag: {sft_output_tag}")
    print(f"{'='*60}\n")

    args = [
        f"--model-tag={model_tag}",
        f"--device-batch-size={SFT_DEVICE_BATCH_SIZE}",
        f"--total-batch-size=32768",
        f"--run={run_name}",
        "--chatcore-every=-1",
        "--eval-every=-1",
        "--load-optimizer=0",
    ]
    if extra_args:
        args.extend(extra_args)

    _python("scripts.chat_sft", args)

    # Rename checkpoint dir to output tag
    src = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", model_tag)
    dst = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", sft_output_tag)
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)
        print(f"Renamed: {model_tag} -> {sft_output_tag}")

    volume.commit()
    print(f"Done: {run_name}")


# =============================================================================
# STAGE 3: EVAL
# =============================================================================

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu=SFT_GPU, timeout=EVAL_TIMEOUT_SEC,
)
def stage_chat_eval(run_name: str, model_tag: str) -> None:
    """Run ChatCORE evaluation on an SFT checkpoint."""
    _setup_cache()
    print(f"\n{'='*60}")
    print(f"Chat Eval: {run_name} ({model_tag})")
    print(f"{'='*60}\n")

    _python("scripts.chat_eval", ["-i", "sft", f"--model-tag={model_tag}"])
    volume.commit()
    print(f"Done: {run_name}")


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
    for subdir in ["base_checkpoints", "chatsft_checkpoints"]:
        base_dir = os.path.join(NANOCHAT_CACHE, subdir)
        print(f"\n{subdir}/")
        if not os.path.exists(base_dir):
            print("  (not found)")
            continue
        for entry in sorted(os.listdir(base_dir)):
            full = os.path.join(base_dir, entry)
            if os.path.isdir(full):
                files = sorted(os.listdir(full))
                print(f"  {entry}/ ({len(files)} files)")
                for f in files:
                    if f.startswith("meta_") and f.endswith(".json"):
                        with open(os.path.join(full, f)) as mf:
                            meta = _json.load(mf)
                        print(f"    {f}: step={meta.get('step')}, val_bpb={meta.get('val_bpb')}")


# =============================================================================
# ENTRYPOINTS
# =============================================================================

@app.local_entrypoint()
def run_pretrain() -> None:
    """Pretrain nanochat-baseline d12 on 4x H200."""
    stage_data.remote()
    stage_tokenizer.remote()
    stage_pretrain.remote()


@app.local_entrypoint()
def run_baseline_sft() -> None:
    """SFT with original data mixture."""
    stage_sft.remote(
        run_name="sft-baseline-d12",
        model_tag=PRETRAIN_MODEL_TAG,
        sft_output_tag=BASELINE_SFT_TAG,
    )


@app.local_entrypoint()
def run_enhanced_sft() -> None:
    """SFT with enhanced data mixture."""
    stage_sft.remote(
        run_name="sft-enhanced-d12",
        model_tag=PRETRAIN_MODEL_TAG,
        sft_output_tag=ENHANCED_SFT_TAG,
        extra_args=[
            f"--metamathqa-size={METAMATHQA_SIZE}",
            f"--orcamath-size={ORCAMATH_SIZE}",
            f"--ultrachat-size={ULTRACHAT_SIZE}",
        ],
    )


@app.local_entrypoint()
def run_eval_baseline() -> None:
    """Evaluate baseline SFT checkpoint only."""
    stage_chat_eval.remote(run_name="eval-sft-baseline-d12", model_tag=BASELINE_SFT_TAG)


@app.local_entrypoint()
def run_eval_enhanced() -> None:
    """Evaluate enhanced SFT checkpoint only."""
    stage_chat_eval.remote(run_name="eval-sft-enhanced-d12", model_tag=ENHANCED_SFT_TAG)


@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    cpu=1, timeout=60 * 60 * 10,  # 10h orchestrator
)
def run_pipeline() -> None:
    """
    Full pipeline running entirely on Modal servers (survives laptop disconnect).
    Pretrain d12 → SFT baseline → SFT enhanced → evals.
    """
    _setup_cache()
    print("\n" + "="*60)
    print("A4P2: Pretrain + SFT Pipeline (server-side)")
    print("="*60 + "\n")

    print("[1/5] Data + tokenizer...")
    stage_data.remote()
    stage_tokenizer.remote()

    print("[2/5] Pretraining nanochat-baseline (d12, 4x H100)...")
    stage_pretrain.remote()

    print("[3/5] SFT baseline...")
    stage_sft.remote(
        run_name="sft-baseline-d12",
        model_tag=PRETRAIN_MODEL_TAG,
        sft_output_tag=BASELINE_SFT_TAG,
    )

    print("[4/5] SFT enhanced...")
    stage_sft.remote(
        run_name="sft-enhanced-d12",
        model_tag=PRETRAIN_MODEL_TAG,
        sft_output_tag=ENHANCED_SFT_TAG,
        extra_args=[
            f"--metamathqa-size={METAMATHQA_SIZE}",
            f"--orcamath-size={ORCAMATH_SIZE}",
            f"--ultrachat-size={ULTRACHAT_SIZE}",
        ],
    )

    print("[5/5] Evaluating both...")
    stage_chat_eval.remote(run_name="eval-sft-baseline-d12", model_tag=BASELINE_SFT_TAG)
    stage_chat_eval.remote(run_name="eval-sft-enhanced-d12", model_tag=ENHANCED_SFT_TAG)

    print("\nAll done! Check W&B: https://wandb.ai/alvinay73-university-of-toronto/nanochat-sft")


@app.local_entrypoint()
def main() -> None:
    """Submit pipeline to Modal servers. Safe to close laptop immediately."""
    print("Running full pipeline on Modal servers...")
    run_pipeline.remote()
    print("Pipeline complete!")
    print("  W&B: https://wandb.ai/alvinay73-university-of-toronto/nanochat-sft")
    print("  Modal: https://modal.com/apps")
    print("  Check status: uv run modal container list")
