"""
A4P2: Replicate Karpathy's exact nanochat pipeline (relu2 + RoPE 10K, d24).
Then run enhanced SFT with MetaMathQA + Orca-Math + UltraChat.

Run: modal run --detach sft_modal.py::main
"""

import os
import subprocess
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION — Karpathy's exact speedrun.sh config
# =============================================================================

DEPTH = 24
TARGET_PARAM_DATA_RATIO = 9.5
NPROC = 8
DEVICE_BATCH_SIZE = 16
GPU = "H100:8"
NUM_DATA_SHARDS = 170

PRETRAIN_MODEL_TAG = "nanochat-d24-original"
BASELINE_SFT_TAG = "sft-baseline-original"
ENHANCED_SFT_TAG = "sft-enhanced-original"
HF_REPO = "alvina-yang/a4_original"

METAMATHQA_SIZE = 50000
ORCAMATH_SIZE = 50000
ULTRACHAT_SIZE = 100000

VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"

# =============================================================================
# MODAL PRIMITIVES
# =============================================================================

app = App("nanochat-a4p2-original")
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

def _python(module, args=None):
    args = args or []
    cmd = f"cd /root/nanochat && uv run python -m {module} {' '.join(args)}"
    _run(cmd)

def _torchrun(module, args=None, *, nproc=NPROC):
    args = args or []
    args_str = (" -- " + " ".join(args)) if args else ""
    cmd = f"cd /root/nanochat && uv run torchrun --standalone --nproc_per_node={nproc} -m {module}{args_str}"
    _run(cmd)

def _run(cmd):
    print(f"\n>>>  {cmd}\n", flush=True)
    result = subprocess.run(["bash", "-c", cmd], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}):\n  {cmd}")

def _setup_cache():
    os.makedirs(NANOCHAT_CACHE, exist_ok=True)

def _download_identity_conversations():
    filepath = os.path.join(NANOCHAT_CACHE, "identity_conversations.jsonl")
    if not os.path.exists(filepath):
        _run(f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")

def _upload_to_hf():
    _run(f"""cd /root/nanochat && uv run python -c "
import os
from huggingface_hub import HfApi, create_repo
api = HfApi()
repo_id = '{HF_REPO}'
create_repo(repo_id, repo_type='model', exist_ok=True)
cache = '{NANOCHAT_CACHE}'
for subdir, tags in [('base_checkpoints', ['{PRETRAIN_MODEL_TAG}']), ('chatsft_checkpoints', ['{BASELINE_SFT_TAG}', '{ENHANCED_SFT_TAG}'])]:
    base = os.path.join(cache, subdir)
    if not os.path.exists(base): continue
    for tag in tags:
        tag_dir = os.path.join(base, tag)
        if not os.path.exists(tag_dir): continue
        for f in sorted(os.listdir(tag_dir)):
            if not (f.endswith('.pt') or f.endswith('.json')): continue
            if 'optim_' in f: continue  # skip optimizer files to save space
            filepath = os.path.join(tag_dir, f)
            path_in_repo = f'{{tag}}/{{f}}'
            size_mb = os.path.getsize(filepath) / 1e6
            print(f'Uploading {{path_in_repo}} ({{size_mb:.0f}} MB)...')
            api.upload_file(path_or_fileobj=filepath, path_in_repo=path_in_repo, repo_id=repo_id)
print('Upload complete: https://huggingface.co/{HF_REPO}')
"
""")

# =============================================================================
# FULL PIPELINE — single 4x H100 container
# =============================================================================

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu=GPU, timeout=60 * 60 * 6,
)
def run_pipeline() -> None:
    """Download pretrained from HF → SFT baseline → upload → SFT enhanced → upload → evals."""
    _setup_cache()
    print("\n" + "="*60)
    print("A4P2: Karpathy's EXACT config (relu2 + RoPE 10K, d24)")
    print("="*60 + "\n")

    # [1] Data
    print("[1/8] Data...")
    _python("nanochat.dataset", [f"-n {NUM_DATA_SHARDS}"])
    volume.commit()

    # [2] Tokenizer
    print("[2/8] Tokenizer...")
    tokenizer_path = os.path.join(NANOCHAT_CACHE, "tokenizer.model")
    if not os.path.exists(tokenizer_path):
        _python("scripts.tok_train", ["--max-chars=2000000000"])
        volume.commit()
    else:
        print("Tokenizer exists, skipping.")

    # [3] Pretrain d24 relu2+RoPE10K (NO FP8 — avoids dtype mismatch in SFT)
    print("[3/8] Pretraining d24 (relu2 + RoPE 10K, NO FP8)...")
    pretrain_dir = os.path.join(NANOCHAT_CACHE, "base_checkpoints", PRETRAIN_MODEL_TAG)
    if os.path.exists(pretrain_dir) and any(f.startswith("meta_") for f in os.listdir(pretrain_dir)):
        print("  Pretrained model exists, skipping.")
    else:
        _python("nanochat.report", ["reset"])
        _torchrun("scripts.base_train", [
            f"--depth={DEPTH}",
            f"--target-param-data-ratio={TARGET_PARAM_DATA_RATIO}",
            f"--device-batch-size={DEVICE_BATCH_SIZE}",
            "--mlp-type=relu2", "--rope-base=10000", "--fp8",
            f"--model-tag={PRETRAIN_MODEL_TAG}",
            "--run=a4p2-pretrain-d24-original",
            "--save-every=500", "--eval-every=250",
            "--core-metric-every=-1", "--sample-every=-1",
        ], nproc=NPROC)
    volume.commit()

    # [4] SFT baseline
    print("[4/8] SFT baseline (Karpathy's exact mixture)...")
    _download_identity_conversations()
    _torchrun("scripts.chat_sft", [
        f"--model-tag={PRETRAIN_MODEL_TAG}",
        f"--device-batch-size={DEVICE_BATCH_SIZE}",
        "--run=sft-baseline-original",
    ], nproc=NPROC)
    src = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", PRETRAIN_MODEL_TAG)
    dst = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", BASELINE_SFT_TAG)
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)
    volume.commit()

    # [5] Upload baseline to HF
    print("[5/8] Uploading baseline to HuggingFace...")
    _upload_to_hf()

    # [6] SFT enhanced
    print("[6/8] SFT enhanced (+MetaMathQA +OrcaMath +UltraChat)...")
    _torchrun("scripts.chat_sft", [
        f"--model-tag={PRETRAIN_MODEL_TAG}",
        f"--device-batch-size={DEVICE_BATCH_SIZE}",
        "--run=sft-enhanced-original",
        f"--metamathqa-size={METAMATHQA_SIZE}",
        f"--orcamath-size={ORCAMATH_SIZE}",
        f"--ultrachat-size={ULTRACHAT_SIZE}",
    ], nproc=NPROC)
    src = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", PRETRAIN_MODEL_TAG)
    dst = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", ENHANCED_SFT_TAG)
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)
    volume.commit()

    # [7] Upload enhanced to HF
    print("[7/8] Uploading enhanced to HuggingFace...")
    _upload_to_hf()

    # [8] Evals (3 tasks only)
    print("[8/8] Evaluating both (GSM8K + SpellingBee + ARC-Easy)...")
    _torchrun("scripts.chat_eval", [
        "-i", "sft", f"--model-tag={BASELINE_SFT_TAG}",
        "-a", "GSM8K|SpellingBee|ARC-Easy",
    ], nproc=NPROC)
    _torchrun("scripts.chat_eval", [
        "-i", "sft", f"--model-tag={ENHANCED_SFT_TAG}",
        "-a", "GSM8K|SpellingBee|ARC-Easy",
    ], nproc=NPROC)
    volume.commit()

    print("\nAll done!")
    print(f"  HF: https://huggingface.co/{HF_REPO}")

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu=GPU, timeout=60 * 60 * 4,
)
def run_remaining() -> None:
    """SFT enhanced + evals only (pretrain + baseline already on volume)."""
    _setup_cache()
    _download_identity_conversations()

    # SFT enhanced
    print("[1/3] SFT enhanced (+MetaMathQA +OrcaMath +UltraChat)...")
    _torchrun("scripts.chat_sft", [
        f"--model-tag={PRETRAIN_MODEL_TAG}",
        f"--device-batch-size={DEVICE_BATCH_SIZE}",
        "--run=sft-enhanced-original",
        f"--metamathqa-size={METAMATHQA_SIZE}",
        f"--orcamath-size={ORCAMATH_SIZE}",
        f"--ultrachat-size={ULTRACHAT_SIZE}",
    ], nproc=NPROC)
    src = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", PRETRAIN_MODEL_TAG)
    dst = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", ENHANCED_SFT_TAG)
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)
    volume.commit()

    # Evals
    print("[2/3] Eval baseline (GSM8K + SpellingBee + ARC-Easy)...")
    _torchrun("scripts.chat_eval", [
        "-i", "sft", f"--model-tag={BASELINE_SFT_TAG}",
        "-a", "GSM8K|SpellingBee|ARC-Easy",
    ], nproc=NPROC)

    print("[3/3] Eval enhanced (GSM8K + SpellingBee + ARC-Easy)...")
    _torchrun("scripts.chat_eval", [
        "-i", "sft", f"--model-tag={ENHANCED_SFT_TAG}",
        "-a", "GSM8K|SpellingBee|ARC-Easy",
    ], nproc=NPROC)
    volume.commit()

    print("\nAll done!")

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu=GPU, timeout=60 * 60 * 3,
)
def run_evals() -> None:
    """Eval both SFT checkpoints only."""
    _setup_cache()
    print("[1/2] Eval baseline (GSM8K + SpellingBee + ARC-Easy)...")
    _python("scripts.chat_eval", [
        "-i", "sft", f"--model-tag={BASELINE_SFT_TAG}",
        "-a", "GSM8K|SpellingBee|ARC-Easy",
    ])
    print("[2/2] Eval enhanced (GSM8K + SpellingBee + ARC-Easy)...")
    _python("scripts.chat_eval", [
        "-i", "sft", f"--model-tag={ENHANCED_SFT_TAG}",
        "-a", "GSM8K|SpellingBee|ARC-Easy",
    ])
    volume.commit()
    print("All evals done!")

@app.local_entrypoint()
def main() -> None:
    """Launch evals. Use --detach to survive laptop close."""
    print("Launching evals on 8x H100...")
    run_evals.remote()
    print("Evals complete!")
