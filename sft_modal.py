"""
A4P2: Replicate nanochat speedrun.sh on Modal — pretrain d24, SFT, RL, evals.
Matches Karpathy's exact configuration, then runs enhanced mixture for comparison.

Setup (one-time):
    modal setup
    modal secret create nanochat-secrets \\
        WANDB_API_KEY=<your_key> \\
        HF_TOKEN=hf_<your_token> \\
        WANDB_ENTITY=<your_wandb_username>

Run full pipeline:
    modal run --detach sft_modal.py::main
"""

import os
import subprocess
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION — matches speedrun.sh exactly
# =============================================================================

DEPTH = 24                        # d24 like speedrun.sh
TARGET_PARAM_DATA_RATIO = 9.5     # speedrun.sh uses 9.5 (slightly undertrained)
NPROC = 8                         # 8 GPUs like speedrun.sh
DEVICE_BATCH_SIZE = 16            # speedrun.sh uses 16
GPU = "H100:8"                    # 8x H100

PRETRAIN_MODEL_TAG = "nanochat-d24"
BASELINE_SFT_TAG = "sft-baseline-d24"
ENHANCED_SFT_TAG = "sft-enhanced-d24"

# Enhanced mixture sizes
METAMATHQA_SIZE = 50000
ORCAMATH_SIZE = 50000
ULTRACHAT_SIZE = 100000

# Data
NUM_DATA_SHARDS = 170             # speedrun.sh downloads 170 shards

VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"

PRETRAIN_TIMEOUT_SEC = 60 * 60 * 6
SFT_TIMEOUT_SEC = 60 * 60 * 4
EVAL_TIMEOUT_SEC = 60 * 60 * 3

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


def _torchrun(module: str, args: list | None = None, *, nproc: int = NPROC) -> None:
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
        print(result.stdout[-5000:], flush=True)  # last 5K chars to avoid truncation
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}):\n  {cmd}\n{result.stdout[-2000:] if result.stdout else ''}")


def _setup_cache() -> None:
    os.makedirs(NANOCHAT_CACHE, exist_ok=True)


def _download_identity_conversations() -> None:
    filepath = os.path.join(NANOCHAT_CACHE, "identity_conversations.jsonl")
    if not os.path.exists(filepath):
        _run(f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")


# =============================================================================
# STAGE 0: DATA + TOKENIZER
# =============================================================================

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    cpu=8, memory=16384, timeout=60 * 60,
)
def stage_data() -> None:
    _setup_cache()
    _python("nanochat.dataset", [f"-n {NUM_DATA_SHARDS}"])
    volume.commit()


@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu="A10G:1", timeout=60 * 30,
)
def stage_tokenizer() -> None:
    _setup_cache()
    tokenizer_path = os.path.join(NANOCHAT_CACHE, "tokenizer.model")
    if os.path.exists(tokenizer_path):
        print("Tokenizer exists, skipping.")
    else:
        _python("scripts.tok_train", ["--max-chars=2000000000"])
        volume.commit()


# =============================================================================
# STAGE 1: PRETRAIN d24 — exact match to speedrun.sh
# =============================================================================

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu=GPU, timeout=PRETRAIN_TIMEOUT_SEC,
)
def stage_pretrain() -> None:
    """Pretrain d24 — exact replica of speedrun.sh base_train command."""
    _setup_cache()
    _python("nanochat.report", ["reset"])

    # Exact match: torchrun --nproc=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=9.5 --device-batch-size=16 --run=...
    _torchrun(
        "scripts.base_train",
        [
            f"--depth={DEPTH}",
            f"--target-param-data-ratio={TARGET_PARAM_DATA_RATIO}",
            f"--device-batch-size={DEVICE_BATCH_SIZE}",
            "--fp8",
            f"--model-tag={PRETRAIN_MODEL_TAG}",
            "--run=a4p2-pretrain-d24",
            "--save-every=500",
            "--eval-every=250",
            "--core-metric-every=-1",
            "--sample-every=-1",
        ],
        nproc=NPROC,
    )
    volume.commit()

    # Base eval
    _torchrun("scripts.base_eval", [f"--device-batch-size={DEVICE_BATCH_SIZE}"], nproc=NPROC)
    volume.commit()


# =============================================================================
# STAGE 2: SFT — exact match to speedrun.sh
# =============================================================================

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu=GPU, timeout=SFT_TIMEOUT_SEC,
)
def stage_sft(
    run_name: str,
    model_tag: str,
    sft_output_tag: str,
    extra_args: list | None = None,
) -> None:
    """SFT using torchrun with 8 GPUs — matches speedrun.sh exactly."""
    _setup_cache()
    _download_identity_conversations()

    # Exact match: torchrun --nproc=8 -m scripts.chat_sft -- --device-batch-size=16 --run=...
    args = [
        f"--model-tag={model_tag}",
        f"--device-batch-size={DEVICE_BATCH_SIZE}",
        f"--run={run_name}",
    ]
    if extra_args:
        args.extend(extra_args)

    _torchrun("scripts.chat_sft", args, nproc=NPROC)

    # Rename checkpoint dir
    src = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", model_tag)
    dst = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", sft_output_tag)
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)

    volume.commit()


# =============================================================================
# STAGE 3: EVAL
# =============================================================================

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu=GPU, timeout=EVAL_TIMEOUT_SEC,
)
def stage_chat_eval(run_name: str, model_tag: str) -> None:
    _setup_cache()
    _torchrun("scripts.chat_eval", ["-i", "sft", f"--model-tag={model_tag}"], nproc=NPROC)
    volume.commit()



# =============================================================================
# STAGE 5: UPLOAD TO HUGGINGFACE
# =============================================================================

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    cpu=2, memory=8192, timeout=60 * 60 * 2,
)
def stage_upload_hf() -> None:
    """Upload all d24 checkpoints to HuggingFace."""
    _setup_cache()
    _run("uv pip install huggingface_hub 2>/dev/null || true")

    import json as _json
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    repo_id = "alvina-yang/csc490a4p2"
    create_repo(repo_id, repo_type="model", exist_ok=True)

    # Upload from each checkpoint directory
    for subdir, tags in [
        ("base_checkpoints", [PRETRAIN_MODEL_TAG]),
        ("chatsft_checkpoints", [BASELINE_SFT_TAG, ENHANCED_SFT_TAG]),
        ("chatrl_checkpoints", [BASELINE_SFT_TAG, ENHANCED_SFT_TAG]),
    ]:
        base = os.path.join(NANOCHAT_CACHE, subdir)
        if not os.path.exists(base):
            continue
        for tag in tags:
            tag_dir = os.path.join(base, tag)
            if not os.path.exists(tag_dir):
                continue
            for f in os.listdir(tag_dir):
                filepath = os.path.join(tag_dir, f)
                path_in_repo = f"{tag}/{f}"
                size_mb = os.path.getsize(filepath) / 1e6
                print(f"Uploading {path_in_repo} ({size_mb:.0f} MB)...")
                api.upload_file(path_or_fileobj=filepath, path_in_repo=path_in_repo, repo_id=repo_id)
    print(f"All uploads complete: https://huggingface.co/{repo_id}")
    volume.commit()


# =============================================================================
# DIAGNOSTICS
# =============================================================================

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    cpu=1, timeout=60 * 5,
)
def list_checkpoints() -> None:
    _setup_cache()
    import json as _json
    for subdir in ["base_checkpoints", "chatsft_checkpoints", "chatrl_checkpoints"]:
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
    stage_data.remote()
    stage_tokenizer.remote()
    stage_pretrain.remote()


@app.local_entrypoint()
def run_baseline_sft() -> None:
    stage_sft.remote(run_name="sft-baseline-d24", model_tag=PRETRAIN_MODEL_TAG, sft_output_tag=BASELINE_SFT_TAG)


@app.local_entrypoint()
def run_enhanced_sft() -> None:
    stage_sft.remote(
        run_name="sft-enhanced-d24", model_tag=PRETRAIN_MODEL_TAG, sft_output_tag=ENHANCED_SFT_TAG,
        extra_args=[f"--metamathqa-size={METAMATHQA_SIZE}", f"--orcamath-size={ORCAMATH_SIZE}", f"--ultrachat-size={ULTRACHAT_SIZE}"],
    )


@app.local_entrypoint()
def run_eval_baseline() -> None:
    stage_chat_eval.remote(run_name="eval-sft-baseline-d24", model_tag=BASELINE_SFT_TAG)


@app.local_entrypoint()
def run_eval_enhanced() -> None:
    stage_chat_eval.remote(run_name="eval-sft-enhanced-d24", model_tag=ENHANCED_SFT_TAG)


# =============================================================================
# FULL PIPELINE — runs everything in a single 8x H100 container
# No .remote() calls, so nothing can be killed by disconnect.
# =============================================================================

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu=GPU, timeout=60 * 60 * 12,
)
def run_pipeline() -> None:
    """Full pipeline in one container. Pretrain → SFT → upload → SFT → upload → evals."""
    _setup_cache()
    print("\n" + "="*60)
    print("A4P2: nanochat d24 pipeline (8x H100, single container)")
    print("="*60 + "\n")

    # [1] Data
    print("[1/7] Data...")
    _python("nanochat.dataset", [f"-n {NUM_DATA_SHARDS}"])
    volume.commit()

    # [2] Tokenizer
    print("[2/7] Tokenizer...")
    tokenizer_path = os.path.join(NANOCHAT_CACHE, "tokenizer.model")
    if not os.path.exists(tokenizer_path):
        _python("scripts.tok_train", ["--max-chars=2000000000"])
        volume.commit()
    else:
        print("Tokenizer exists, skipping.")

    # [3] Pretrain d24
    print("[3/7] Pretraining d24...")
    _python("nanochat.report", ["reset"])
    _torchrun("scripts.base_train", [
        f"--depth={DEPTH}", f"--target-param-data-ratio={TARGET_PARAM_DATA_RATIO}",
        f"--device-batch-size={DEVICE_BATCH_SIZE}", "--fp8",
        f"--model-tag={PRETRAIN_MODEL_TAG}", "--run=a4p2-pretrain-d24",
        "--save-every=500", "--eval-every=250", "--core-metric-every=-1", "--sample-every=-1",
    ], nproc=NPROC)
    volume.commit()

    # [4] SFT baseline
    print("[4/7] SFT baseline...")
    _download_identity_conversations()
    _torchrun("scripts.chat_sft", [
        f"--model-tag={PRETRAIN_MODEL_TAG}", f"--device-batch-size={DEVICE_BATCH_SIZE}", "--run=sft-baseline-d24",
    ], nproc=NPROC)
    src = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", PRETRAIN_MODEL_TAG)
    dst = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", BASELINE_SFT_TAG)
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)
    volume.commit()

    # [5] Upload baseline to HF
    print("[5/7] Uploading baseline to HuggingFace...")
    _upload_to_hf()

    # [6] SFT enhanced
    print("[6/7] SFT enhanced...")
    _torchrun("scripts.chat_sft", [
        f"--model-tag={PRETRAIN_MODEL_TAG}", f"--device-batch-size={DEVICE_BATCH_SIZE}", "--run=sft-enhanced-d24",
        f"--metamathqa-size={METAMATHQA_SIZE}", f"--orcamath-size={ORCAMATH_SIZE}", f"--ultrachat-size={ULTRACHAT_SIZE}",
    ], nproc=NPROC)
    src = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", PRETRAIN_MODEL_TAG)
    dst = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", ENHANCED_SFT_TAG)
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)
    volume.commit()

    # [7] Upload enhanced to HF
    print("[7/7] Uploading enhanced to HuggingFace...")
    _upload_to_hf()

    # Evals
    print("[eval] Running evals...")
    _torchrun("scripts.chat_eval", ["-i", "sft", f"--model-tag={BASELINE_SFT_TAG}"], nproc=NPROC)
    _torchrun("scripts.chat_eval", ["-i", "sft", f"--model-tag={ENHANCED_SFT_TAG}"], nproc=NPROC)
    volume.commit()

    print("\nAll done! HF: https://huggingface.co/alvina-yang/csc490a4p2")


def _upload_to_hf():
    """Upload all current checkpoints to HuggingFace via uv run subprocess."""
    _run(f"""cd /root/nanochat && uv run python -c "
import os
from huggingface_hub import HfApi, create_repo
api = HfApi()
repo_id = 'alvina-yang/csc490a4p2'
create_repo(repo_id, repo_type='model', exist_ok=True)
cache = '{NANOCHAT_CACHE}'
for subdir, tags in [('base_checkpoints', ['{PRETRAIN_MODEL_TAG}']), ('chatsft_checkpoints', ['{BASELINE_SFT_TAG}', '{ENHANCED_SFT_TAG}'])]:
    base = os.path.join(cache, subdir)
    if not os.path.exists(base): continue
    for tag in tags:
        tag_dir = os.path.join(base, tag)
        if not os.path.exists(tag_dir): continue
        for f in os.listdir(tag_dir):
            filepath = os.path.join(tag_dir, f)
            path_in_repo = f'{{tag}}/{{f}}'
            size_mb = os.path.getsize(filepath) / 1e6
            print(f'Uploading {{path_in_repo}} ({{size_mb:.0f}} MB)...')
            api.upload_file(path_or_fileobj=filepath, path_in_repo=path_in_repo, repo_id=repo_id)
print('Upload complete!')
"
""")


@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu=GPU, timeout=60 * 60 * 8,
)
def run_remaining() -> None:
    """Resume from after baseline SFT: upload baseline → SFT enhanced → upload → evals."""
    _setup_cache()
    _download_identity_conversations()

    print("[1/5] Uploading baseline to HuggingFace...")
    _upload_to_hf()

    print("[2/5] SFT enhanced...")
    _torchrun("scripts.chat_sft", [
        f"--model-tag={PRETRAIN_MODEL_TAG}", f"--device-batch-size={DEVICE_BATCH_SIZE}",
        "--run=sft-enhanced-d24",
        f"--metamathqa-size={METAMATHQA_SIZE}", f"--orcamath-size={ORCAMATH_SIZE}", f"--ultrachat-size={ULTRACHAT_SIZE}",
    ], nproc=NPROC)
    src = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", PRETRAIN_MODEL_TAG)
    dst = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", ENHANCED_SFT_TAG)
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)
    volume.commit()

    print("[3/5] Uploading enhanced to HuggingFace...")
    _upload_to_hf()

    print("[4/5] Eval baseline...")
    _torchrun("scripts.chat_eval", ["-i", "sft", f"--model-tag={BASELINE_SFT_TAG}"], nproc=NPROC)

    print("[5/5] Eval enhanced...")
    _torchrun("scripts.chat_eval", ["-i", "sft", f"--model-tag={ENHANCED_SFT_TAG}"], nproc=NPROC)
    volume.commit()

    print("\nAll done! HF: https://huggingface.co/alvina-yang/csc490a4p2")


@app.local_entrypoint()
def main() -> None:
    """Launch pipeline. Use --detach to survive laptop close."""
    print("Launching remaining pipeline on 8x H100...")
    run_remaining.remote()
    print("Pipeline complete!")
