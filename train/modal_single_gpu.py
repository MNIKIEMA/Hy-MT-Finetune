"""Run single-GPU dense Hy-MT LoRA fine-tuning on Modal.

Example:
    modal run train/modal_single_gpu.py --model-id Tencent-Hunyuan/Hy-MT1.5-1.8B

The model, input JSONL, and checkpoints live in Modal Volumes, so retries do
not need to upload/download them again.  See the repository README for setup.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import modal

APP_NAME = "hy-mt-single-gpu-finetune"
PROJECT_DIR = "/root/hy-finetune"
DATA_DIR = "/data"
MODEL_DIR = "/models"
OUTPUT_DIR = "/outputs"

app = modal.App(APP_NAME)

# These names are deliberately stable: models and checkpoints survive after a
# function exits. `create_if_missing` makes first use a one-command setup.
data_volume = modal.Volume.from_name("hy-mt-training-data", create_if_missing=True)
model_volume = modal.Volume.from_name("hy-mt-models", create_if_missing=True)
output_volume = modal.Volume.from_name("hy-mt-training-output", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch>=2.10.0,<2.12",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        "transformers>=5.6.0",
        "datasets>=4.4.0",
        "accelerate>=1.11.0",
        "peft>=0.18.1",
        "sentencepiece",
        "protobuf",
        "wandb",
        "python-dotenv",
        "huggingface-hub",
    )
    .add_local_file("train/train_dense.py", f"{PROJECT_DIR}/train/train_dense.py")
)


def _safe_relative_path(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a relative path without '..': {value!r}")
    return path


def _model_directory(model_id: str) -> str:
    """Map a Hub repo ID to a stable, filesystem-safe volume directory."""
    return re.sub(r"[^A-Za-z0-9._-]+", "--", model_id).strip(".-")


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 12,
    # Modal requires explicit ephemeral-disk requests to be at least 512 GiB.
    ephemeral_disk=512 * 1024,
    volumes={
        DATA_DIR: data_volume,
        MODEL_DIR: model_volume,
        OUTPUT_DIR: output_volume,
    },
    # Create this once with: modal secret create huggingface-secret HF_TOKEN=...
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def train(
    model_id: str,
    train_data_file: str = "train.jsonl",
    model_size: str = "1.8B",
    output_name: str = "lora",
    max_steps: int = -1,
    num_train_epochs: float = 1,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 16,
    max_seq_length: int = 512,
    learning_rate: float = 2e-4,
    warmup_steps: int = 10,
    use_flash_attn: bool = False,
) -> str:
    """Fine-tune a Hub model and return its checkpoint directory on the output Volume."""
    if model_size not in {"0.5B", "1.8B", "4B", "7B"}:
        raise ValueError("model_size must be one of: 0.5B, 1.8B, 4B, 7B")

    data_path = Path(DATA_DIR) / _safe_relative_path(train_data_file, "train_data_file")
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Training data was not found at {data_path}. "
            "Upload it to the hy-mt-training-data Modal Volume first."
        )

    output_path = Path(OUTPUT_DIR) / _safe_relative_path(output_name, "output_name")
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(MODEL_DIR) / _model_directory(model_id)

    if not (model_path / "config.json").is_file():
        from huggingface_hub import snapshot_download

        print(f"Downloading {model_id} into the model Volume...", flush=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=model_path,
            token=os.environ["HF_TOKEN"],
        )
        model_volume.commit()

    command = [
        "python",
        f"{PROJECT_DIR}/train/train_dense.py",
        "--do_train",
        "--model_size", model_size,
        "--model_name_or_path", str(model_path),
        "--tokenizer_name_or_path", str(model_path),
        "--train_data_file", str(data_path),
        "--output_dir", str(output_path),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--gradient_checkpointing",
        "--lr_scheduler_type", "cosine_with_min_lr",
        "--logging_steps", "1",
        "--max_steps", str(max_steps),
        "--num_train_epochs", str(num_train_epochs),
        "--save_steps", "200",
        "--save_total_limit", "2",
        "--learning_rate", str(learning_rate),
        "--min_lr", "1e-6",
        # `warmup_ratio` is not accepted by some current Transformers builds;
        # explicit steps work across the versions supported by this project.
        "--warmup_steps", str(warmup_steps),
        "--save_strategy", "steps",
        "--model_max_length", str(max_seq_length),
        "--max_seq_length", str(max_seq_length),
        "--use_qk_norm",
        "--use_lora",
        "--lora_rank", "64",
        "--lora_alpha", "128",
        "--lora_dropout", "0.05",
        "--bf16",
        "--report_to", "none",
    ]
    if use_flash_attn:
        command.append("--use_flash_attn")

    try:
        subprocess.run(command, check=True)
    finally:
        # A commit in finally retains completed checkpoints/logs even when a
        # training process is interrupted after it has written a checkpoint.
        output_volume.commit()

    return str(output_path)
