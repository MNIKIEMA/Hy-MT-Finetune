"""Merge a LoRA adapter into its base model on Modal, as a new base model.

The adapter is read from the `hy-mt-training-output` Volume and the merged
model is written to the `hy-mt-models` Volume, where `modal_single_gpu.py`
(`--model-id <output-model>`) and `modal_vllm_server.py`
(`HY_MT_BASE_MODEL=<output-model>`) find it. Used to start translation
training from a continued-pretraining (CPT) adapter:

    MODAL_PROFILE=nikiemamahamadi01 modal run train/modal_merge.py \
        --adapter-path hy-mt1_5-1_8b-mos-cpt/checkpoint-1000 \
        --output-model hy-mt1_5-1_8b-mos-cpt-v1

The adapter checkpoint is only read, never changed. The merged model is saved
as bf16 safetensors with the base model's tokenizer and config files. An
existing output model is never overwritten.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import modal

APP_NAME = "hy-mt-merge-lora"
PROJECT_DIR = "/root/hy-finetune"
MODEL_DIR = "/models"
OUTPUT_DIR = "/outputs"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("hy-mt-models")
output_volume = modal.Volume.from_name("hy-mt-training-output")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch>=2.10.0,<2.12", index_url="https://download.pytorch.org/whl/cpu")
    .uv_pip_install("transformers>=5.6.0", "peft>=0.18.1", "accelerate>=1.11.0", "safetensors")
    .add_local_file("train/merge_lora_weight.py", f"{PROJECT_DIR}/train/merge_lora_weight.py")
)


def model_directory(model_id: str) -> str:
    """Same mapping as `modal_single_gpu.py`: a Hub repo ID to its Volume directory."""
    return re.sub(r"[^A-Za-z0-9._-]+", "--", model_id).strip(".-")


def relative_path(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{label} must be a relative path without '..': {value!r}")
    return path


@app.function(
    image=image,
    cpu=8,
    memory=32 * 1024,
    timeout=60 * 60,
    volumes={MODEL_DIR: model_volume, OUTPUT_DIR: output_volume},
)
def merge(
    adapter_path: str,
    output_model: str,
    base_model_id: str = "tencent/HY-MT1.5-1.8B",
    save_dtype: str = "bf16",
) -> str:
    """Merge `/outputs/<adapter_path>` into the base model as `/models/<output_model>`."""
    if model_directory(output_model) != output_model:
        raise ValueError(
            f"--output-model {output_model!r} must use only letters, digits, '.', '_' and '-' "
            "(no '/'), so that `--model-id` maps to the same folder."
        )
    base_path = Path(MODEL_DIR) / model_directory(base_model_id)
    if not (base_path / "config.json").is_file():
        raise FileNotFoundError(f"Base model {base_model_id} is not in the hy-mt-models Volume ({base_path}).")
    adapter = Path(OUTPUT_DIR) / relative_path(adapter_path, "--adapter-path")
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No LoRA adapter (adapter_config.json) at {adapter}.")
    output_path = Path(MODEL_DIR) / output_model
    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists; choose another --output-model.")

    try:
        subprocess.run(
            [
                "python",
                f"{PROJECT_DIR}/train/merge_lora_weight.py",
                "--base_model_path", str(base_path),
                "--adapter_model_path", str(adapter),
                "--output_path", str(output_path),
                "--save_dtype", save_dtype,
            ],
            check=True,
        )
    except BaseException:
        # Do not leave a half-written model that the launcher would pick up.
        shutil.rmtree(output_path, ignore_errors=True)
        raise
    model_volume.commit()
    print(f"Merged {adapter} into {base_model_id}: /models/{output_model}")
    return output_model
