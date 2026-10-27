"""Serve a Hy-MT base model and its LoRA adapters with vLLM on Modal.

The server exposes an OpenAI-compatible API. The base model is served as
`hy-mt` and each adapter under its own name, so one server scores both:

    MODAL_GPU=A100 modal deploy train/modal_vllm_server.py

Requests need the API key stored in the `hy-mt-vllm` secret:

    modal secret create hy-mt-vllm VLLM_API_KEY=<key>

Adapters are read from the training output Volume. Override the defaults at
deploy time, e.g.:

    HY_MT_LORAS="mos-dyu=hy-mt1_5-1_8b-mos-dyu-lora/checkpoint-2400" modal deploy ...
"""

from __future__ import annotations

import os
import subprocess

import modal

APP_NAME = "hy-mt-vllm"
MODEL_DIR = "/models"
OUTPUT_DIR = "/outputs"
PORT = 8000
# Read at deploy time and baked into the image env, so the container sees the
# same values: MODAL_GPU=A100 modal deploy ...
GPU = os.environ.get("MODAL_GPU", "A100")
BASE_MODEL = os.environ.get("HY_MT_BASE_MODEL", "tencent--HY-MT1.5-1.8B")
LORAS = os.environ.get("HY_MT_LORAS", "mos-dyu=hy-mt1_5-1_8b-mos-dyu-lora/checkpoint-2400")
MAX_LORA_RANK = os.environ.get("HY_MT_MAX_LORA_RANK", "64")

app = modal.App(APP_NAME)

model_volume = modal.Volume.from_name("hy-mt-models")
output_volume = modal.Volume.from_name("hy-mt-training-output")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.31.0")
    .env(
        {
            "HY_MT_BASE_MODEL": BASE_MODEL,
            "HY_MT_LORAS": LORAS,
            "HY_MT_MAX_LORA_RANK": MAX_LORA_RANK,
        }
    )
)


def _lora_modules(spec: str) -> list[str]:
    """Turn `name=relative/path,...` into vLLM `--lora-modules name=/outputs/...` values."""
    modules = []
    for item in filter(None, (part.strip() for part in spec.split(","))):
        name, _, path = item.partition("=")
        if not name or not path or path.startswith("/") or ".." in path.split("/"):
            raise ValueError(f"Invalid adapter {item!r}; expected name=relative/path.")
        modules.append(f"{name}={OUTPUT_DIR}/{path}")
    return modules


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60 * 4,
    # Scale to zero after 10 idle minutes; the next request starts a new container.
    scaledown_window=60 * 10,
    volumes={MODEL_DIR: model_volume, OUTPUT_DIR: output_volume},
    secrets=[modal.Secret.from_name("hy-mt-vllm", required_keys=["VLLM_API_KEY"])],
)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=PORT, startup_timeout=60 * 15)
def serve() -> None:
    command = [
        "vllm",
        "serve",
        f"{MODEL_DIR}/{os.environ['HY_MT_BASE_MODEL']}",
        "--served-model-name", "hy-mt",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--api-key", os.environ["VLLM_API_KEY"],
        "--dtype", "bfloat16",
        # FLORES+ and moore-web rows stay far below this with a 256-token reply.
        "--max-model-len", "1024",
        "--gpu-memory-utilization", "0.90",
        # vLLM runs HunYuanDenseV1 through its Transformers backend, whose dynamic
        # RoPE branches on sequence length and breaks CUDA graph capture.
        "--enforce-eager",
    ]
    loras = _lora_modules(os.environ["HY_MT_LORAS"])
    if loras:
        command += ["--enable-lora", "--max-lora-rank", os.environ["HY_MT_MAX_LORA_RANK"], "--lora-modules", *loras]
    subprocess.Popen(command)
