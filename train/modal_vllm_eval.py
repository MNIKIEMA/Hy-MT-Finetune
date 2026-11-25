"""Score a Hy-MT base model and LoRA adapters offline with vLLM on Modal.

One `modal run` loads the base model and adapters in a single GPU container,
translates every set, returns the translations and exits: nothing stays
deployed. Sets, prompts, scoring and the output layout are those of
`scripts/evaluate_translation.py` (one prediction JSONL per model, set and
direction, plus `summary.json` / `summary.md`), so results compare with
server-based runs. Datasets are read locally; the container only needs the
model and output Volumes.

    MODAL_GPU=A100 uv run --extra modal --extra eval modal run train/modal_vllm_eval.py \
      --base-model hy-mt1_5-1_8b-mos-cpt-v1-ep1 \
      --loras cpt-ep1=hy-mt1_5-1_8b-cpt-v1-ep1-sft/checkpoint-782 \
      --sets bouquet,moore-web --bouquet-split dev --moore-web-split validation \
      --output-dir eval/results/cpt-v1-dev

`--base-model` is a folder in the `hy-mt-models` Volume; `--loras` takes
`name=path,...` relative to `hy-mt-training-output` (as the server's
`HY_MT_LORAS`). The base model is scored as `hy-mt` when no adapter is given
or with `--score-base`. Independent comparisons can run as parallel
`modal run`s, one per base model.
"""

from __future__ import annotations

import os

import modal

APP_NAME = "hy-mt-vllm-eval"
MODEL_DIR = "/models"
OUTPUT_DIR = "/outputs"
BASE_NAME = "hy-mt"
# Read at launch time: MODAL_GPU=A100 modal run ...
GPU = os.environ.get("MODAL_GPU", "A100")

app = modal.App(APP_NAME)

model_volume = modal.Volume.from_name("hy-mt-models")
output_volume = modal.Volume.from_name("hy-mt-training-output")

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("vllm==0.31.0")


def _parse_loras(spec: str) -> dict[str, str]:
    """`name=relative/path,...` -> {name: path}, paths relative to the output Volume."""
    loras = {}
    for item in filter(None, (part.strip() for part in spec.split(","))):
        name, _, path = item.partition("=")
        if not name or not path or path.startswith("/") or ".." in path.split("/") or name == BASE_NAME:
            raise ValueError(f"Invalid adapter {item!r}; expected name=relative/path (name not {BASE_NAME!r}).")
        loras[name] = path
    return loras


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60 * 2,
    volumes={MODEL_DIR: model_volume, OUTPUT_DIR: output_volume},
)
def translate(
    base_model: str,
    loras: dict[str, str],
    models: list[str],
    jobs: list[dict],
    repetition_penalty: float,
) -> dict[str, dict[str, list[str]]]:
    """Translate every job with every model; returns {model: {job key: [translation, ...]}}."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    base_path = f"{MODEL_DIR}/{base_model}"
    if not os.path.isfile(f"{base_path}/config.json"):
        raise FileNotFoundError(f"No model at {base_path} in the hy-mt-models Volume.")
    for name, path in loras.items():
        if not os.path.isfile(f"{OUTPUT_DIR}/{path}/adapter_config.json"):
            raise FileNotFoundError(f"Adapter {name!r}: no adapter_config.json in {OUTPUT_DIR}/{path}.")

    # Same engine settings as train/modal_vllm_server.py, so scores match served runs.
    llm = LLM(
        model=base_path,
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.90,
        # HunYuanDenseV1 runs through vLLM's Transformers backend, whose dynamic
        # RoPE branches on sequence length and breaks CUDA graph capture.
        enforce_eager=True,
        enable_lora=bool(loras),
        max_lora_rank=64,
    )
    requests = {name: LoRARequest(name, index + 1, f"{OUTPUT_DIR}/{path}") for index, (name, path) in enumerate(loras.items())}

    results: dict[str, dict[str, list[str]]] = {}
    for model in models:
        results[model] = {}
        for job in jobs:
            params = SamplingParams(temperature=0.0, max_tokens=job["max_tokens"], repetition_penalty=repetition_penalty)
            outputs = llm.chat(job["messages"], params, lora_request=requests.get(model), use_tqdm=False)
            results[model][job["key"]] = [output.outputs[0].text.strip() for output in outputs]
            print(f"{model} {job['key']}: {len(outputs)} rows", flush=True)
    return results


@app.local_entrypoint()
def main(
    base_model: str,
    output_dir: str,
    loras: str = "",
    score_base: bool = False,
    sets: str = "bouquet,moore-web",
    flores_dataset: str = "openlanguagedata/flores_plus",
    flores_split: str = "devtest",
    moore_web_split: str = "test",
    bouquet_split: str = "test",
    bouquet_levels: str = "sentence_level,paragraph_level",
    instruction_version: str = "v1",
    max_tokens: int = 256,
    paragraph_max_tokens: int = 384,
    repetition_penalty: float = 1.05,
    max_examples: int = 0,
    overwrite: bool = False,
) -> None:
    import json
    import sys
    from argparse import Namespace
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_translation import (
        bouquet_rows,
        flores_rows,
        moore_web_rows,
        prompt_messages,
        read_predictions,
        score,
        write_summary,
    )

    adapters = _parse_loras(loras)
    models = ([BASE_NAME] if score_base or not adapters else []) + list(adapters)
    set_names = [name.strip() for name in sets.split(",") if name.strip()]
    unknown = set(set_names) - {"flores", "moore-web", "bouquet"}
    if unknown:
        raise SystemExit(f"Unknown sets {sorted(unknown)}; choose from flores, moore-web, bouquet.")

    data = {}
    if "flores" in set_names:
        data[f"flores_{flores_split}"] = flores_rows(flores_dataset, flores_split)
    if "moore-web" in set_names:
        data[f"moore_web_{moore_web_split}"] = moore_web_rows(moore_web_split)
    if "bouquet" in set_names:
        for level in filter(None, bouquet_levels.split(",")):
            data[f"bouquet_{level}_{bouquet_split}"] = bouquet_rows(level, bouquet_split)
    if max_examples:
        data = {name: {d: rows[:max_examples] for d, rows in dirs.items()} for name, dirs in data.items()}

    out = Path(output_dir)
    prompt_args = Namespace(instruction_version=instruction_version, instruction=None)

    def path_for(model: str, set_name: str, direction: tuple[str, str]) -> Path:
        return out / model / set_name / f"{direction[0]}_{direction[1]}.jsonl"

    # Only translate what is missing, unless --overwrite.
    todo = {
        model: [(s, d) for s, dirs in data.items() for d in dirs if overwrite or not path_for(model, s, d).exists()]
        for model in models
    }
    needed = sorted({key for keys in todo.values() for key in keys})
    if needed:
        jobs = [
            {
                "key": f"{s}|{d[0]}|{d[1]}",
                "max_tokens": paragraph_max_tokens if "paragraph" in s else max_tokens,
                "messages": [prompt_messages(row["src_text"], d[0], d[1], prompt_args) for row in data[s][d]],
            }
            for s, d in needed
        ]
        to_run = [model for model in models if todo[model]]
        print(f"Translating {sum(len(j['messages']) for j in jobs)} rows x {len(to_run)} model(s) on {GPU}...")
        translations = translate.remote(base_model, adapters, to_run, jobs, repetition_penalty)
        for model in to_run:
            for s, d in todo[model]:
                texts = translations[model][f"{s}|{d[0]}|{d[1]}"]
                path = path_for(model, s, d)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as handle:
                    for row, text in zip(data[s][d], texts, strict=True):
                        prediction = {"id": row["id"], "src_lang": d[0], "tgt_lang": d[1], **row, "mt_text": text}
                        handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    results = [
        {"model": model, "set": s, "direction": f"{d[0]}->{d[1]}", **score(read_predictions(path_for(model, s, d)))}
        for model in models
        for s, dirs in data.items()
        for d in dirs
    ]
    out.mkdir(parents=True, exist_ok=True)
    write_summary(results, models, out)
