# Hy-Finetune

Minimal DeepSpeed setup for fine-tuning Hy-MT 1.5 and Hy-MT2 models.

For full project documentation, model details, and upstream updates, see [Tencent-Hunyuan/Hy-MT2](https://github.com/Tencent-Hunyuan/Hy-MT2).

## Contents

```text
data/example_data.jsonl          # small ShareGPT-style example dataset
data.md                          # fr-mos/en-mos data mixing strategy
pyproject.toml                   # uv project dependencies and optional extras
train/train_dense.sh             # dense full fine-tuning launcher for 1.8B and 7B
train/train_dense_lora.sh        # dense LoRA fine-tuning launcher for 1.8B and 7B
train/single_gpu_finetune.sh     # single-GPU dense LoRA shortcut
train/train_dense.py             # dense training entrypoint
train/train.sh                   # MoE full fine-tuning launcher
train/train_lora.sh              # MoE LoRA fine-tuning launcher
train/train.py                   # MoE training entrypoint
train/merge_lora_weight.sh       # LoRA merge launcher
train/merge_lora_weight.py       # LoRA merge entrypoint
scripts/mix_translation_data.py   # fr/en -> mos weighted data mixer
train/ds_zero*.json              # DeepSpeed configs referenced by the launchers
```

## Install

Install the base training dependencies with uv:

```bash
cd Hy-Finetune
uv sync
```

Dataset loading, DeepSpeed, and FlashAttention are optional extras:

```bash
uv sync --extra data
uv sync --extra deepspeed
uv sync --extra flash
uv sync --extra deepspeed --extra flash
```

## Configure

Edit the launcher for the training run you want.

For dense models, update `model_path` in:

```text
train/train_dense.sh
train/train_dense_lora.sh
```

The dense launchers use the example dataset by default:

```text
data/example_data.jsonl
```

To train on your own data, replace `data/example_data.jsonl` or update the `train_data_file` value in the launcher.

## Run

Run commands from the repository root or from `train/`. The launchers resolve their own script directory.

1.8B dense full fine-tuning:

```bash
bash train/train_dense.sh 1.8B
```

7B dense full fine-tuning:

```bash
bash train/train_dense.sh 7B
```

1.8B dense LoRA fine-tuning:

```bash
bash train/train_dense_lora.sh 1.8B
```

Single-GPU dense LoRA shortcut:

```bash
MODEL_PATH=/path/to/Hy-MT-1.5-or-Hy-MT2-1.8B bash train/single_gpu_finetune.sh
```

7B dense LoRA fine-tuning:

```bash
bash train/train_dense_lora.sh 7B
```

The dense scripts default to `1.8B` if no model size is passed.

Continue LoRA training from a previous adapter checkpoint:

```bash
ADAPTER_PATH=dense_1_8b_lora_output/checkpoint-30 bash train/train_dense_lora.sh 1.8B
```

For staged data mixing, keep `model_path` pointed at the original base model and set `ADAPTER_PATH` to the previous stage adapter checkpoint for stage 2 and stage 3.

Build validation JSONL for dense training:

```bash
uv run python scripts/mix_translation_data.py validation
```

This writes `eval/fr_mos_natural.jsonl` from `madoss/fr-mos-final-data` validation and `eval/en_mos_flores_dev.jsonl` from FLORES+ `eng_Latn`/`mos_Latn` dev.

## Config Guide

The dense launchers select model-specific defaults:

| Dense model size | Full fine-tuning | LoRA fine-tuning |
| --- | --- | --- |
| 1.8B | `bash train/train_dense.sh 1.8B` | `bash train/train_dense_lora.sh 1.8B` |
| 7B | `bash train/train_dense.sh 7B` | `bash train/train_dense_lora.sh 7B` |

Full fine-tuning uses a lower learning rate, currently `1e-5`.

LoRA fine-tuning uses a higher learning rate, currently `2e-4`, and trains adapter weights with:

```bash
--use_lora
--lora_rank 64
--lora_alpha 128
--lora_dropout 0.05
```

For lower memory use, start with LoRA before trying full fine-tuning, especially for 7B.

## Outputs

Dense training outputs are written to the launcher-specific output folders, for example:

```text
dense_1_8b_output
dense_7b_output
dense_1_8b_lora_output
dense_7b_lora_output
```

## Quick Checklist

Before launching, check these items:

- `model_path` points to the correct local model folder.
- `train_data_file` points to your training data.
- `HOST_GPU_NUM` matches the number of GPUs you want to use.
- The selected `ds_config_file` matches your memory and hardware setup.
- For 7B training, start with LoRA unless you know full fine-tuning fits on your hardware.
