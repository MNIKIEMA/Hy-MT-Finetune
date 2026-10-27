# Hy-Finetune

Minimal setup for fine-tuning Hy-MT 1.5 and Hy-MT2 models.

For full project documentation, model details, and upstream updates, see [Tencent-Hunyuan/Hy-MT2](https://github.com/Tencent-Hunyuan/Hy-MT2).

## Contents

```text
data/example_data.jsonl          # small ShareGPT-style example dataset
data.md                          # fr-mos/en-mos data mixing strategy
pyproject.toml                   # uv project dependencies and optional extras
train/train_dense.sh             # dense full fine-tuning launcher for 1.8B and 7B
train/train_dense_lora.sh        # dense LoRA fine-tuning launcher for 1.8B and 7B
train/single_gpu_finetune.sh     # single-GPU dense LoRA shortcut
train/modal_single_gpu.py        # Modal single-GPU dense LoRA launcher
train/train_dense.py             # dense training entrypoint
train/train.sh                   # MoE full fine-tuning launcher
train/train_lora.sh              # MoE LoRA fine-tuning launcher
train/train.py                   # MoE training entrypoint
train/merge_lora_weight.sh       # LoRA merge launcher
train/merge_lora_weight.py       # LoRA merge entrypoint
scripts/mix_translation_data.py  # mos/dyu/fra/eng bidirectional data mixer
train/ds_zero*.json              # DeepSpeed configs for full fine-tuning launchers
```

## Install

Install the base training dependencies with uv:

```bash
cd Hy-Finetune
uv sync
```

DeepSpeed and FlashAttention are optional extras:

```bash
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

### Single GPU on Modal

The Modal launcher runs dense LoRA training on one `L4`, stores models,
data, and checkpoints in persistent Modal Volumes, and downloads a Hugging Face
model only when it is not already present in its model Volume.

Install and authenticate the Modal client, then create a Hugging Face token
secret (needed for gated and public Hub downloads alike):

```bash
uv sync --extra modal
uv run modal setup
uv run modal secret create huggingface-secret HF_TOKEN=your_huggingface_token
```

Upload a ShareGPT JSONL dataset to the data Volume. The remote path is relative
to `/data` in the training container. Create the Volume first; `modal volume put`
cannot create it:

```bash
uv run modal volume create hy-mt-training-data
uv run modal volume put hy-mt-training-data data/example_data.jsonl train.jsonl
```

Launch training from the repository root:

```bash
uv run modal run train/modal_single_gpu.py \
  --model-id tencent/HY-MT1.5-1.8B \
  --model-size 1.8B \
  --train-data-file train.jsonl \
  --output-name hy-mt-1_8b-lora
```

The output is kept in the `hy-mt-training-output` Volume. Download it with:

```bash
uv run modal volume get hy-mt-training-output hy-mt-1_8b-lora ./hy-mt-1_8b-lora
```

For a 7B model, use `--model-size 7B`; an L4 may not have enough memory, so
reduce the batch size or sequence length, or choose a larger GPU. To choose a
different GPU, edit `gpu="L4"` in
`train/modal_single_gpu.py`.

7B dense LoRA fine-tuning:

```bash
bash train/train_dense_lora.sh 7B
```

The dense scripts default to `1.8B` if no model size is passed.

Continue LoRA training from a previous adapter checkpoint:

```bash
ADAPTER_PATH=dense_1_8b_lora_output/checkpoint-30 bash train/train_dense_lora.sh 1.8B
```

To continue on a new data mix, keep `model_path` pointed at the original base model and set `ADAPTER_PATH` to the previous adapter checkpoint.

## Data Mixing

The mixer builds bidirectional training data for Mooré (`mos`), Dyula (`dyu`),
French (`fra`) and English (`eng`) from Hugging Face datasets:

```text
madoss/moore-web-parallel        mos-fra
madoss/faso-web-parallel         dyu-fra, dyu-eng, dyu-mos
burkimbia/nllb-burkina-parallel  eng-mos, fra-mos, eng-dyu, fra-dyu, mos-dyu
```

FLORES+ is gated: accept its terms on the Hub and put `HF_TOKEN` in `.env`.

Build validation JSONL:

```bash
uv run python scripts/mix_translation_data.py validation
```

This writes one file per direction, plus a small combined file for evaluation
during training:

```text
eval/flores_dev/<src>_<tgt>.jsonl            # 10 directions, FLORES+ dev
eval/moore_web_validation/<src>_<tgt>.jsonl  # fra->mos and mos->fra
eval/validation.jsonl                        # 100 rows per direction
```

Build the training mix:

```bash
uv run python scripts/mix_translation_data.py mix \
  --total-examples 200000 \
  --output data/mixed.jsonl
```

The mixer drops moore-web validation/test and FLORES+ dev/devtest sentences
from training, removes duplicates within each pair, and prints the usable pool
and repeat factor of every source. Change a pair share with
`--pair-weight dyu-mos=0.2`, and the share of rows translated into mos/dyu with
`--into-local-ratio`.

See `data.md` for the pair weights, source filters, and evaluation plan.

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
- For full fine-tuning, the selected `ds_config_file` matches your memory and hardware setup.
- For 7B training, start with LoRA unless you know full fine-tuning fits on your hardware.
