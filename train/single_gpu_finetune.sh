#!/bin/bash

# Single-GPU dense LoRA fine-tuning shortcut.
# This keeps the simple flow from hy-mt/single_gpu_finetune.sh while using
# train/train_dense.py and this repo's ShareGPT-style JSONL data format.
#
# Usage:
#   MODEL_PATH=/path/to/Hy-MT-1.5-or-Hy-MT2-1.8B bash train/single_gpu_finetune.sh
#   MODEL_PATH=/path/to/Hy-MT-1.5-or-Hy-MT2-7B MODEL_SIZE=7B bash train/single_gpu_finetune.sh

set -euo pipefail

USE_UV=false

while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --uv) USE_UV=true ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MODEL_SIZE=${MODEL_SIZE:-"1.8B"}
if [[ "${MODEL_SIZE}" != "1.8B" && "${MODEL_SIZE}" != "7B" ]]; then
    echo "Error: MODEL_SIZE must be '1.8B' or '7B', got '${MODEL_SIZE}'"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "Error: MODEL_PATH must point to a local dense Hy-MT 1.5 or Hy-MT2 model directory."
    echo "Usage: MODEL_PATH=/path/to/Hy-MT-1.5-or-Hy-MT2-1.8B bash train/single_gpu_finetune.sh"
    exit 1
fi

model_path=${MODEL_PATH}
tokenizer_path=${TOKENIZER_PATH:-${model_path}}
train_data_file=${TRAIN_DATA_FILE:-"${SCRIPT_DIR}/../data/example_data.jsonl"}
eval_data_file=${EVAL_DATA_FILE:-""}
output_path=${OUTPUT_PATH:-"${SCRIPT_DIR}/../hf_train_output"}

mkdir -p "${output_path}"

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
log_file="${output_path}/log_${current_time}.txt"

RUNNER=()
if $USE_UV; then
    RUNNER=(uv run)
fi

EVAL_ARGS=()
BEST_MODEL_ARGS=(--load_best_model_at_end false)
if [[ -n "${eval_data_file}" ]]; then
    EVAL_ARGS+=(--eval_data_file "${eval_data_file}" --eval_strategy steps --eval_steps "${EVAL_STEPS:-200}")
    BEST_MODEL_ARGS=(--load_best_model_at_end true --metric_for_best_model eval_loss --greater_is_better false)
fi

echo "============================================"
echo "Single-GPU dense ${MODEL_SIZE} LoRA fine-tuning"
echo "Model path: ${model_path}"
echo "Tokenizer path: ${tokenizer_path}"
echo "Train data: ${train_data_file}"
echo "Eval data: ${eval_data_file:-<none>}"
echo "Output path: ${output_path}"
echo "============================================"

"${RUNNER[@]}" python \
    "${SCRIPT_DIR}/train_dense.py" \
    --do_train \
    --model_size "${MODEL_SIZE}" \
    --model_name_or_path "${model_path}" \
    --tokenizer_name_or_path "${tokenizer_path}" \
    --train_data_file "${train_data_file}" \
    "${EVAL_ARGS[@]}" \
    --output_dir "${output_path}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-16}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
    --gradient_checkpointing \
    --lr_scheduler_type cosine_with_min_lr \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
    --logging_steps "${LOGGING_STEPS:-1}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --max_steps "${MAX_STEPS:--1}" \
    --save_steps "${SAVE_STEPS:-200}" \
    --learning_rate "${LEARNING_RATE:-5e-5}" \
    --min_lr "${MIN_LR:-1e-6}" \
    --warmup_ratio "${WARMUP_RATIO:-0.01}" \
    --save_strategy steps \
    --model_max_length "${MODEL_MAX_LENGTH:-512}" \
    --max_seq_length "${MAX_SEQ_LENGTH:-512}" \
    --use_qk_norm \
    --use_lora \
    --lora_rank "${LORA_RANK:-64}" \
    --lora_alpha "${LORA_ALPHA:-128}" \
    --lora_dropout "${LORA_DROPOUT:-0.05}" \
    --bf16 \
    --report_to wandb \
    "${BEST_MODEL_ARGS[@]}" | tee "${log_file}"
