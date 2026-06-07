#!/bin/bash

# Unified Dense model LoRA fine-tuning script
# Supports: 1.8B and 7B dense models
# Usage: bash train_dense_lora.sh [1.8B|7B]
#   - 1.8B: 1x GPU (24GB+)
#   - 7B:   1x GPU (48GB+ recommended)
# LoRA greatly reduces memory requirements compared to full fine-tuning.
# Optional:
#   ADAPTER_PATH=/path/to/previous/lora/checkpoint bash train_dense_lora.sh 1.8B

# ============== Model Size Selection ==============
MODEL_SIZE=${1:-"1.8B"}

if [[ "${MODEL_SIZE}" != "1.8B" && "${MODEL_SIZE}" != "7B" ]]; then
    echo "Error: MODEL_SIZE must be '1.8B' or '7B', got '${MODEL_SIZE}'"
    echo "Usage: bash train_dense_lora.sh [1.8B|7B]"
    exit 1
fi

# ============== NCCL Configuration ==============
NET_TYPE="high"
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_TIMEOUT=24
export NCCL_NVLS_ENABLE=0
export NCCL_MPI_PROFILE_PRIMS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_SOCKET_IFNAME=eth1
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
else
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=1
fi

# ============== Model-specific Configuration ==============
SCRIPT_DIR=$(dirname "$0")

if [[ "${MODEL_SIZE}" == "1.8B" ]]; then
    model_path=path_to_dense_1_8b_model
    output_path=dense_1_8b_lora_output
else
    model_path=path_to_dense_7b_model
    output_path=dense_7b_lora_output
fi

tokenizer_path=${model_path}
train_data_file=${SCRIPT_DIR}/../data/example_data.jsonl
adapter_path=${ADAPTER_PATH:-""}

LORA_ADAPTER_ARGS=()
if [[ -n "${adapter_path}" ]]; then
    LORA_ADAPTER_ARGS+=(--adapter-path "${adapter_path}")
fi

# ============== Output & Logging ==============
mkdir -p ${output_path}

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
log_file=${output_path}/"log_${current_time}.txt"

echo "============================================"
echo "Dense ${MODEL_SIZE} LoRA fine-tuning"
echo "Model path: ${model_path}"
echo "Adapter path: ${adapter_path:-<fresh adapter>}"
echo "Output path: ${output_path}"
echo "============================================"

# ============== Launch Training ==============
python \
    ${SCRIPT_DIR}/train_dense.py \
    --do_train \
    --model_size ${MODEL_SIZE} \
    --model_name_or_path ${model_path} \
    --tokenizer_name_or_path ${tokenizer_path} \
    --train_data_file ${train_data_file} \
    --output_dir ${output_path} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine_with_min_lr \
    --logging_steps 1 \
    --max_steps 30 \
    --save_steps 30 \
    --learning_rate 2e-4 \
    --min_lr 1e-5 \
    --warmup_ratio 0.01 \
    --save_strategy steps \
    --bf16 \
    --model_max_length 4096 \
    --max_seq_length 4096 \
    --use_qk_norm \
    --use_lora \
    "${LORA_ADAPTER_ARGS[@]}" \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 | tee ${log_file}
