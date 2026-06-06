MODEL_PATH=./hf_train_output/checkpoint-600

python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8021 \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --served-model-name hy-mt \
    --gpu_memory_utilization 0.92 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 512 \
    --disable-log-stats \
    2>&1 | tee log_server.txt
