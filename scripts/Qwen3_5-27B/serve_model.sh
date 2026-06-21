vllm serve Qwen/Qwen3.5-27B --port 8000 --tensor-parallel-size 1 --gpu-memory-utilization 0.8 \
    --max-num-seqs 10 --max-model-len 262144 --reasoning-parser qwen3
