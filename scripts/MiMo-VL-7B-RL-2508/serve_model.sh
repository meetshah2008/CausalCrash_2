CUDA_VISIBLE_DEVICES=1 vllm serve XiaomiMiMo/MiMo-VL-7B-RL-2508 \
    --port 8001 \
    --tensor-parallel-size 1
    # --trust-remote-code \
    # --max-num-seqs 10 \
    # --max-model-len 40960 \
    # --gpu-memory-utilization 0.9
