vllm serve OpenGVLab/InternVL3_5-14B --trust-remote-code --tensor-parallel-size 1 \
    --port 8000 --max-num-seqs 10 --max-model-len 40960 # --gpu-memory-utilization 0.9
