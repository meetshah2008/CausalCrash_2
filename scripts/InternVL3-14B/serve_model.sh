CUDA_VISIBLE_DEVICES=1 vllm serve OpenGVLab/InternVL3-14B --trust-remote-code --tensor-parallel-size 1 \
     --port 8001 --max-num-seqs 10 --max-model-len 65536 # --gpu-memory-utilization 0.9
