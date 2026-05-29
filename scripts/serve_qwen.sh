#!/bin/bash
# Serve Qwen3-4B-Instruct on GPUs 0-3, port 8002

CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_V1=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --served-model-name qwen3-4b \
  --trust-remote-code \
  --data-parallel-size 4 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --max-logprobs 20 \
  --host 0.0.0.0 --port 8002 \
  --disable-log-requests \
  --disable-uvicorn-access-log
