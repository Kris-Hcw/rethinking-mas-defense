#!/bin/bash
# Serve LLaMA-3.1-8B-Instruct on GPUs 4-7, port 8001
# Requires vLLM and Hugging Face model access.

CUDA_VISIBLE_DEVICES=4,5,6,7 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --served-model-name llama3-8b \
  --data-parallel-size 4 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port 8001 \
  --max-logprobs 20 \
  --disable-log-requests \
  --disable-uvicorn-access-log
