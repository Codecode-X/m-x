#!/bin/bash

set -e

echo "===== Monet vLLM patch start ====="

# 当前目录
WORK_PATH=$(pwd)
echo "WORK DIR: $WORK_PATH"

MONET_GPU_MODEL_RUNNER_FILE_PATH="./vllm/monet_gpu_model_runner.py"

# 检查文件
if [ ! -f "$MONET_GPU_MODEL_RUNNER_FILE_PATH" ]; then
  echo "ERROR: monet_gpu_model_runner.py not found!"
  exit 1
fi

# vLLM 安装路径（如果你不是这个路径，需要改！！）
VLLM_PATH=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")

echo "VLLM PATH: $VLLM_PATH"

# # backup（❗️只备份一次，避免重复备份覆盖原文件）
# cp -n $VLLM_PATH/worker/model_runner.py $VLLM_PATH/worker/bkp-model_runner.py || true
# cp -n $VLLM_PATH/v1/worker/gpu_model_runner.py $VLLM_PATH/v1/worker/bkp-gpu_model_runner.py || true

# === 核心替换 ===

cp $MONET_GPU_MODEL_RUNNER_FILE_PATH $VLLM_PATH/worker/model_runner.py

cp $MONET_GPU_MODEL_RUNNER_FILE_PATH $VLLM_PATH/worker/gpu_model_runner.py || true

cp $MONET_GPU_MODEL_RUNNER_FILE_PATH $VLLM_PATH/v1/worker/gpu_model_runner.py

echo "===== Patch done ====="