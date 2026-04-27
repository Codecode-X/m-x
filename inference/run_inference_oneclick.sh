#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/xiaojunhao/m-x"
VENV_PYTHON="${VENV_PYTHON:-/home/xiaojunhao/.venvs/monet_isolated/bin/python}"
LATENT_SIZE="${LATENT_SIZE:-10}"

# 优先级：命令行参数 > 环境变量 > 默认 HF 模型
MODEL_PATH="${1:-${MONET_MODEL_PATH:-NOVAglow646/Monet-7B}}"
export MONET_MODEL_PATH="$MODEL_PATH"
export LATENT_SIZE

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "ERROR: python not found in isolated venv: $VENV_PYTHON"
  echo "Please check your venv path."
  exit 1
fi

if [[ ! -f "$REPO_ROOT/images/example_question.png" ]]; then
  echo "ERROR: missing example image: $REPO_ROOT/images/example_question.png"
  exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"

echo "Using python: $VENV_PYTHON"
echo "Using model : $MONET_MODEL_PATH"
echo "LATENT_SIZE : $LATENT_SIZE"

"$VENV_PYTHON" -m inference.vllm_inference_example
