# cd Monet

# ❗️❗️❗️运行前请先执行 inference/patch_vllm.sh 给 vLLM 打补丁（只需执行一次，补丁会备份原文件）。补丁会让 vLLM 支持 Monet 的模型和推理方式。
# 🥳 bash inference/patch_vllm.sh

echo "❗️❗️❗️运行前请先执行 inference/patch_vllm.sh 给 vLLM 打补丁（只需执行一次，补丁会备份原文件）。补丁会让 vLLM 支持 Monet 的模型和推理方式。"


source /home/xiaojunhao/miniconda3/etc/profile.d/conda.sh && conda activate monet
export LATENT_SIZE=10

# 关键：在启动 Ray 之前设置所有环境变量
export CUDA_VISIBLE_DEVICES=4,5,6,7
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -m inference.vllm_inference_example