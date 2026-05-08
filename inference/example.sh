# cd Monet

# 运行前请先执行 inference/patch_vllm.sh 给 vLLM 打补丁（只需执行一次，补丁会备份原文件）。补丁会让 vLLM 支持 Monet 的模型和推理方式。
# bash inference/patch_vllm.sh

source /home/xiaojunhao/miniconda3/etc/profile.d/conda.sh && conda activate monet
export LATENT_SIZE=10
python -m inference.vllm_inference_example