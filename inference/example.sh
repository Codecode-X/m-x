# cd Monet

# ❗️❗️❗️运行前请先执行 inference/patch_vllm.sh 给 vLLM 打补丁（只需执行一次，补丁会备份原文件）。补丁会让 vLLM 支持 Monet 的模型和推理方式。
# 🥳 bash inference/patch_vllm.sh

echo "❗️❗️❗️运行前请先执行 inference/patch_vllm.sh 给 vLLM 打补丁（只需执行一次，补丁会备份原文件）。补丁会让 vLLM 支持 Monet 的模型和推理方式。"


source /home/xiaojunhao/miniconda3/etc/profile.d/conda.sh && conda activate monet
export LATENT_SIZE=10

# 设置本地模型路径（stage2 是 SFT 最终阶段模型）
# export MONET_MODEL_PATH=/home/xiaojunhao/m-x/data/Monet-SFT-7B/stage3
# Model Output: <abs_vis_token> csak丰富多彩棕泥2 bins bins bins沉默</abs_vis_token><abs_vis_token>short�一定的他人olley checksCASikes bins</abs_vis_token><abs_vis_token> sl竞他人olleyoCASCASCASCAS</abs_vis_token><abs_vis_token>1希共产olleyolleyolololol</abs_vis_token>LOST MY MIND

# 关键：在启动 Ray 之前设置所有环境变量
export CUDA_VISIBLE_DEVICES=4,5,6,7
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MONET_MODEL_PATH=$MONET_MODEL_PATH"
echo "LATENT_SIZE=$LATENT_SIZE"

python -m inference.vllm_inference_example