"""
本文件的作用：推理阶段的 vLLM "偷梁换柱" 补丁（同 apply_qwen2_5_monet.py 的推理版）

与 monet_qwen_model/apply_qwen2_5_monet.py 的区别：
- SFT 版（apply_qwen2_5_monet.py）是替换 Transformers 里的模型，用于训练时的 forward
- 本文件（apply_vllm_monet.py）是替换 vLLM 里的 GPU 推理引擎，用于推理时的 token 生成

具体做什么：
1. 设置环境变量，告诉 vLLM 使用 V1 引擎、latent token 的起止 ID
2. 把我们修改过的 monet_gpu_model_runner.py 注入到 vLLM 的模块缓存，
   替换 vLLM 官方的 GPU 推理引擎，使推理时支持 latent 思考向量的生成

必须在 `import vllm` 之前运行本文件，否则官方 vLLM 代码已经进了缓存，替换无效。

❗️❗️❗️❗️多卡有bug，还是采用cp文件直接覆盖的方式❗️❗️❗️❗️
# 运行前请先执行 inference/patch_vllm.sh 给 vLLM 打补丁（只需执行一次，补丁会备份原文件）。补丁会让 vLLM 支持 Monet 的模型和推理方式。
# bash inference/patch_vllm.sh
"""

# 导入三个必要工具：os（环境变量）、sys（模块缓存）、importlib（动态加载模块）
import os, sys, importlib


def patch():
    """
    执行所有"偷梁换柱"操作的函数，在文件底部会立即调用。
    """
    
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    workspace = os.path.abspath(".")
    old_path = os.environ.get("PYTHONPATH", "")

    os.environ["PYTHONPATH"] = f"{workspace}:{old_path}" if old_path else workspace
    
    # 设置 latent 推理的"开始 token ID"（即 <abs_vis_token> 的 token ID）
    # vLLM 在解码每一步时，检测到这个 ID 就切换到 latent 推理模式
    os.environ["LATENT_START_ID"] = "151666"
    
    # 设置 latent 推理的"结束 token ID"（即 </abs_vis_token> 的 token ID）
    # vLLM 检测到这个 ID 就退出 latent 推理模式，继续正常文本生成
    os.environ["LATENT_END_ID"] = "151667"
    
    try:

        patched = importlib.import_module("inference.vllm.monet_gpu_model_runner")

        for key in (
            "vllm.v1.worker.gpu_model_runner",   # V1 引擎的 GPU runner（新版 vLLM）
            "vllm.worker.gpu_model_runner",       # 旧版 vLLM 的 GPU runner
            "vllm.worker.model_runner",           # 更旧版本的路径
        ):
            sys.modules[key] = patched   # 注入模块缓存，实现偷梁换柱
        
        print("[Monet] vLLM runner patched via sitecustomize:", __file__)
    
    except Exception as e:
        print("[Monet] sitecustomize failed:", repr(e))

patch()
