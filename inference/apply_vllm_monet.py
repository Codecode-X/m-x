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
"""

# 导入三个必要工具：os（环境变量）、sys（模块缓存）、importlib（动态加载模块）
import os, sys, importlib


def patch():
    """
    执行所有"偷梁换柱"操作的函数，在文件底部会立即调用。
    """
    
    # 强制 vLLM 使用 V1 版本的推理引擎（V1 是 vLLM 的新架构，Monet 基于此开发补丁）
    os.environ["VLLM_USE_V1"] = "1"
    
    # 禁用 vLLM 的使用统计上报（避免网络请求干扰，加快启动速度）
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    
    # 获取当前工作目录的绝对路径（通常是项目根目录 Monet-main/）
    workspace = os.path.abspath(".")
    
    # 读取现有的 PYTHONPATH 环境变量（如果有的话）
    old_path = os.environ.get("PYTHONPATH", "")
    
    # 把项目根目录加入 PYTHONPATH，保证 Python 能找到 inference/ 等目录下的模块
    # 格式：新路径:旧路径（如果旧路径非空），或直接用新路径
    os.environ["PYTHONPATH"] = f"{workspace}:{old_path}" if old_path else workspace
    
    # 设置 latent 推理的"开始 token ID"（即 <abs_vis_token> 的 token ID）
    # vLLM 在解码每一步时，检测到这个 ID 就切换到 latent 推理模式
    os.environ["LATENT_START_ID"] = "151666"
    
    # 设置 latent 推理的"结束 token ID"（即 </abs_vis_token> 的 token ID）
    # vLLM 检测到这个 ID 就退出 latent 推理模式，继续正常文本生成
    os.environ["LATENT_END_ID"] = "151667"
    
    try:
        # 动态导入我们修改过的 vLLM GPU 推理引擎文件
        # 路径对应：inference/vllm/monet_gpu_model_runner.py
        # 注意：这里用 import_module，不是 importlib.util，说明它走的是正常的包导入路径
        patched = importlib.import_module("inference.vllm.monet_gpu_model_runner")
        
        # 把我们的修改版 GPU 推理引擎，注入到 vLLM 的三个相关模块名下
        # 无论 vLLM 用哪个内部路径来导入 GPU runner，都会拿到我们的版本
        for key in (
            "vllm.v1.worker.gpu_model_runner",   # V1 引擎的 GPU runner（新版 vLLM）
            "vllm.worker.gpu_model_runner",       # 旧版 vLLM 的 GPU runner
            "vllm.worker.model_runner",           # 更旧版本的路径
        ):
            sys.modules[key] = patched   # 注入模块缓存，实现偷梁换柱
        
        # 打印成功提示
        print("[Monet] vLLM runner patched via sitecustomize:", __file__)
    
    except Exception as e:
        # 如果替换失败（例如路径不对、文件缺失），打印错误但不崩溃
        # 程序会继续运行，但用的是官方 vLLM（latent 推理功能不可用）
        print("[Monet] sitecustomize failed:", repr(e))


# 文件被导入时立即执行 patch() 函数，不需要手动调用
# 所以只需要在推理脚本开头写 `import inference.apply_vllm_monet` 就能自动打补丁
patch()
