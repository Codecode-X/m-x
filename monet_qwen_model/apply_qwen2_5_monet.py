"""
这行代码用了猴子补丁：不修改官方源码，偷偷把官方的模型代码换成我们自己改好的版本。

❗️❗️❗️❗️多卡有bug，还是采用cp文件直接覆盖的方式❗️❗️❗️❗️
# 运行前请先执行 inference/patch_vllm.sh 给 vLLM 打补丁（只需执行一次，补丁会备份原文件）。补丁会让 vLLM 支持 Monet 的模型和推理方式。
# bash inference/patch_vllm.sh
"""

import importlib.util, sys, pathlib, os

patch_path = pathlib.Path(__file__).with_name("modeling_qwen2_5_vl_monet.py")

spec  = importlib.util.spec_from_file_location(
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    patch_path,
)

patched_mod = importlib.util.module_from_spec(spec)

spec.loader.exec_module(patched_mod)

sys.modules["transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"] = patched_mod

print("Replaced the original Qwen2.5-VL model with the Monet version.")
