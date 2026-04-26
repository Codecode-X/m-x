"""
本文件的作用：封装 vLLM 推理引擎的初始化、输入处理和生成三个核心步骤。

供 vllm_inference_example.py 及评估脚本调用，避免在每个推理脚本里重复写相同的初始化代码。

包含三个主要函数：
1. vllm_mllm_init()：初始化 vLLM 推理引擎和采样参数
2. vllm_mllm_process_batch_from_messages()：把多模态对话格式转换为 vLLM 输入格式
3. vllm_generate()：调用 vLLM 引擎批量生成回答
"""

# 导入 PyTorch，vLLM 底层依赖（此处未直接用到，可能是遗留导入）
import torch
# 导入 vLLM 核心类：LLM（推理引擎）、SamplingParams（采样参数）、EngineArgs（引擎启动参数）
from vllm import LLM, SamplingParams, EngineArgs
# 把 dataclass 对象转为普通字典，用于把 EngineArgs 解包传给 LLM
from dataclasses import asdict
# 导入类型注解工具
from typing import List
# 数学工具（此处未直接使用）
import math
# 用于在内存中构造字节流（此处未直接使用）
from io import BytesIO
# 更多类型注解
from typing import Any, Dict, List, Optional, Union
# 数值计算库（此处未直接使用）
import numpy as np
# 图像处理：PIL Image 类
from PIL import Image
# PIL Image 类型别名（用于类型注解）
from PIL.Image import Image as ImageObject
# 加载 HuggingFace tokenizer 和 processor（用于文本分词和图像预处理）
from transformers import AutoTokenizer, AutoProcessor
# Qwen-VL 官方提供的视觉信息提取工具（从多模态对话里提取图像数据）
from qwen_vl_utils import process_vision_info
# 进度条工具，显示批处理进度
from tqdm import tqdm
# GPU 显存清理工具（此处未直接使用，可能是遗留导入）
import gc
# 再次导入 math（重复导入，无害）
import math
# 再次导入 PIL Image（重复导入，无害）
from PIL import Image


# ─────────────────────────────────────────────
# 全局推理参数（可根据需要修改）
# ─────────────────────────────────────────────

# vLLM 同时处理的最大序列数（并发量），越大越快但显存占用越多
max_num_seqs = 512

# 采样温度：0.1 接近贪婪解码（确定性强），1.0 更随机（多样性强）
temperature = 0.1

# Top-K 采样：每步只从概率最高的 50 个 token 中采样
top_k = 50

# Top-P（核采样）：只从累计概率达到 80% 的 token 中采样
top_p = 0.8

# 重复惩罚系数：> 1 会降低已生成 token 的再次出现概率，减少重复
repetition_penalty = 1.01

# 每个输入生成几个候选答案（best_of=1 表示只生成 1 个）
best_of = 1
# 实际采样数量（与 best_of 相同）
n_generate_sample = best_of

# 最大生成 token 数（超过就截断）
max_tokens = 4096

# CPU swap 空间大小（GB），当 GPU 显存不足时把部分内容换出到 CPU 内存
swap_space = 7

# 随机种子（保证可复现性；temperature=0 时才真正生效）
seed = 0

# 停止词（None 表示不设置额外的停止条件，只靠 max_tokens 或 EOS token 停止）
stop = None

# 图片最大像素数（防止超大图片撑爆显存）：8192 × 28 × 28 ≈ 6.4M 像素
max_pixels = 8192 * 28 * 28

# 图片最小像素数（太小的图片会被上采样到这个尺寸）：256 × 28 × 28 ≈ 200K 像素
min_pixels = 256 * 28 * 28


def vllm_mllm_init(mllm_dir: str, tp=4, gpu_memory_utilization=0.95, max_model_len=4096):
    """
    初始化 vLLM 推理引擎和默认采样参数。
    
    参数：
    - mllm_dir: 模型目录路径（本地路径或 HuggingFace 模型名）
    - tp: tensor parallel size，张量并行数量，即用几张 GPU 分片模型（默认 4）
    - gpu_memory_utilization: 每张 GPU 最多使用多少比例的显存（默认 95%）
    - max_model_len: 最大输入+输出序列长度（默认 4096 tokens）
    
    返回：
    - mllm: 初始化好的 LLM 推理引擎对象
    - sampling_params: 默认的采样参数对象
    """
    
    # 构造引擎启动参数（dataclass 格式）
    engine_args = EngineArgs(
        model=mllm_dir,                   # 模型路径
        max_model_len=max_model_len,       # 最大序列长度
        max_num_seqs=max_num_seqs,         # 最大并发序列数
        tensor_parallel_size=tp,           # 张量并行 GPU 数量
        trust_remote_code=True,            # 允许加载模型目录里的自定义代码
        seed=seed,                         # 随机种子
        swap_space=swap_space,             # CPU swap 空间（GB）
        gpu_memory_utilization=gpu_memory_utilization,  # GPU 显存使用比例
        enforce_eager=True,                # 禁用 CUDA graph 优化（避免与 Monet 补丁冲突）
        # tp > 1 时使用 Ray 做多 GPU 分布式推理；单卡时不需要 Ray
        distributed_executor_backend='ray' if tp > 1 else None,
        dtype="bfloat16",                  # 使用 BF16 精度（推理更快，精度损失极小）
        mm_processor_kwargs={
            "min_pixels": min_pixels,      # 图像最小像素数（传给 processor）
            "max_pixels": max_pixels,      # 图像最大像素数（传给 processor）
        },
        enable_sleep_mode=True,            # 允许引擎在空闲时休眠，释放部分显存
        enable_chunked_prefill=True,       # 启用分块预填充，提升长序列处理效率
    )
    
    # 把 dataclass 转为普通字典，方便用 ** 解包传给 LLM
    engine_args = asdict(engine_args)
    
    # 创建 vLLM 推理引擎（这一步会加载模型权重到 GPU，比较耗时）
    mllm = LLM(**engine_args)
    
    # 创建默认的采样参数对象
    sampling_params = SamplingParams(
        temperature=temperature,           # 采样温度
        top_k=top_k,                       # Top-K 采样
        top_p=top_p,                       # Top-P 核采样
        repetition_penalty=repetition_penalty,  # 重复惩罚
        max_tokens=max_tokens,             # 最大生成长度
        n=n_generate_sample,               # 每个输入生成几个候选
        stop=stop,                         # 停止词
        skip_special_tokens=False,         # 不跳过特殊 token（保留 <abs_vis_token> 等标记）
        # 只在 temperature=0（贪婪解码）时设置种子，以确保可复现性
        seed=seed if temperature == 0 else None,
    )
    
    # 返回引擎和采样参数（供调用方使用）
    return mllm, sampling_params


def vllm_mllm_process_batch_from_messages(messages: List[List[dict]], processor):
    """
    把多模态对话格式转换为 vLLM 能接受的输入格式。
    
    输入格式（messages）：
        [
            [  # 第 1 个样本的对话
                {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image", "image": PIL.Image}]},
                ...
            ],
            [  # 第 2 个样本的对话
                ...
            ],
            ...
        ]
    
    输出格式（vllm_inputs）：
        [
            {"prompt": "...", "multi_modal_data": {"image": [PIL.Image, ...]}},  # 第 1 个样本
            ...
        ]
    
    参数：
    - messages: 批次对话列表，外层是 batch，内层是单个样本的对话轮次
    - processor: Qwen2.5-VL 的处理器，用于把对话转成 token 序列
    
    返回：
    - vllm_inputs: vLLM 格式的输入列表
    """
    
    # 检查输入格式是否正确：必须是列表的列表
    assert isinstance(messages, list) and all(isinstance(msg, list) for msg in messages), \
        "messages should be a list of lists"
    
    # 初始化结果列表
    vllm_inputs = []
    
    # 遍历 batch 里的每一个样本（用 tqdm 显示进度条）
    for msg in tqdm(messages, total=len(messages), desc="Processing vllm inputs"):
        
        # 把对话格式转换成 Qwen 模型的 prompt 字符串
        # tokenize=False：只生成文字 prompt，不做 tokenization（vLLM 内部会处理）
        # add_generation_prompt=True：在末尾加上生成提示（触发模型开始生成）
        prompt = processor.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # 从对话里提取所有图像数据（PIL Image 列表）
        # return_video_kwargs=False：只处理图片，不处理视频
        image_inputs, _ = process_vision_info(msg, return_video_kwargs=False)
        
        # 如果有图片，但 prompt 里没有图像占位符（某些模型格式的特殊情况），
        # 就在 prompt 开头手动加上 "<image>\n"
        if image_inputs and ("<image>" not in prompt and "<im_start>" not in prompt):
            prompt = "<image>\n" + prompt
        
        # 把 prompt 和图像数据打包成 vLLM 要求的输入格式
        vllm_inputs.append({
            "prompt": prompt,                              # 文字 prompt（含特殊 token）
            "multi_modal_data": {"image": image_inputs},  # 对应的图像数据列表
        })
    
    return vllm_inputs


def vllm_generate(
    inputs,
    sampling_params: SamplingParams,
    engine: LLM,
):
    """
    调用 vLLM 引擎，对一批输入执行批量生成。
    
    参数：
    - inputs: vllm_mllm_process_batch_from_messages() 的输出（批次输入列表）
    - sampling_params: 采样参数（温度、top_k、max_tokens 等）
    - engine: 初始化好的 vLLM LLM 推理引擎
    
    返回：
    - outputs: vLLM 输出列表，每个元素对应一个输入样本，
               通过 outputs[i].outputs[0].text 取得生成的文本
    """
    
    # 如果输入为空，直接返回空列表，避免无意义调用
    if not inputs:
        return []
    
    # 调用 vLLM 引擎批量生成
    # use_tqdm=True：显示生成进度条
    outputs = engine.generate(inputs, sampling_params=sampling_params, use_tqdm=True)
    
    return outputs
