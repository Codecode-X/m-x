"""
本文件的作用：推理示例脚本（快速上手体验 Monet-7B 模型）

这是整个项目中最简单、最直接的入口文件，只有 41 行，演示了：
1. 如何加载 Monet-7B 模型（带 latent 推理能力）
2. 如何输入一张图片和一个问题
3. 如何运行推理，得到模型的回答
4. 如何清理输出中 latent token 之间的不可读内容

运行方式（在项目根目录下）：
    export LATENT_SIZE=10          # 设置每次 latent 推理生成的向量数量
    python -m inference.vllm_inference_example
"""

# ★ 必须最先导入这个补丁！在 vllm 被加载之前偷梁换柱，替换 GPU 推理引擎
# 如果这行放到后面，vLLM 官方代码已经进了缓存，latent 推理功能就不生效了
import inference.apply_vllm_monet

# 导入 PIL 图像处理库，用于打开本地图片文件
import PIL.Image

# 导入 load_and_gen_vllm.py 里的所有工具函数（vllm_mllm_init、vllm_generate 等）
# 用 * 导入，所以后面可以直接用这些函数名，不加前缀
from inference.load_and_gen_vllm import *

# 导入操作系统接口（此处未直接使用，可能是遗留导入）
import os
# 导入 PIL 图像库（与 PIL.Image 重复，实际只需要一个）
import PIL
# 导入正则表达式库，用于清理输出文本中的 latent token 内容
import re

# 指定 Monet-7B 模型的本地路径（需要用户自行修改为实际路径）
# 可以从 HuggingFace 下载：https://huggingface.co/NOVAglow646/Monet-7B
model_path = 'Path/to/your/model'


def replace_abs_vis_token_content(s: str) -> str:
    """
    清理模型输出中 latent token 之间的不可读内容。
    
    模型输出格式（原始）：
        <abs_vis_token>【一堆乱码/不可见字符（latent 向量的 token 表示）】</abs_vis_token>
    
    处理后变为：
        <abs_vis_token><latent></abs_vis_token>
    
    参数：s - 原始输出字符串
    返回：清理后的字符串（latent 内容被替换为 <latent> 占位符）
    """
    # 编译正则表达式：
    # (<abs_vis_token>)  → 捕获组1：开始标记
    # (.*?)              → 捕获组2：中间所有内容（非贪婪，尽量短）
    # (</abs_vis_token>) → 捕获组3：结束标记
    # flags=re.DOTALL    → 让 . 也能匹配换行符（latent 内容可能跨行）
    pattern = re.compile(r'(<abs_vis_token>)(.*?)(</abs_vis_token>)', flags=re.DOTALL)
    
    # 用 \1<latent>\3 替换匹配到的内容：
    # \1 保留开始标记，\3 保留结束标记，中间换成 <latent>
    return pattern.sub(r'\1<latent>\3', s)


def main():
    """主函数：加载模型 → 处理输入 → 生成输出 → 打印结果"""
    
    # 初始化 vLLM 推理引擎和采样参数
    # tp=1 表示只用 1 张 GPU（tensor parallelism = 1）
    # gpu_memory_utilization=0.8 表示最多使用 80% 的 GPU 显存
    mllm, sampling_params = vllm_mllm_init(model_path, tp=1, gpu_memory_utilization=0.8)
    
    # 加载 Qwen2.5-VL 的处理器（包含 tokenizer 和图像预处理器）
    # trust_remote_code=True 允许加载模型目录里的自定义代码
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 构造输入对话列表
    # 外层列表：batch（这里只有 1 个样本，所以只有一个元素）
    # 内层列表：这个样本的多轮对话（这里只有 1 轮 user 消息）
    conversations = [
        [
            {
                "role": "user",          # 用户角色
                "content": [
                    # 第一个内容块：文字问题
                    {"type": "text", "text": "Question:  Which car has the longest rental period? The choices are listed below:\n(A)DB11 COUPE.\n(B) V12 VANTAGES COUPES.\n(C) VANQUISH VOLANTE.\n(D) V12 VOLANTE.\n(E) The image does not feature the time. Put your final answer in \\boxed{}."},
                    # 第二个内容块：图片（打开本地示例图片，转为 RGB 格式）
                    {"type": "image", "image": PIL.Image.open('images/example_question.png').convert("RGB")}
                ]
            }
        ]
    ]
    
    # 把对话格式的输入转换成 vLLM 能接受的格式（包括 prompt 字符串和图像张量）
    inputs = vllm_mllm_process_batch_from_messages(conversations, processor)
    
    # 用 vLLM 引擎生成回答
    # output 是一个列表，每个元素对应 batch 里的一个输入样本
    output = vllm_generate(inputs, sampling_params, mllm)
    
    # 取出第一个样本（output[0]）的第一个生成结果（.outputs[0]）的文本
    raw_output_text = output[0].outputs[0].text
    
    # 清理输出：把 latent token 之间的不可读内容替换为 <latent> 占位符
    cleaned_output_text = replace_abs_vis_token_content(raw_output_text)
    
    # 打印最终的可读输出
    print(cleaned_output_text)


# Python 标准入口：当本文件被直接运行时（而不是被 import 时），才执行 main()
if __name__ == '__main__':
    main()
