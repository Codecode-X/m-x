"""
本文件的作用：SFT 训练数据的预处理（Preprocessing）

在 SFT 训练开始之前，原始数据集（Monet-SFT-125K）里的每个样本都是一段"多轮对话"，
格式包含用户问题、图片路径（相对路径）、助手回答（含 <observation> 文字标注
和 <abs_vis_token> latent 占位符）。

本文件定义了两个预处理函数：
1. Monet_single_input_images_preprocess_function：
   完整版预处理，用于 SFT 训练，会严格验证数据格式的合法性，过滤不合格样本。
2. Monet_single_input_images_preprocess_function_question_only：
   只保留"问题"部分的简化预处理，用于推理/评估阶段（不需要完整答案）。

最后，task_preporcess_config 字典把任务名称映射到对应的预处理函数，
供 src/utils.py 在加载数据集时按任务类型自动调用。
"""

# 导入图像处理库 PIL（用于加载图片文件）
from PIL import Image
# 导入路径操作工具（这里实际未直接使用，可能用于其他地方 import 时间接依赖）
from pathlib import Path
# 导入操作系统接口，主要用于 os.path.join 拼接图片绝对路径
import os
# 导入 src/utils.py 里的所有工具函数（如 get_args、seed_everything 等）
from src.utils import *
# 导入 Qwen-VL 官方提供的视觉信息处理工具（解析多模态对话格式）
from qwen_vl_utils import process_vision_info


def Monet_single_input_images_preprocess_function(sample, dataset_root="", allow_no_observation=False):
    """
    完整版 SFT 数据预处理函数。
    
    功能：
    - 把对话中图片的相对路径转为绝对路径（方便训练时加载图片）
    - 验证数据格式合法性：每张助手图片前必须有 <abs_vis_token></abs_vis_token> 占位符
    - 过滤没有 <observation> 标注的样本（这种样本对 Stage 2 训练无用）
    
    参数：
    - sample: 单个训练样本，包含 "data" 字段（多轮对话列表）
    - dataset_root: 数据集根目录路径，用于拼接图片绝对路径
    - allow_no_observation: 是否允许样本中没有 <observation> 标签（默认不允许）
    
    返回：
    - 处理好的 sample（合法）或 None（不合法，会被过滤掉）
    """
    
    # 计数器：统计对话中助手回复里 <abs_vis_token></abs_vis_token> 占位符的个数
    n_img_pad = 0
    # 计数器：统计对话中助手回复里实际图片（type=="image"）的个数
    n_img = 0
    # 从样本中取出多轮对话列表（每个元素是一个"轮次"，包含 role 和 content）
    conversations = sample["data"]
    # 标记是否见过至少一个 <observation> 标签
    seen_observation = False
    
    # 遍历对话中的每一个轮次（i 是下标，step 是该轮次的字典）
    for i, step in enumerate(conversations):
        # 浅拷贝这一轮次，避免在原数据上直接修改（防止影响原始样本）
        new_step = step.copy()
        
        # 如果这一轮是"系统提示"（role == "system"），统一把内容改为标准的 helpful assistant 提示
        if step["role"] == "system":
            new_step["content"][0]["text"] = "You are a helpful assistant."
        
        # 如果当前轮次是助手（assistant），初始化"是否见过助手图片"的标记为 False；
        # 如果是用户（user）或系统（system），则不需要追踪，设为 None
        seen_assistant_image = False if step["role"] == "assistant" else None
        
        # 遍历这一轮次 content 里的每个内容块（j 是下标，content 是该内容块）
        for j, content in enumerate(new_step["content"]):
            
            # ── 处理图片类型的内容块 ──
            if content["type"] == "image":
                # 从 content 字典中取出相对路径，并删除原始 key（等会儿换成绝对路径）
                img_file_name = content.pop("image")
                
                # 针对特殊数据集（kling_mm）做路径前缀清理
                if "kling_mm" in dataset_root:
                    img_file_name = img_file_name.replace("created_dataset/filtered_data/", "")
                
                # 将相对路径转为绝对路径，存回 content["image"]
                content["image"] = os.path.join(dataset_root, img_file_name)
                
                # 合法性检查：如果这张图片不是 content[0]（即它前面还有内容），
                # 且前一个内容块是文本，且当前轮次是助手说的，
                # 那么该文本块里必须包含 <abs_vis_token></abs_vis_token> 占位符
                # （表示"这里有一个 latent 推理槽，对应后面这张图片"）
                # 如果没有占位符，说明数据格式不合法，直接过滤掉这个样本（返回 None）
                if j > 0 and new_step["content"][j-1]["type"] == "text" and step["role"] == "assistant":
                    if "<abs_vis_token></abs_vis_token>" not in new_step["content"][j-1]["text"]:
                        # print("[Preprocess] No <abs_vis_token> before assistant image. Discard this sample")
                        return None  # 返回 None 表示这个样本不合格，会被过滤
                
                # 如果当前轮次是助手（assistant），累计图片计数，并标记"已见过助手图片"
                if step["role"] == "assistant":
                    n_img += 1           # 图片总数 +1
                    seen_assistant_image = True   # 标记在这一轮里已经见到过图片
            
            # ── 处理文本类型的内容块 ──
            elif content["type"] == "text":
                
                # 如果当前轮次是助手说的
                if step["role"] == "assistant":
                    # 统计该文本块中 <abs_vis_token></abs_vis_token> 的数量，累加到 n_img_pad
                    n_img_pad += content['text'].count('<abs_vis_token></abs_vis_token>')
                    
                    # 合法性检查：如果文本里有 <observation> 标签，但在此之前没有见过助手图片
                    # 则说明"先有观察结论，但还没看到图"——这是不合理的数据，把 observation 标签删掉
                    if "<observation>" in content.get("text", "") and not seen_assistant_image:
                        content['text'] = content['text'].replace("<observation>", "").replace("</observation>", "")
                    
                    # 如果（清理后）文本里仍然包含 <observation>，标记"确实有观察标注"
                    if "<observation>" in content.get("text", ""):
                        seen_observation = True
                
                # 如果当前轮次是用户说的
                elif step["role"] == "user":
                    img_key = "image"
                    # 对于不是 Zebra_CoT_visual_search 和 Zebra_CoT_count 数据集的用户文本，
                    # 删除末尾多余的 "Put your final answer within \boxed{}." 提示（这些数据集已内置了该提示）
                    if 'Zebra_CoT_visual_search' not in new_step["content"][0][img_key] and \
                       'Zebra_CoT_count' not in new_step["content"][0][img_key]:
                        content["text"] = content["text"].replace("\nPut your final answer within \\boxed{}.", "")
            
            # 把修改后的 content 写回到新的轮次中
            new_step["content"][j] = content
        
        # 把修改后的轮次写回到 conversations 列表中
        conversations[i] = new_step
    
    # 把处理好的 conversations 写回到 sample 字典里
    sample["data"] = conversations
    
    # ── 最终合法性检查 1：图片数量和占位符数量必须匹配 ──
    # 每张助手图片都必须对应一个 <abs_vis_token></abs_vis_token> 占位符，数量要相等
    # 如果不相等，说明数据不完整或格式错误，过滤掉
    if n_img != n_img_pad:
        print(f"n_img ({n_img}) != num of <abs_vis_token></abs_vis_token> ({n_img_pad}), discard this sample")
        return None  # 不合格，返回 None
    
    # ── 最终合法性检查 2：必须有至少一个 <observation> 标注（除非明确允许没有）──
    # SFT Stage 2/3 的训练目标是把文字观察替换成 latent 向量，
    # 所以如果连文字观察都没有，这个样本对这些阶段毫无意义
    if not seen_observation and not allow_no_observation:
        # print("[Preprocess] No observation found in assistant responses. Discard this sample")
        return None  # 不合格，返回 None
    
    # 通过所有检查，返回处理好的样本
    return sample


def Monet_single_input_images_preprocess_function_question_only(sample, dataset_root="", cur_max=-1, id=0, rank=-1):
    """
    只保留问题部分的简化预处理函数（用于推理/评估，不需要完整答案）。
    
    与完整版的区别：
    - 只处理对话的前两轮（sample[:2]），通常是 system + user（即问题），不含助手的回答
    - 格式检查更简单，主要确保图片路径正确
    
    参数：
    - sample: 原始样本（对话列表）
    - dataset_root: 数据集根目录路径
    - cur_max: 当前最大样本数限制（用于控制加载数量，-1 表示不限制）
    - id: 当前样本的编号（用于日志/调试）
    - rank: 当前进程的分布式 rank（用于多进程数据加载时的去重）
    
    返回：
    - (conversations, cur_max)：处理好的前两轮对话，和更新后的 cur_max
    - 如果样本不合法，返回 (None, cur_max)
    """
    
    # 初始化一个空列表，用于存放处理好的前两轮对话
    conversations = []
    
    # 只遍历样本的前两轮（通常是 system prompt + user question），不处理助手回答
    for i, step in enumerate(sample[:2]):
        # 浅拷贝这一轮次，防止修改原始数据
        new_step = step.copy()
        
        # 初始化"是否见过助手图片"的标记（仅对 assistant 轮次需要追踪）
        seen_assistant_image = False if step["role"] == "assistant" else None
        
        # 遍历这一轮次 content 里的每个内容块
        for j, content in enumerate(new_step["content"]):
            
            # ── 处理图片类型的内容块 ──
            if content["type"] == "image":
                # 将图片相对路径转为绝对路径（在 content 字典里原地替换）
                content["image"] = os.path.join(dataset_root, content.pop("image"))
                
                # 合法性检查：如果图片前面有文本块，且是助手说的，
                # 文本里必须有 <abs_vis_token></abs_vis_token> 占位符
                if j > 0 and new_step["content"][j-1]["type"] == "text" and step["role"] == "assistant":
                    if "<abs_vis_token></abs_vis_token>" not in new_step["content"][j-1]["text"]:
                        return None, cur_max  # 不合法，过滤掉
                
                # 如果是助手轮次，标记"已见过助手图片"
                if step["role"] == "assistant":
                    seen_assistant_image = True
            
            # ── 处理文本类型的内容块 ──
            elif content["type"] == "text" and step["role"] == "assistant":
                # 如果助手文本里有 <observation>，但前面没有见过图片，则这个样本不合法
                if "<observation>" in content.get("text", "") and not seen_assistant_image:
                    return None, cur_max  # 不合法，过滤掉
            
            # 把修改后的 content 写回
            new_step["content"][j] = content
        
        # 把这一轮次追加到结果列表中
        conversations.append(new_step)
    
    # 返回处理好的前两轮对话，以及不变的 cur_max
    return conversations, cur_max


# 任务预处理函数的注册表（字典）
# key 是任务名称字符串，value 是对应的预处理函数
# src/utils.py 在加载数据集时，会根据任务类型查这个字典，调用对应的函数
task_preporcess_config = {
    'mm-reasoning': Monet_single_input_images_preprocess_function  # 多模态推理任务用完整版预处理
}
