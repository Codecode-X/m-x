"""
本文件的作用：定义 SFT 三个阶段各自的 Trainer 类和两个辅助工具函数。

背景知识：
- trl 库的 SFTTrainer 是对 HuggingFace Trainer 的封装，专门用于监督微调（SFT）
- Monet 的 SFT 训练分三个阶段，每个阶段的损失计算逻辑不同，所以各自继承 SFTTrainer 并重写 compute_loss

三个 Trainer 类：
1. CustomTrainerSFT_STAGE1：Stage 1 热身阶段，只用交叉熵（CE）损失
2. CustomTrainerSFT_STAGE2：Stage 2 引入 latent token，CE + 对齐损失
3. CustomTrainerSFT_STAGE3：Stage 3 latent 蒸馏精炼，CE + latent 对齐损失

两个辅助函数：
- compute_latents_only_loss：计算只流经 latent 向量的梯度代理损失
- load_offline_tensor：从磁盘加载预计算好的 teacher 表示
"""

# 导入 trl 库的 SFTTrainer（监督微调训练器）和 SFTConfig（训练配置）
from trl import SFTTrainer, SFTConfig
# 导入类型注解
from typing import Optional
# 导入日志工具
import logging
# 导入 PyTorch，用于张量操作和自动求导
import torch
# 导入操作系统、CSV、PyTorch、日期时间工具
import os, csv, torch, datetime
# 导入垃圾回收工具（用于手动触发显存释放）
import gc
# 导入 NumPy（此处未直接使用，可能是遗留导入）
import numpy as np
# 导入数学工具（此处未直接使用）
import math
# 导入计时工具（此处未直接使用）
from time import time


def compute_latents_only_loss(latents, loss_for_latents):
    """
    计算一个特殊的"代理损失"：让梯度只流经 latent 向量，不流经模型参数。
    
    背景：
    在 SFT Stage 2 中，对齐损失（alignment_loss）需要优化 latent 向量的方向，
    但我们不想让对齐损失的梯度直接传给模型的线性层（那会破坏 CE 损失的优化方向）。
    
    解决方案（代理梯度技巧）：
    1. 先用 autograd.grad 计算损失对 latent 向量的梯度（grad）
    2. 构造一个"代理损失" = sum(latent × grad.detach())
       - 这个代理损失的梯度 = grad（与原始损失对 latent 的梯度相同）
       - 但因为 grad 已经 detach，梯度不会继续往前传给模型参数
    3. 把这个代理损失加入总损失，就能只更新 latent 向量，不影响模型参数
    
    参数：
    - latents: latent 向量列表（嵌套结构，每个元素是 Tensor）
    - loss_for_latents: 需要"只通过 latent"反向传播的损失值（标量张量）
    
    返回：
    - proxy_loss: 代理损失（标量张量），可以直接加入总损失参与反向传播
    """
    
    def _flatten_tensors(x):
        """
        递归地把嵌套的列表/元组结构中的所有 Tensor 展平为一维列表。
        例如：[[t1, t2], t3] → [t1, t2, t3]
        """
        if isinstance(x, (list, tuple)):
            out = []
            for y in x:
                # 递归处理每个子元素
                out.extend(_flatten_tensors(y))
            return out
        # 如果不是列表/元组，说明是 Tensor 本身，直接包装成列表返回
        return [x]
    
    # 把嵌套的 latent 向量结构展平为普通列表
    ce_vec_list = _flatten_tensors(latents)
    
    # 使用 torch.autograd.grad 手动计算梯度
    # outputs=loss_for_latents：要求导的损失（目标函数）
    # inputs=ce_vec_list：对哪些变量求导（即 latent 向量）
    # retain_graph=True：保留计算图（后续还会做第二次 backward）
    # create_graph=False：不构建高阶梯度图（只需要一阶梯度）
    # allow_unused=True：如果某个 latent 向量没有参与计算，允许其梯度为 None
    grads = torch.autograd.grad(
        outputs=loss_for_latents,
        inputs=ce_vec_list,
        retain_graph=True,
        create_graph=False,
        allow_unused=True
    )
    
    # 处理梯度中可能存在的 None（表示该 latent 未参与计算，梯度为零）
    safe_grads = []
    for v, g in zip(ce_vec_list, grads):
        if g is None:
            # 用与 v 同形状、同设备、同数据类型的零张量代替
            g = torch.zeros_like(v)
        # detach() 切断梯度传播链：这样 proxy_loss 的梯度只能到达 v 这一层，不会继续往前
        safe_grads.append(g.detach())
    
    # 构造代理损失：sum_i(v_i · g_i)
    # 对每个 latent 向量，计算 (向量值 × 梯度).sum()，然后把所有 latent 的结果求和
    # 关键：g_i 已经 detach，所以对这个 proxy_loss 做 backward 时，
    # 梯度只会传给 v_i（latent 向量），不会继续向前传给生成 v_i 的模型参数
    proxy_loss = torch.stack([(v * g).sum() for v, g in zip(ce_vec_list, safe_grads)]).sum()
    
    return proxy_loss


def load_offline_tensor(tensor_dir, batch_metadata, alignment_layer="all_layers", rep_type="rep", align_poss="obs"):
    """
    从磁盘加载预计算好的 teacher 表示（隐状态向量）。
    
    在 SFT Stage 2 中，用于加载 Stage 1 模型预计算的 observation token 隐状态；
    在 SFT Stage 3 中，用于加载 Stage 2 模型预计算的 latent token 目标隐状态。
    
    文件命名规则：
    - obs 模式：{rep_type}_{alignment_layer}_{dataset_name}_{sample_id}.pt
    - latent_end 模式：{rep_type}_latent_end_{alignment_layer}_{dataset_name}_{sample_id}.pt
    
    参数：
    - tensor_dir: 预计算张量的存储目录
    - batch_metadata: 当前批次每个样本的元信息列表（包含 dataset_name 和 sample_id）
    - alignment_layer: 对齐哪些层的隐状态（"all_layers" 或具体层号）
    - rep_type: 表示类型（"rep" 表示隐状态，"latent" 表示 latent 向量）
    - align_poss: 对齐位置类型（"obs" 表示 observation token，"latent_end" 表示 latent token）
    
    返回：
    - teacher_reps: 预计算好的 teacher 隐状态列表（每个元素是一个 Tensor，对应 batch 里的一个样本）
    """
    
    # 初始化返回值（如果加载失败则返回 None）
    teacher_reps = None
    # 存储加载结果的列表（batch 里每个样本对应一个 Tensor）
    latents_list = []
    
    # 遍历 batch 里每个样本的元信息
    for metadata in batch_metadata:
        # 从元信息中取出数据集名称（如 "Visual_CoT"、"CogCoM" 等）
        dataset_name = metadata['dataset_name']
        # 从元信息中取出样本 ID（唯一标识这个样本）
        sample_id = metadata['sample_id']
        
        # 拼接元信息字符串，作为文件名的中间部分
        metadata_info = f"{alignment_layer}_{dataset_name}_{sample_id}"
        
        # 根据对齐位置类型，构造完整的文件名
        if align_poss == 'obs':
            # observation token 对齐：文件名格式为 rep_{info}.pt
            metadata_str = f"{rep_type}_{metadata_info}.pt"
        elif align_poss == 'latent_end':
            # latent token 对齐：文件名格式为 rep_latent_end_{info}.pt
            metadata_str = f"{rep_type}_latent_end_{metadata_info}.pt"
        
        # 拼接完整的文件路径
        path = os.path.join(tensor_dir, metadata_str)
        
        # 检查文件是否存在，不存在则报错
        if not os.path.isfile(path):
            latents_list = []
            raise RuntimeError(f"Missing teacher latent file: {path}")
        
        # 从磁盘加载预计算的张量（map_location='cpu' 先加载到 CPU，后续会移到 GPU）
        data = torch.load(path, map_location='cpu')
        
        # 提取 'latent' 字段（预计算脚本保存时用的 key），并 detach 脱离计算图
        latents_list.append(data['latent'].detach())
    
    # 检查加载的数量是否和 batch 大小匹配
    if batch_metadata is not None and len(latents_list) == len(batch_metadata):
        teacher_reps = latents_list  # 数量匹配，赋值给返回值
    
    return teacher_reps


# ══════════════════════════════════════════════════════════════════════
# SFT Stage 1 Trainer：热身阶段（只用 CE 损失）
# ══════════════════════════════════════════════════════════════════════

class CustomTrainerSFT_STAGE1(SFTTrainer):
    """
    SFT Stage 1 的自定义训练器。
    
    Stage 1 目标：让模型学会"先看图观察，再推理回答"的思维格式。
    损失：只有交叉熵（CE）损失，对 observation token 位置加权（ce_emphasize_factor 倍）。
    数据：含 <observation> 文字标注的完整对话（teacher 数据）。
    """
    
    def __init__(self, *args, **kwargs):
        # 从 kwargs 中取出并保存实验名称（自定义参数，不传给父类）
        self.exp_name = kwargs.pop('exp_name')
        
        # 兼容处理：如果用的是旧版 trl（用 'tokenizer' 参数），
        # 把它改为新版 trl 要求的 'processing_class' 参数名
        if 'processing_class' not in kwargs and 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        
        # 调用父类 SFTTrainer 的初始化
        super().__init__(*args, **kwargs)
        
        # 追踪 observation token 的预测准确率（用于监控模型是否学会了预测观察内容）
        self.observation_token_acc = 0.       # 累计准确率
        self.observation_token_acc_step = 0   # 累计步数
        
        # 追踪训练过程中的 CE 损失（用于日志记录）
        self.teacher_ce_cum = 0.0       # 累计 CE 损失值
        self.teacher_ce_steps = 0       # 累计步数

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Stage 1 的损失计算：只计算 CE 损失，对 observation token 位置加权。
        
        关键细节：
        - latent_mode=False：Stage 1 没有 latent 推理，使用标准前向传播
        - 使用 teacher_input_ids（含 <observation> 文字标注的数据）
        - ce_emphasize_poss 指定 observation token 的位置，这些位置的 CE 损失乘以 ce_emphasize_factor
        """
        
        # Stage 1 不使用 latent 推理模式（用普通前向传播）
        inputs['latent_mode'] = False
        
        # 使用 teacher 数据（含 <observation> 文字标注）作为输入
        # teacher_input_ids 是经过完整处理的 token ID 序列（含 observation 文字）
        inputs['input_ids'] = inputs['teacher_input_ids']
        inputs['attention_mask'] = inputs['teacher_attention_mask']
        inputs['pixel_values'] = inputs['teacher_pixel_values']
        inputs['image_grid_thw'] = inputs['teacher_image_grid_thw']
        inputs['labels'] = inputs['teacher_labels']
        
        # 指定 observation token 的位置（这些位置的 CE 损失会被加权放大）
        inputs['ce_emphasize_poss'] = inputs['teacher_observation_poss']
        
        # 设置加权倍数（从训练配置中取，通常大于 1）
        inputs['ce_emphasize_factor'] = self.args.ce_emphasize_factor
        
        # 指定只计算 CE 损失
        inputs['loss_type'] = ['ce']
        
        # 计算是否需要统计 observation token 的预测准确率
        inputs['compute_emphasize_acc'] = True
        
        # 调用父类的 compute_loss，执行实际的前向传播和损失计算
        (teacher_ce_loss, teacher_outputs) = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch
        )
        
        # 累积 CE 损失值（用于后续日志记录）
        self.teacher_ce_cum += teacher_ce_loss.item()
        self.teacher_ce_steps += 1
        
        # 如果模型输出了 observation token 准确率，累积保存
        if getattr(teacher_outputs, 'mean_emphasize_acc', None) is not None:
            self.observation_token_acc += getattr(teacher_outputs, 'mean_emphasize_acc')
            self.observation_token_acc_step += 1
        
        # 主动释放中间输出，减少显存占用
        del teacher_outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        # 根据调用方需求，返回 (损失, 输出) 或只返回损失
        return (teacher_ce_loss, None) if return_outputs else teacher_ce_loss

    def on_epoch_end(self):
        """每个 epoch 结束时的回调（直接调用父类逻辑）"""
        return super().on_epoch_end()

    def log(self, logs: dict, start_time: float | None = None):
        """
        重写日志记录函数，把自定义的统计指标（CE 损失、observation 准确率）
        加入到标准日志中，一并输出到 swanlab/终端等。
        """
        # 复制原始日志字典（避免修改原始对象）
        merged = dict(logs)
        
        # 如果有累积的 CE 损失，计算平均值并加入日志
        if self.teacher_ce_steps > 0:
            merged["student_ce_loss"] = round(self.teacher_ce_cum / max(1, self.teacher_ce_steps), 6)
            # 重置累积计数器
            self.teacher_ce_cum = 0.0
            self.teacher_ce_steps = 0
        
        # 如果有累积的 observation 准确率，计算平均值并加入日志
        if self.observation_token_acc_step > 0:
            merged["observation_token_acc"] = round(self.observation_token_acc / max(1, self.observation_token_acc_step), 6)
            # 重置累积计数器
            self.observation_token_acc = 0.
            self.observation_token_acc_step = 0
        
        # 调用父类 log，实际写入 swanlab/终端/TensorBoard 等
        return super().log(merged, start_time)


# ══════════════════════════════════════════════════════════════════════
# SFT Stage 2 Trainer：引入 Latent Token（CE + 对齐损失）
# ══════════════════════════════════════════════════════════════════════

class CustomTrainerSFT_STAGE2(SFTTrainer):
    """
    SFT Stage 2 的自定义训练器。
    
    Stage 2 目标：用 latent 向量替代文字 observation，并让 latent 向量的
                  语义方向向 Stage 1 模型的 observation token 隐状态对齐。
    
    损失：CE 损失 + 对齐损失（alignment_loss）
    
    训练过程（两步 forward）：
    步骤1：latent_mode=True 的 forward（自回归生成 latent 向量）
           获取 ce_patch_pos（latent token 的位置）和 ce_patch_vec（latent 向量值）
    步骤2：latent_mode=False 的 forward（把 latent 向量插入序列，做标准前向传播）
           同时计算 CE 损失和对齐损失
    """
    
    def __init__(self, *args, **kwargs):
        # 保存实验名称
        self.exp_name = kwargs.pop('exp_name')
        
        # 兼容旧版 trl 的 'tokenizer' 参数
        if 'processing_class' not in kwargs and 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        
        # 调用父类初始化
        super().__init__(*args, **kwargs)
        
        # 从训练配置中取出 CE 加权因子
        self.ce_emphasize_factor = self.args.ce_emphasize_factor
        
        # 追踪各类损失和准确率（用于日志）
        self.teacher_ce_loss_cum = 0.0          # 累计 CE 损失
        self.teacher_ce_loss_steps = 0           # CE 损失步数
        self.observation_token_acc = 0.          # 累计 observation 准确率
        self.observation_token_acc_step = 0      # observation 准确率步数
        self.alignment_loss_cum = 0.             # 累计对齐损失
        self.alignment_loss_steps = 0            # 对齐损失步数

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Stage 2 的损失计算：两步 forward。
        
        第一步：latent forward（自回归生成 latent 向量）
        第二步：CE forward（把 latent 向量插入序列，计算最终损失）
        """
        
        # ──────────────────────────────────────────────────────────────
        # 第一步：latent forward
        # 目的：自回归地生成每个 latent token 位置的向量
        # ──────────────────────────────────────────────────────────────
        
        # 开启 latent 模式（模型会对每个 latent token 单步 forward）
        inputs['latent_mode'] = True
        # 第一步只收集 latent 向量，不计算任何损失
        inputs['loss_type'] = []
        
        # latent forward 需要用 KV Cache（use_cache=True），
        # 而 gradient_checkpointing 与 use_cache 不兼容，所以必须先禁用
        model.gradient_checkpointing_disable()
        
        # 执行 latent 前向传播
        outputs = model(**inputs, return_dict=True, output_hidden_states=False)
        
        # outputs.ce_patch_pos：latent token 在序列中的位置列表（List[List[int]]）
        # outputs.ce_patch_vec：对应的 latent 向量（List[Tensor]）
        
        # ──────────────────────────────────────────────────────────────
        # 第二步：CE forward（标准前向传播，计算损失）
        # ──────────────────────────────────────────────────────────────
        
        # 重新开启梯度 checkpointing（节省显存，use_reentrant=False 是新版推荐）
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        # 关闭 latent 模式（使用标准前向传播）
        inputs['latent_mode'] = False
        
        # 把第一步得到的 latent 位置和向量传给第二步
        # 第二步的模型会把这些 latent 向量"嵌入"到序列里对应的位置
        inputs['ce_patch_pos'] = outputs.ce_patch_pos
        inputs['ce_patch_vec'] = outputs.ce_patch_vec
        
        # 指定 observation token 的位置（CE 损失加权位置）
        inputs['ce_emphasize_poss'] = inputs['observation_poss']
        # 设置加权倍数
        inputs['ce_emphasize_factor'] = self.ce_emphasize_factor
        
        # 指定计算 CE 损失；如果对齐权重不为 0，还要计算对齐损失
        inputs['loss_type'] = ['ce']
        if self.args.alignment_weight != 0:
            inputs['loss_type'].append('alignment')
        
        # 统计 observation 预测准确率
        inputs['compute_emphasize_acc'] = True
        
        # 移除可能影响 gradient checkpointing 重计算的参数
        inputs.pop('output_attentions', None)
        inputs.pop('attn_analysis', None)
        
        # 如果需要计算对齐损失，从磁盘加载 teacher 的 observation 隐状态
        if self.args.alignment_weight != 0:
            teacher_reps = load_offline_tensor(
                self.args.teacher_reps_dir,          # 预计算 teacher 表示的目录
                batch_metadata=inputs['metadata'],   # 当前 batch 的元信息
                alignment_layer=self.args.alignment_layer  # 对齐哪些层
            )
            # 告诉模型在 observation token 位置做对齐
            inputs['alignment_poss'] = inputs['observation_poss']
            # 传入 teacher 隐状态，用于计算对齐损失
            inputs['teacher_hidden_states_for_alignment'] = teacher_reps
        
        # 调用父类 compute_loss 执行第二步前向传播和损失计算
        teacher_ce_loss, teacher_output = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch
        )
        
        # 从输出的 loss_dict 中取出对齐损失（如果没有，默认为 0）
        alignment_loss = teacher_output.loss_dict.get('alignment', torch.tensor(0.0))
        
        # 计算最终总损失
        if self.args.emphasize_latent_weight != 0.0 and alignment_loss.item() != 0.0:
            # 方式一（推荐）：用代理梯度技巧，让对齐损失只反向传播到 latent 向量
            # emphasize_latent_weight 控制这部分的权重
            latent_only_loss = compute_latents_only_loss(
                outputs.ce_patch_vec,                            # latent 向量
                self.args.alignment_weight * alignment_loss      # 原始对齐损失
            )
            loss = self.args.emphasize_latent_weight * latent_only_loss + teacher_ce_loss
        else:
            # 方式二（简单）：直接把对齐损失加到总损失里（梯度会流经所有参数）
            loss = teacher_ce_loss + self.args.alignment_weight * alignment_loss
        
        # 累积 observation 准确率（用于日志）
        if getattr(teacher_output, 'mean_emphasize_acc', None) is not None:
            self.observation_token_acc += getattr(teacher_output, 'mean_emphasize_acc')
            self.observation_token_acc_step += 1
        
        # 累积 CE 损失和对齐损失（用于日志）
        self.teacher_ce_loss_cum += teacher_ce_loss.item()
        self.teacher_ce_loss_steps += 1
        self.alignment_loss_cum += alignment_loss.item()
        self.alignment_loss_steps += 1
        
        # 每 50 步清理一次显存（避免 OOM，但不频繁 sync 影响速度）
        step = int(getattr(self.state, 'global_step', 0) or 0)
        if step % 50 == 0:
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
        
        # 返回损失
        return (loss, None) if return_outputs else loss

    def on_epoch_end(self):
        """每个 epoch 结束时的回调（调用父类）"""
        return super().on_epoch_end()

    def log(self, logs: dict, start_time: float | None = None):
        """
        重写日志记录函数，加入自定义指标（CE 损失、对齐损失、observation 准确率）。
        """
        merged = dict(logs)
        
        # 输出累计 CE 损失的平均值
        if self.teacher_ce_loss_cum > 0:
            merged["teacher_ce_loss"] = round(self.teacher_ce_loss_cum / max(1, self.teacher_ce_loss_steps), 6)
            self.teacher_ce_loss_cum = 0.0
            self.teacher_ce_loss_steps = 0
        
        # 输出累计对齐损失的平均值
        if self.alignment_loss_cum > 0:
            merged[f'alignment_loss'] = round(self.alignment_loss_cum / max(1, self.alignment_loss_steps), 6)
            self.alignment_loss_cum = 0.0
            self.alignment_loss_steps = 0
        
        # 输出累计 observation 准确率的平均值
        if self.observation_token_acc_step > 0:
            merged["observation_token_acc"] = round(self.observation_token_acc / max(1, self.observation_token_acc_step), 6)
            self.observation_token_acc = 0.
            self.observation_token_acc_step = 0
        
        # 调用父类 log 完成实际写入
        return super().log(merged, start_time)


# ══════════════════════════════════════════════════════════════════════
# SFT Stage 3 Trainer：Latent 蒸馏精炼（CE + latent 对齐损失）
# ══════════════════════════════════════════════════════════════════════

class CustomTrainerSFT_STAGE3(SFTTrainer):
    """
    SFT Stage 3 的自定义训练器。
    
    Stage 3 目标：进一步精炼 latent 向量的质量，让其向 Stage 2 模型生成的
                  "教师 latent 向量"对齐（知识蒸馏）。
    
    与 Stage 2 的区别：
    - Stage 2 对齐的是 observation token（文字）的隐状态
    - Stage 3 对齐的是 latent token 本身的隐状态（更直接的 latent 蒸馏）
    
    损失：CE 损失 + latent 对齐损失
    """
    
    def __init__(self, *args, **kwargs):
        # 保存实验名称
        self.exp_name = kwargs.pop('exp_name')
        
        # 调用父类初始化（注意：Stage 3 没有做旧版 tokenizer 参数兼容处理）
        super().__init__(*args, **kwargs)
        
        # 对齐损失的权重系数
        self.alignment_weight = self.args.alignment_weight
        
        # CE 加权因子（对 observation 位置的 CE 损失放大倍数）
        self.ce_emphasize_factor: float = float(getattr(self.args, 'ce_emphasize_factor', 1.0))
        
        # 预计算的 teacher latent 向量存储目录（Stage 3 必须指定）
        self.teacher_latent_dir = getattr(self.args, 'teacher_latent_dir', None)
        if not self.teacher_latent_dir:
            raise ValueError("teacher_latent_dir must be specified for SFT Stage 3")
        
        # 追踪各类指标（用于日志）
        self.observation_token_acc = 0.       # 累计 observation 准确率
        self.observation_token_acc_step = 0   # 步数计数器
        self.al_loss_cum = 0.0                # 累计对齐损失
        self.al_steps = 0                     # 对齐损失步数
        self.student_ce_loss_cum = 0.0        # 累计 student CE 损失
        self.student_ce_loss_steps = 0        # CE 损失步数

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Stage 3 的损失计算：latent forward → CE forward（含对齐损失）。
        
        与 Stage 2 的关键区别：
        - 对齐目标是 teacher latent 向量（不是 teacher observation 隐状态）
        - 同时对 student 和 teacher 数据做双路计算
        """
        
        # 从磁盘加载 Stage 2 预计算的 teacher latent 向量
        # rep_type="latent" 表示加载的是 latent 向量而不是 observation 隐状态
        teacher_latents = load_offline_tensor(
            self.teacher_latent_dir,
            batch_metadata=inputs['metadata'],
            alignment_layer=self.args.alignment_layer,
            rep_type="latent"
        )
        
        # ──────────────────────────────────────────────────────────────
        # 第一步：student latent forward（自回归生成 student latent 向量）
        # ──────────────────────────────────────────────────────────────
        
        # 开启 latent 模式
        inputs['latent_mode'] = True
        
        # 使用 student 数据（与 Stage 2 的格式可能略有不同，用 student_ 前缀区分）
        inputs['input_ids'] = inputs['student_input_ids']
        inputs['attention_mask'] = inputs['student_attention_mask']
        inputs['pixel_values'] = inputs['student_pixel_values']
        inputs['image_grid_thw'] = inputs['student_image_grid_thw']
        
        # Stage 3 的 latent forward 不计算标签损失，先把 labels 删掉
        if 'labels' in inputs:
            inputs.pop('labels')
        
        # 指定对齐的位置（student latent token 的位置）
        inputs['alignment_poss'] = inputs['student_alignment_poss']
        
        # 传入 teacher latent 向量（用于 latent forward 内部计算对齐损失）
        inputs['teacher_hidden_states_for_alignment'] = teacher_latents
        
        # 禁用 gradient checkpointing（因为 latent forward 用 KV Cache）
        model.gradient_checkpointing_disable()
        
        # 不计算任何损失（只收集 latent 向量）
        inputs['loss_type'] = []
        inputs['output_hidden_states'] = False
        
        # 执行 student latent 前向传播
        student_outputs_latent = model(**inputs)
        # 此时 student_outputs_latent.ce_patch_pos 和 .ce_patch_vec 包含了生成的 latent 信息
        
        # ──────────────────────────────────────────────────────────────
        # 第二步：student CE forward（计算最终损失）
        # ──────────────────────────────────────────────────────────────
        
        # 关闭 latent 模式，进入标准前向传播
        inputs['latent_mode'] = False
        
        # 传入 student 标签（用于计算 CE 损失）
        inputs['labels'] = inputs['student_labels']
        
        # 把第一步生成的 latent 向量注入序列
        inputs['ce_patch_pos'] = student_outputs_latent.ce_patch_pos
        inputs['ce_patch_vec'] = student_outputs_latent.ce_patch_vec
        
        # 设置 observation 位置的 CE 损失加权
        inputs['ce_emphasize_factor'] = self.ce_emphasize_factor
        inputs['ce_emphasize_poss'] = inputs['observation_poss']
        
        # 重新开启 gradient checkpointing（第二步做完整反向传播）
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        # 同时计算 CE 损失和对齐损失
        inputs['loss_type'] = ['ce', 'alignment']
        inputs['compute_emphasize_acc'] = True
        
        # 如果有 4D attention mask，替换为 student 专用的版本
        if 'student_attention_mask_4d' in inputs:
            inputs['attention_mask_4d'] = inputs.pop('student_attention_mask_4d')
        
        # 调用父类 compute_loss 执行第二步
        (student_ce_loss, student_outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        
        # 收集 observation 预测准确率
        if getattr(student_outputs, 'mean_emphasize_acc', None) is not None:
            self.observation_token_acc += getattr(student_outputs, 'mean_emphasize_acc')
            self.observation_token_acc_step += 1
        
        # 从 loss_dict 中取出对齐损失
        alignment_loss = student_outputs.loss_dict['alignment']
        
        # 最终总损失 = CE 损失 + alignment_weight × 对齐损失
        loss = student_ce_loss + self.alignment_weight * alignment_loss
        outputs_student_loss = student_ce_loss.item()
        
        # 释放不再需要的中间输出，减少显存占用
        del student_outputs
        
        # 每 20 步清理一次显存
        step = int(getattr(self.state, 'global_step', 0) or 0)
        if step > 0 and (step % 20 == 0):
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
        
        # 累积指标用于日志
        self.al_loss_cum += float(alignment_loss.detach().item())
        self.al_steps += 1
        self.student_ce_loss_cum += outputs_student_loss
        self.student_ce_loss_steps += 1
        
        # 返回损失
        return (loss, None) if return_outputs else loss

    def log(self, logs: dict, start_time: float | None = None):
        """
        重写日志记录函数，加入 Stage 3 的自定义指标。
        """
        merged = dict(logs)
        
        # 输出累计对齐损失
        if self.al_steps > 0:
            merged["alignment_loss"] = round(self.al_loss_cum / max(1, self.al_steps), 6)
            self.al_loss_cum = 0.0
            self.al_steps = 0
        
        # 输出累计 student CE 损失
        if self.student_ce_loss_steps > 0:
            merged["student_ce_loss"] = round(self.student_ce_loss_cum / max(1, self.student_ce_loss_steps), 6)
            self.student_ce_loss_cum = 0.0
            self.student_ce_loss_steps = 0
        
        # 输出累计 observation 准确率
        if self.observation_token_acc_step > 0:
            merged["observation_token_acc"] = round(self.observation_token_acc / max(1, self.observation_token_acc_step), 6)
            self.observation_token_acc = 0.
            self.observation_token_acc_step = 0
        
        # 调用父类 log
        return super().log(merged, start_time)
