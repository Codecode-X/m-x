"""
本文件的作用：RL 训练的数据加载器创建模块。

负责把 RLHF 数据集（题目+图片+答案）加载成 PyTorch DataLoader，
供 RayPPOTrainer 在训练循环中按批次获取数据。

包含一个函数：create_dataloader()，同时创建训练集和验证集的 DataLoader。

数据流程：
    原始文件（JSON/parquet）
    → RLHFDataset（加载、过滤、tokenize）
    → Subset（可选截断）
    → StatefulDataLoader（批次化、采样）
    → RayPPOTrainer.fit() 消费

StatefulDataLoader vs 标准 DataLoader：
    StatefulDataLoader 支持保存/恢复 DataLoader 的状态（用于断点续训），
    标准 DataLoader 在中断后只能从头开始遍历数据集。
"""

# 导入类型注解（Optional 用于表示可为 None 的参数）
from typing import Optional

# 导入 PyTorch
import torch
# 导入采样器：RandomSampler（随机采样）和 SequentialSampler（顺序采样）
from torch.utils.data import RandomSampler, SequentialSampler
# 导入支持断点续训的有状态 DataLoader（来自 torchdata 库）
from torchdata.stateful_dataloader import StatefulDataLoader
# 导入 HuggingFace tokenizer 和 processor 的基类（用于类型注解）
from transformers import PreTrainedTokenizer, ProcessorMixin

# 导入项目自定义的数据集类和批次化函数
from ..utils.dataset import RLHFDataset, collate_fn
# 导入数据配置类（包含 train_files、batch_size 等参数）
from .config import DataConfig


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    """
    根据配置创建训练集和验证集的 DataLoader。
    
    参数：
    - config：DataConfig 对象，包含所有数据相关配置：
        - train_files：训练数据文件路径（支持 glob 模式）
        - val_files：验证数据文件路径
        - rollout_batch_size：训练批次大小（每次生成 rollout 的题目数）
        - val_batch_size：验证批次大小（-1 表示一次性加载全部验证数据）
        - max_prompt_length：prompt 的最大 token 长度
        - shuffle：是否随机打乱训练数据
        - seed：随机种子（保证可复现性）
        - train_max_samples / val_max_samples：限制数据集大小（调试用）
    - tokenizer：文本 tokenizer（用于编码 prompt）
    - processor：多模态 processor（包含 tokenizer + 图像预处理）
    
    返回：
    - (train_dataloader, val_dataloader)：训练集和验证集的 DataLoader 元组
    """
    
    # ── 创建训练数据集 ──
    
    train_dataset = RLHFDataset(
        data_path=config.train_files,         # 训练数据文件路径
        tokenizer=tokenizer,                   # 文本 tokenizer
        processor=processor,                   # 多模态 processor（含图像预处理）
        prompt_key=config.prompt_key,          # 数据集里 prompt 字段的键名（如 "question"）
        answer_key=config.answer_key,          # 数据集里答案字段的键名（如 "answer"）
        image_key=config.image_key,            # 数据集里图片字段的键名（如 "image"）
        max_prompt_length=config.max_prompt_length,  # 超过此长度的 prompt 会被截断或过滤
        truncation="right",                    # 从右边截断（保留左边的 prompt 内容）
        format_prompt=config.format_prompt,    # 可选：对 prompt 做额外格式化处理
        min_pixels=config.min_pixels,          # 图片最小像素数（过小的图会被过滤）
        max_pixels=config.max_pixels,          # 图片最大像素数（防止超大图 OOM）
        filter_overlong_and_invalid_prompts=config.filter_overlong_and_invalid_prompts,  # 是否过滤超长或无效 prompt
    )
    
    # 如果配置了 train_max_samples，截断数据集大小（常用于快速调试）
    if config.train_max_samples is not None:
        # 取 train_max_samples 和实际数据集大小中的较小值
        max_samples = min(config.train_max_samples, len(train_dataset))
        indices = list(range(len(train_dataset)))
        # 用 Subset 截取前 max_samples 个样本
        train_dataset = torch.utils.data.Subset(train_dataset, indices[:max_samples])
    
    # ── 创建训练采样器 ──
    
    if config.shuffle:
        # 随机采样（训练时推荐）：使用固定种子确保可复现性
        # 注意：使用独立的 Generator 而不是全局随机种子，避免影响其他地方的随机性
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        # 顺序采样（调试时使用，每次遍历数据顺序相同）
        sampler = SequentialSampler(data_source=train_dataset)
    
    # ── 创建训练 DataLoader ──
    
    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=config.rollout_batch_size,      # 每批次的题目数量（每道题会生成多个 rollout）
        sampler=sampler,                            # 上面创建的采样器
        num_workers=getattr(config, "dataloader_num_workers", 4),  # 数据加载的工作进程数（默认4）
        collate_fn=collate_fn,                      # 把一批样本拼成 batch 的函数
        pin_memory=False,                           # 不使用 pinned memory（多 GPU 时通常关闭）
        drop_last=True,                             # 丢弃最后一个不完整的批次（保证批次大小一致）
    )
    
    # ── 创建验证数据集 ──
    
    # 验证数据集的创建方式与训练集相同
    val_dataset = RLHFDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_and_invalid_prompts=config.filter_overlong_and_invalid_prompts,
    )
    
    # 如果配置了 val_max_samples，截断验证集大小
    if config.val_max_samples is not None:
        max_samples = min(config.val_max_samples, len(val_dataset))
        indices = list(range(len(val_dataset)))
        val_dataset = torch.utils.data.Subset(val_dataset, indices[:max_samples])
    
    # ── 创建验证 DataLoader ──
    
    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        # 验证时批次大小：val_batch_size=-1 表示一次性加载全部验证数据（方便计算整体指标）
        batch_size=len(val_dataset) if config.val_batch_size == -1 else config.val_batch_size,
        shuffle=False,          # 验证时不需要打乱（保证评估结果可比较）
        num_workers=getattr(config, "dataloader_num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,        # 验证时不丢弃最后一批（评估需要全部数据）
    )
    
    # 检查数据加载器不为空（至少有 1 个批次）
    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    
    # 打印数据集大小信息（便于确认数据加载是否正常）
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    
    return train_dataloader, val_dataloader
