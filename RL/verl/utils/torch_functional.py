"""
本文件的作用：提供 RL/verl 框架中所有 PyTorch 相关的工具函数和自定义优化器。

这是整个 verl 框架的"瑞士军刀"，被几乎所有其他模块导入。

主要内容：
1. 对数概率计算（log_probs_from_logits）
   - 支持 Flash Attention 2 的高效交叉熵版本
   - 普通 PyTorch 版本（fp32 上转精度计算）

2. Masked 统计工具（masked_mean / masked_var / masked_whiten）
   - 支持掩码（mask）的均值/方差/规范化
   - 在 RL 训练中，EOS 之后的 padding 位置不应参与统计

3. 序列生成工具
   - get_response_mask：根据 EOS 生成回答部分的 attention mask
   - pad_2d_list_to_length：把变长的 token ID 列表 padding 成等长张量
   - pad_sequence_to_length：对单个张量做 padding
   - postprocess_data：对 (input_ids, attention_mask, position_ids) 三元组做 padding/截断

4. 学习率调度器
   - get_constant_schedule_with_warmup：线性 warmup + 常数学习率

5. AnyPrecisionAdamW 自定义优化器
   - 可以指定 momentum、variance 的精度（默认 bfloat16）
   - 支持 Kahan 求和（高精度权重更新，可完全替代 fp32 优化器状态）
   - 来源：Meta LLaMA Cookbook
"""

# 类型注解
from typing import List, Literal, Optional, Tuple, Union

# PyTorch 核心
import torch
# 分布式通信（导入但本文件不直接使用，其他模块通过 from . import torch_functional as VF 使用）
import torch.distributed
# PyTorch 函数式接口（交叉熵等）
import torch.nn.functional as F
# 学习率调度器基类
from torch.optim.lr_scheduler import LambdaLR

# 精度类型转换工具（"bfloat16" ↔ torch.bfloat16）
from .torch_dtypes import PrecisionType


# ── 尝试导入 Flash Attention 的高效交叉熵实现 ──
try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    # 没有安装 flash-attn，回退到标准 PyTorch 实现
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════
# 对数概率计算
# ══════════════════════════════════════════════════════════════════════

@torch.compiler.disable()
def log_probs_from_logits_flash_attn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    使用 Flash Attention 的高效交叉熵实现计算对数概率。
    
    优势：
    - 在 Triton 内核中直接计算，不需要把 logits 转为 fp32（节省显存和时间）
    - 支持 inplace_backward（节省梯度计算的临时内存）
    
    @torch.compiler.disable()：禁用 torch.compile 对这个函数的编译（避免与 Triton 冲突）
    
    要求：flash-attn >= 2.4.3（cross_entropy_loss 才返回 (losses, z_losses) 元组格式）
    
    参数：
    - logits：模型输出 logits，形状 (batch_size × seqlen, vocab_size)
    - labels：目标 token ID，形状 (batch_size × seqlen,)
    
    返回：
    - 对数概率（负交叉熵），形状 (batch_size × seqlen,)
    """
    output = cross_entropy_loss(logits, labels, inplace_backward=True)
    
    # flash-attn 2.4.3+ 返回 (losses, z_losses) 元组
    if not isinstance(output, tuple):
        raise ValueError(
            "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
        )
    
    # output[0] 是交叉熵损失（正值），取负号得到对数概率
    return -output[0]


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    根据 logits 计算 labels 位置的对数概率。
    
    这是计算 policy log probability 的核心函数：
    - 给定模型输出的 logits（每个位置对所有词的预测分数）
    - 和实际生成的 token IDs（labels）
    - 计算这些 token 的对数概率（用于 PPO ratio 和 KL 散度计算）
    
    公式：log_prob[i] = log(softmax(logits[i])[labels[i]])
                      = logits[i][labels[i]] - log(sum(exp(logits[i])))
    
    优先使用 Flash Attention 的高效实现，没有时用标准 PyTorch（转 fp32 避免数值问题）。
    
    参数：
    - logits：模型输出，形状 (batch_size, seqlen, vocab_size)
    - labels：目标 token ID，形状 (batch_size, seqlen)
    
    返回：
    - 每个位置的对数概率，形状 (batch_size, seqlen)
    """
    batch_dim = logits.shape[:-1]  # (batch_size, seqlen)
    vocab_dim = logits.shape[-1]   # vocab_size
    
    # 展平前两维：(batch_size × seqlen, vocab_size)
    logits = logits.contiguous().view(-1, vocab_dim)
    labels = labels.contiguous().view(-1)
    
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        # Flash Attention 高效实现（不需要转 fp32）
        output = log_probs_from_logits_flash_attn(logits, labels)
    else:
        # 标准实现：转 fp32 提高数值稳定性
        # F.cross_entropy 计算的是负对数似然（正值），取负得到对数概率
        output = F.cross_entropy(logits.float(), labels, reduction="none")
    
    # 还原形状：(batch_size, seqlen)
    return output.view(*batch_dim)


# ══════════════════════════════════════════════════════════════════════
# Masked 统计工具
# ══════════════════════════════════════════════════════════════════════

def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    计算带掩码的均值（只对 mask=1 的位置计算均值）。
    
    用途：在计算奖励、损失的平均值时，忽略 EOS 之后的 padding 位置。
    
    公式：masked_mean = sum(values × mask) / sum(mask)
    
    参数：
    - values：要计算均值的张量
    - mask：0/1 掩码（0 表示忽略该位置，1 表示计入均值）
    - dim：在哪个维度计算均值（None 表示对所有元素）
    - eps：防止除以零的小量
    
    返回：均值（标量或降维后的张量）
    """
    return (values * mask).sum(dim=dim) / (mask.sum(dim=dim) + eps)


def masked_var(
    values: torch.Tensor,
    mask: torch.Tensor,
    unbiased: bool = True
) -> torch.Tensor:
    """
    计算带掩码的方差（只对 mask=1 的位置计算方差）。
    
    支持 Bessel 修正（unbiased=True 时用 n/(n-1) 因子修正偏差）。
    
    参数：
    - values：要计算方差的张量
    - mask：0/1 掩码
    - unbiased：是否使用无偏估计（True = 除以 n-1，False = 除以 n）
    
    返回：方差（标量）
    """
    mean = masked_mean(values, mask)
    centered_values = values - mean       # 去均值
    variance = masked_mean(centered_values**2, mask)  # 均方
    
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum <= 1:
            # 样本数 ≤ 1 时无法做 Bessel 修正，打印警告并返回有偏估计
            print("The sum of the mask is less than one, which can cause a division by zero.")
            return variance
        
        # Bessel 修正：n/(n-1)
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    
    return variance


def masked_whiten(
    values: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    对带掩码的值做 whitening（零均值、单位方差的规范化）。
    
    用途：在 GRPO/REINFORCE++ 中规范化 advantage 值，使训练更稳定。
    
    公式：whitened = (values - mean) / sqrt(var + eps)
    
    只在 mask=1 的位置计算 mean 和 var，但对所有位置都做变换。
    
    参数：
    - values：要规范化的张量
    - mask：0/1 掩码（决定哪些位置参与均值/方差计算）
    - eps：防止除以零（std 可能接近 0）
    
    返回：规范化后的张量（形状同输入）
    """
    mean = masked_mean(values, mask)
    var = masked_var(values, mask)
    return (values - mean) * torch.rsqrt(var + eps)  # rsqrt = 1/sqrt（更快）


# ══════════════════════════════════════════════════════════════════════
# 序列生成工具
# ══════════════════════════════════════════════════════════════════════

def get_response_mask(
    response_ids: torch.Tensor,
    eos_token_id: Union[int, List[int]] = 2,
    dtype: torch.dtype = torch.long
):
    """
    生成回答部分的 attention mask（EOS 之前包含 EOS 本身为 1，之后为 0）。
    
    EOS token 应该包含在 mask 中（因为 EOS 的生成也需要被学习），
    但 EOS 之后的 padding 不应该参与损失/奖励计算。
    
    示例（eos_token_id=1）：
    response_ids:  [0, 0, 2, 4, 3, 5, 1, 0, 0]   ← token IDs
    response_mask: [1, 1, 1, 1, 1, 1, 1, 0, 0]   ← EOS 及之前全为 1，EOS 之后为 0
    
    参数：
    - response_ids：回答部分的 token ID 张量（批次形状）
    - eos_token_id：EOS token 的 ID（可以是列表，支持多个 EOS token）
    - dtype：输出掩码的数据类型
    
    返回：形状与 response_ids 相同的掩码张量
    """
    # 支持列表形式的 eos_token_id（某些模型有多个 EOS token）
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    
    # 找出所有 EOS token 的位置（多个 EOS token 取 or）
    response_mask = torch.zeros_like(response_ids, dtype=torch.bool)
    for token_id in eos_token_id:
        response_mask |= response_ids.eq(token_id)
    
    response_mask = response_mask.long()
    
    # cumsum 计算"到目前位置已遇到多少个 EOS"
    # 减去 response_mask 本身：把 EOS 位置自己排除在外（EOS 位置不算"EOS 之后"）
    # 结果：EOS 之前（含 EOS）= 0，EOS 之后 = 1
    response_mask = (torch.cumsum(response_mask, dim=1) - response_mask).bool()
    
    # 取逻辑非：EOS 之前（含 EOS）= True(=1)，EOS 之后 = False(=0)
    response_mask = torch.logical_not(response_mask).to(dtype)
    
    return response_mask


def get_attention_mask_from_padded_input_ids(
    input_ids: torch.Tensor,
    pad_token_id: int = 0,
    dtype: torch.dtype = torch.long
) -> torch.Tensor:
    """
    从左 padding 的 input_ids 恢复 attention mask。
    
    非 pad token 的位置为 1，pad token 的位置为 0。
    
    参数：
    - input_ids：左 padding 的 token ID 张量
    - pad_token_id：padding token 的 ID（通常为 0）
    - dtype：输出掩码的数据类型
    
    返回：attention mask 张量（形状同 input_ids）
    """
    input_mask = input_ids.ne(pad_token_id).long()  # 不等于 pad_token_id 的位置为 1
    return input_mask.to(dtype)


def pad_2d_list_to_length(
    response: List[List[int]],
    pad_token_id: int,
    max_length: Optional[int] = None,
    left_or_right: str = "right"
) -> torch.Tensor:
    """
    把变长的 token ID 列表（2D list）padding 成等长的 2D 张量。
    
    用途：vLLM 生成多个回答（变长），需要 padding 成固定长度才能批处理。
    
    策略：
    - 如果序列长度 < target_length：在右侧（或左侧）补 padding
    - 如果序列长度 > target_length：截断（保留前 target_length 个 token）
    
    参数：
    - response：变长 token ID 列表（每个子列表是一个回答的 token IDs）
    - pad_token_id：用于填充的 token ID
    - max_length：目标长度（None = 使用最长序列的长度）
    - left_or_right：padding 方向（"right" = 右侧补 pad；"left" = 左侧补 pad）
    
    返回：形状 (batch_size, target_length) 的 token ID 张量
    """
    # 计算所有序列中最长的长度
    max_response_length = max(len(sub_list) for sub_list in response)
    
    # 确定目标长度
    if max_length is not None and max_length > max_response_length:
        target_length = max_length  # 使用指定的最大长度（比实际最长的还长）
    else:
        target_length = max_response_length  # 使用实际最长的长度
    
    result_responses = []
    for sub_list in response:
        if left_or_right == "right":
            if len(sub_list) > target_length:
                # 截断：保留前 target_length 个
                result_responses.append(tuple(sub_list[:target_length]))
            else:
                # 右侧 padding
                result_responses.append(
                    tuple(sub_list) + (pad_token_id,) * (target_length - len(sub_list))
                )
        elif left_or_right == "left":
            if len(sub_list) > target_length:
                result_responses.append(tuple(sub_list[:target_length]))
            else:
                # 左侧 padding
                result_responses.append(
                    (pad_token_id,) * (target_length - len(sub_list)) + tuple(sub_list)
                )
    
    return torch.tensor(result_responses)


def pad_and_clip_2d_list_to_length(
    response: List[List[int]],
    pad_token_id: int,
    max_length: Optional[int] = None,
    left_or_right: str = "right"
) -> torch.Tensor:
    """
    把变长的 token ID 列表 padding/截断到指定长度（必须提供 max_length）。
    
    与 pad_2d_list_to_length 的区别：
    - 这里强制要求提供 max_length（不允许为 None）
    - 语义上更强调"限制最大长度"
    
    参数：
    - response：变长 token ID 列表
    - pad_token_id：padding token ID
    - max_length：目标长度（必须提供）
    - left_or_right：padding 方向
    
    返回：形状 (batch_size, max_length) 的 token ID 张量
    """
    assert max_length is not None, "max_length must be specified for padding and clipping."
    target_length = max_length
    
    result_responses = []
    for sub_list in response:
        if left_or_right == "right":
            if len(sub_list) > target_length:
                result_responses.append(tuple(sub_list[:target_length]))
            else:
                result_responses.append(
                    tuple(sub_list) + (pad_token_id,) * (target_length - len(sub_list))
                )
        elif left_or_right == "left":
            if len(sub_list) > target_length:
                result_responses.append(tuple(sub_list[:target_length]))
            else:
                result_responses.append(
                    (pad_token_id,) * (target_length - len(sub_list)) + tuple(sub_list)
                )
    
    return torch.tensor(result_responses)


def pad_sequence_to_length(
    tensor: torch.Tensor,
    max_seq_len: int,
    pad_token_id: int,
    left_pad: bool = False
) -> torch.Tensor:
    """
    把张量在最后一个维度上 padding 到指定长度（如果已经够长则不做操作）。
    
    支持 n 维张量（padding 最后一维）。
    支持左 padding（在序列前面补）和右 padding（在序列后面补）。
    
    参数：
    - tensor：待 padding 的张量（支持 nD）
    - max_seq_len：目标序列长度
    - pad_token_id：padding 值
    - left_pad：True = 左 padding；False = 右 padding（默认）
    
    返回：padding 后的张量（最后一维长度 = max_seq_len）
    """
    if tensor.size(-1) >= max_seq_len:
        return tensor  # 已经够长，不需要 padding
    
    # 构建 padding 张量的形状（与原张量形状相同，但最后一维是 padding 长度）
    pad_shape = list(tensor.shape)
    pad_shape[-1] = max_seq_len - tensor.size(-1)
    
    # 创建全 pad_token_id 的填充张量
    pad_tensor = torch.full(
        pad_shape, fill_value=pad_token_id, dtype=tensor.dtype, device=tensor.device
    )
    
    if left_pad:
        # 左 padding：把填充放在前面
        return torch.cat((pad_tensor, tensor), dim=-1)
    else:
        # 右 padding：把填充放在后面
        return torch.cat((tensor, pad_tensor), dim=-1)


def postprocess_data(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    max_length: int,
    pad_token_id: int,
    left_pad: bool = True,
    truncation: Literal["left", "right", "error"] = "error",
):
    """
    对 (input_ids, attention_mask, position_ids) 三元组做 padding 或截断。
    
    这是 RLHFDataset.__getitem__ 中的最后一步，确保所有样本的 prompt 等长。
    
    参数：
    - input_ids：token ID 张量（1D，形状 (seq_len,)）
    - attention_mask：attention mask 张量（1D）
    - position_ids：position ID 张量（1D 或 2D，Qwen2.5-VL MRoPE 是 3D）
    - max_length：目标序列长度
    - pad_token_id：用于填充 input_ids 的 token ID
    - left_pad：是否使用左 padding（True = 在序列前面补；RL 训练通常用左 padding）
    - truncation：截断策略：
        - "left"：截断序列的前半部分（保留末尾）
        - "right"：截断序列的后半部分（保留开头）
        - "error"：如果超长则抛出异常
    
    返回：
    - 处理后的 (input_ids, attention_mask, position_ids) 三元组
    """
    assert truncation in ["left", "right", "error"]
    seq_length = len(input_ids)
    
    if seq_length < max_length:
        # ── 需要 padding ──
        input_ids = pad_sequence_to_length(
            input_ids, max_seq_len=max_length, pad_token_id=pad_token_id, left_pad=left_pad
        )
        # attention_mask 和 position_ids 用 0 填充
        attention_mask = pad_sequence_to_length(
            attention_mask, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad
        )
        position_ids = pad_sequence_to_length(
            position_ids, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad
        )
    
    elif seq_length > max_length:
        # ── 需要截断 ──
        if truncation == "left":
            # 左截断：去掉序列的开头（保留末尾，保留最新的上下文）
            input_ids = input_ids[..., -max_length:]
            attention_mask = attention_mask[..., -max_length:]
            position_ids = position_ids[..., -max_length:]
        elif truncation == "right":
            # 右截断：去掉序列的末尾（保留开头）
            input_ids = input_ids[..., :max_length]
            attention_mask = attention_mask[..., :max_length]
            position_ids = position_ids[..., :max_length]
        elif truncation == "error":
            raise RuntimeError(
                f"Input sequence length {seq_length} is longer than max length {max_length}."
            )
        else:
            raise NotImplementedError(f"Unknown truncation method {truncation}.")
    
    return input_ids, attention_mask, position_ids


# ══════════════════════════════════════════════════════════════════════
# 学习率调度器
# ══════════════════════════════════════════════════════════════════════

def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    创建"线性 Warmup + 常数"学习率调度器。
    
    调度策略：
    - 前 num_warmup_steps 步：学习率从 0 线性增加到设定的 lr
    - 之后：学习率保持不变（常数）
    
    这是 RL 训练中最常用的调度策略：
    - warmup 阶段避免训练初期更新幅度过大（策略不稳定）
    - 之后保持常数学习率（与 SFT 的余弦衰减不同，RL 不需要逐渐降温）
    
    参数：
    - optimizer：优化器对象
    - num_warmup_steps：warmup 步数（通常是总步数的 5-10%）
    - last_epoch：上次的 epoch 数（用于恢复训练时的状态，-1 表示从头开始）
    
    返回：LambdaLR 调度器
    """
    def lr_lambda(current_step: int) -> float:
        """
        在 current_step 时，学习率乘数的计算函数。
        
        - current_step < num_warmup_steps：乘数 = current_step / num_warmup_steps（从 0 线性增加到 1）
        - current_step >= num_warmup_steps：乘数 = 1.0（保持不变）
        """
        return min(1.0, float(current_step) / float(max(1, num_warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# ══════════════════════════════════════════════════════════════════════
# AnyPrecisionAdamW 自定义优化器
# ══════════════════════════════════════════════════════════════════════

# 来源：https://github.com/meta-llama/llama-cookbook/blob/v0.0.5/src/llama_cookbook/policies/anyprecision_optimizer.py
class AnyPrecisionAdamW(torch.optim.Optimizer):
    """
    可配置精度的 AdamW 优化器，带可选的 Kahan 求和补偿。
    
    相比标准 AdamW 的优势：
    1. 可以把 momentum（m₁）和 variance（m₂）用 bf16 存储（节省一半显存）
    2. 可选的 Kahan 求和：补偿 bf16 的精度损失，在全 bf16 训练下效果媲美 fp32
    
    Kahan 求和原理：
    - 在 bf16 下，每次加法都有舍入误差
    - Kahan 求和跟踪并积累这些舍入误差，在下次更新时补偿
    - 公式：param += Δ + compensation；compensation += (param_old - param_new) + Δ
    
    使用场景：
    - 当 GPU 显存紧张时，把优化器状态从 fp32 降到 bf16（节省 ~50% 优化器显存）
    - 配合 FSDP 混合精度训练
    
    参数：
    - params：可优化的参数列表（与标准 AdamW 相同）
    - lr：学习率（default: 1e-3）
    - betas：Adam 的 β₁、β₂（default: (0.9, 0.999)）
    - eps：数值稳定项（default: 1e-8）
    - weight_decay：权重衰减（default: 0.0）
    - use_kahan_summation：是否启用 Kahan 求和（default: True）
    - momentum_dtype：m₁ 的存储精度（default: "bfloat16"）
    - variance_dtype：m₂ 的存储精度（default: "bfloat16"）
    - compensation_buffer_dtype：Kahan 补偿缓冲的精度（default: "bfloat16"）
    """
    
    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        use_kahan_summation: bool = True,
        momentum_dtype: str = "bfloat16",
        variance_dtype: str = "bfloat16",
        compensation_buffer_dtype: str = "bfloat16",
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "use_kahan_summation": use_kahan_summation,
            "momentum_dtype": momentum_dtype,
            "variance_dtype": variance_dtype,
            "compensation_buffer_dtype": compensation_buffer_dtype,
        }
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        执行一次优化步骤（更新所有参数）。
        
        参数：closure - 可选的闭包函数（重新计算损失，用于 L-BFGS 等方法）
        """
        if closure is not None:
            with torch.enable_grad():
                closure()
        
        for group in self.param_groups:
            # 提取当前参数组的超参数
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_kahan_summation = group["use_kahan_summation"]
            
            # 确定各缓冲区的精度
            momentum_dtype = PrecisionType.to_dtype(group["momentum_dtype"])
            variance_dtype = PrecisionType.to_dtype(group["variance_dtype"])
            compensation_buffer_dtype = PrecisionType.to_dtype(group["compensation_buffer_dtype"])
            
            for p in group["params"]:
                assert isinstance(p, torch.Tensor)
                
                if p.grad is None:
                    continue  # 没有梯度（被冻结或未参与前向传播），跳过
                
                if p.grad.is_sparse:
                    raise RuntimeError("AnyPrecisionAdamW does not support sparse gradients.")
                
                state = self.state[p]
                
                # ── 状态初始化（第一次 step 时）──
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    
                    # m₁：梯度的指数移动平均（momentum）
                    state["exp_avg"] = torch.zeros_like(p, dtype=momentum_dtype)
                    
                    # m₂：梯度平方的指数移动平均（uncentered variance）
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=variance_dtype)
                    
                    # Kahan 补偿缓冲（只在 use_kahan_summation=True 时创建）
                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(p, dtype=compensation_buffer_dtype)
                
                # ── 参数更新 ──
                
                state["step"] += 1  # 全局步数（用于 bias correction）
                step = state["step"]
                
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = p.grad
                
                # ① AdamW 风格权重衰减（先于梯度更新，与 L2 正则化不完全等价）
                if weight_decay:
                    p.data.mul_(1 - lr * weight_decay)
                
                # ② 更新 m₁（梯度的指数移动平均）：m₁ = β₁ × m₁ + (1-β₁) × g
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # ③ 更新 m₂（梯度平方的指数移动平均）：m₂ = β₂ × m₂ + (1-β₂) × g²
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # ④ Bias correction（消除初期的 momentum 偏差）
                bias_correction1 = 1 - beta1**step      # 1 - β₁ᵗ
                step_size = lr / bias_correction1        # 修正后的学习率
                
                bias_correction2 = (1 - beta2**step) ** 0.5  # sqrt(1 - β₂ᵗ)
                # 修正后的分母：(sqrt(m₂) / sqrt(1-β₂ᵗ)) + eps
                centered_variance = (exp_avg_sq.sqrt() / bias_correction2).add_(eps, alpha=1)
                
                if use_kahan_summation:
                    # ⑤a Kahan 求和：把精度损失积累到 compensation 缓冲中
                    compensation = state["compensation"]
                    
                    # 先把参数更新量加到 compensation 里
                    # compensation += -step_size × m₁ / centered_variance
                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)
                    
                    # 然后把 compensation 加到参数里（Kahan 求和的本体）
                    temp_buffer = p.detach().clone()  # 保存更新前的参数值
                    p.data.add_(compensation)         # p += compensation
                    
                    # 把更新引起的实际变化量和预期变化量之差存回 compensation
                    # compensation += (p_before - p_after)
                    # 这样下次迭代时会补偿这次的舍入误差
                    compensation.add_(temp_buffer.sub_(p.data))
                else:
                    # ⑤b 标准 AdamW 更新：p -= step_size × m₁ / centered_variance
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)
