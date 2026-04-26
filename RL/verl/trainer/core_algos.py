"""
本文件的作用：实现 PPO/GRPO/VLPO 算法的核心数学计算函数。

这是整个 RL 训练的数学核心，包含：
1. KL 散度控制器（防止策略更新幅度过大）
2. 各种 Advantage（优势）计算方法
3. PPO 策略损失（policy loss）计算
4. 价值函数损失（value loss）计算
5. KL 散度惩罚计算

Advantage 是 RL 中的关键概念：
- A(s,a) = Q(s,a) - V(s) = 这个动作比平均水平好多少
- 正的 Advantage 意味着"做对了，应该更多地做这个"
- 负的 Advantage 意味着"做错了，应该减少做这个"

本项目支持多种 Advantage 估计方法：
1. GAE（Generalized Advantage Estimation）：用价值网络做 bootstrap，最精确但需要 Critic
2. GRPO（Outcome）：只用最终奖励，在 batch 内做 z-score 规范化，不需要 Critic
3. GRPO（Step）：按推理步骤分配奖励，更细粒度的信号（VLPO 的核心之一）
4. GRPO（Latent）：专为 latent token 的优势估计（VLPO 专用）
5. RLOO：Leave-One-Out 基线，GRPO 的变体
6. REINFORCE++：带折扣因子的 REINFORCE，结合 whitening
7. ReMax：用贪婪解码结果作为基线
"""

# 导入抽象基类工具（用于定义 KLController 的接口）
from abc import ABC, abstractmethod
# 导入 defaultdict（用于按题目 ID 分组统计奖励）
from collections import defaultdict
# 导入类型注解
from typing import TYPE_CHECKING, Tuple

# 导入 NumPy（用于对数计算等）
import numpy as np
# 导入 PyTorch（所有张量操作）
import torch
# 导入 PyTorch 函数式接口（KL 散度等）
import torch.nn.functional as F

# 导入项目自定义的张量工具函数（masked_mean、masked_whiten 等）
from ..utils import torch_functional as VF

# 仅用于类型检查时导入（避免循环导入，运行时不会实际导入）
if TYPE_CHECKING:
    from .config import AlgorithmConfig


# ══════════════════════════════════════════════════════════════════════
# KL 散度控制器：防止策略更新步子迈太大
# ══════════════════════════════════════════════════════════════════════

class KLController(ABC):
    """
    KL 散度控制器的抽象基类。
    
    KL 散度惩罚的作用：
    - 训练时在奖励函数里减去 KL(π||π_ref)，防止模型偏离参考策略太远
    - 如果 KL 惩罚系数（kl_coef）太小，模型可能学会"钻空子"（reward hacking）
    - 如果 kl_coef 太大，模型更新太慢
    
    两种实现：
    1. FixedKLController：kl_coef 固定不变
    2. AdaptiveKLController：根据当前 KL 散度自动调整 kl_coef
    """
    
    kl_coef: float  # KL 惩罚系数（越大越保守）
    
    @abstractmethod
    def update(self, current_kl: float, n_steps: int) -> None:
        """
        根据当前 KL 散度更新 kl_coef。
        
        参数：
        - current_kl：当前的 KL 散度值
        - n_steps：当前训练步数
        """
        ...


class AdaptiveKLController(KLController):
    """
    自适应 KL 控制器：根据当前 KL 散度自动调整 kl_coef。
    
    原理（来自 InstructGPT 论文 https://arxiv.org/pdf/1909.08593.pdf）：
    - 如果当前 KL > 目标 KL（target_kl）：增大 kl_coef（加重惩罚，让模型更保守）
    - 如果当前 KL < 目标 KL：减小 kl_coef（减轻惩罚，让模型更自由地探索）
    
    调整公式：
    proportional_error = clip(current_kl / target_kl - 1, -0.2, 0.2)
    kl_coef *= (1 + proportional_error × n_steps / horizon)
    
    horizon 是一个平滑参数，控制调整的速度（horizon 越大，调整越慢越稳定）
    """
    
    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        """
        参数：
        - init_kl_coef：初始 KL 惩罚系数
        - target_kl：目标 KL 散度（希望维持在这个水平）
        - horizon：调整速度控制参数（越大调整越慢）
        """
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon
    
    def update(self, current_kl: float, n_steps: int) -> None:
        """
        根据当前 KL 散度更新 kl_coef（自适应调整）。
        
        参数：
        - current_kl：当前观测到的 KL 散度
        - n_steps：当前全局训练步数
        """
        target = self.target
        
        # 计算比例误差：(当前KL / 目标KL - 1) 表示偏离目标的程度
        # clip 到 [-0.2, 0.2] 防止调整幅度过大（稳定性保护）
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        
        # 计算调整乘数：正误差 → mult > 1 → kl_coef 增大；负误差 → mult < 1 → kl_coef 减小
        mult = 1 + proportional_error * n_steps / self.horizon
        
        # 更新 kl_coef
        self.kl_coef *= mult


class FixedKLController(KLController):
    """
    固定 KL 控制器：kl_coef 不变。
    
    适用场景：
    - 对 KL 惩罚不需要自适应调整（如 GRPO 通常用固定系数）
    - 或者通过其他机制（如 clip ratio）控制策略更新幅度
    """
    
    def __init__(self, init_kl_coef: float):
        """
        参数：
        - init_kl_coef：KL 惩罚系数（训练全程不变）
        """
        self.kl_coef = init_kl_coef
    
    def update(self, current_kl: float, n_steps: int) -> None:
        """固定控制器不做任何更新（空操作）"""
        pass


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """
    根据训练配置创建对应的 KL 控制器。
    
    参数：
    - algorithm_config：算法配置对象，包含：
        - kl_type：控制器类型（"fixed" 或 "adaptive"）
        - kl_coef：初始 KL 系数
        - kl_target（adaptive 时使用）：目标 KL 散度
        - kl_horizon（adaptive 时使用）：调整速度
    
    返回：
    - KLController 对象（FixedKLController 或 AdaptiveKLController）
    """
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, \
            f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")
    
    return kl_ctrl


# ══════════════════════════════════════════════════════════════════════
# Advantage 计算函数
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,   # 形状：(batch_size, response_length)
    values: torch.Tensor,                # 形状：(batch_size, response_length)
    response_mask: torch.Tensor,         # 形状：(batch_size, response_length)
    gamma: torch.Tensor,                 # 折扣因子（通常 0.99）
    lam: torch.Tensor,                   # GAE lambda（通常 0.95）
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 GAE（Generalized Advantage Estimation）计算 advantage 和 return。
    
    GAE 公式（从后往前递推）：
    δ_t = r_t + γ × V(s_{t+1}) - V(s_t)   （TD 残差）
    A_t = δ_t + (γλ) × A_{t+1}              （GAE 递推）
    
    其中：
    - r_t：第 t 步的奖励
    - V(s_t)：价值网络对第 t 步状态的估计
    - γ（gamma）：折扣因子（未来奖励的衰减率）
    - λ（lam）：GAE 平滑参数（0=TD(0)，1=MC估计，通常0.95折中）
    
    参数：
    - token_level_rewards：每个 token 位置的奖励（大部分位置为0，只有最后一个有值）
    - values：价值网络对每个位置的价值估计
    - response_mask：有效 token 的掩码（EOS 之后的位置为 0）
    - gamma：折扣因子
    - lam：GAE lambda
    
    返回：
    - advantages：规范化后的优势值（形状同输入）
    - returns：计算出的回报（advantages + values）
    """
    # 从后往前递推（时间倒序）
    lastgaelam = 0       # 上一步的 GAE 值（初始为 0）
    advantages_reversed = []  # 倒序存储（最后翻转回来）
    gen_len = token_level_rewards.shape[-1]  # 序列总长度
    
    for t in reversed(range(gen_len)):
        # 取下一步的价值估计（最后一步设为 0，没有下一步）
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        
        # TD 残差：即时奖励 + 折扣后的下一步价值 - 当前步价值
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        
        # GAE 递推：δ_t + γλ × A_{t+1}
        lastgaelam = delta + gamma * lam * lastgaelam
        
        # 倒序存储
        advantages_reversed.append(lastgaelam)
    
    # 翻转回正序，并堆叠成 (batch_size, gen_len) 张量
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    
    # 回报 = 优势 + 价值估计
    returns = advantages + values
    
    # 对优势做 whitening（规范化）：减均值、除标准差（只在有效位置计算）
    advantages = VF.masked_whiten(advantages, response_mask)
    
    return advantages, returns


# 注意：这里只考虑 outcome supervision（最终奖励），不做 per-token 奖励分配
@torch.no_grad()
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,   # 形状：(batch_size, response_length)
    response_mask: torch.Tensor,         # 形状：(batch_size, response_length)
    index: torch.Tensor,                 # 形状：(batch_size,)，每个样本的题目 ID
    eps: float = 1e-6                    # 防止除以零的小量
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO 算法的 Outcome Advantage 计算（最终奖励级别，不区分推理步骤）。
    
    GRPO 的核心思想：
    - 对同一道题（sample_id 相同）采样 n 个回答（rollout.n > 1）
    - 把每个回答的总奖励在同组内做 z-score 规范化：A_i = (r_i - mean) / std
    - 规范化后，高于平均水平的回答 advantage 为正（鼓励），低于平均的为负（惩罚）
    - 不需要 Critic（价值网络），因为用同组的均值作为基线
    
    与 GAE 的区别：
    - GAE 需要 Critic 对每个 token 做价值估计
    - GRPO 只需要多次采样，用组内均值做基线
    
    参数：
    - token_level_rewards：token 级别的奖励（对 response 求和得到总奖励）
    - response_mask：有效 token 的掩码
    - index：每个样本对应的题目 ID（相同 ID 的样本在同一组内规范化）
    - eps：防止除以零
    
    返回：
    - returns：规范化后的 advantage（形状与 token_level_rewards 相同，每个有效 token 位置填充同一个值）
    - returns：（GRPO 里 returns = advantages，不需要区分）
    """
    # 对每个样本的 token 级别奖励求和，得到整个回答的总奖励
    scores = token_level_rewards.sum(dim=-1)  # (batch_size,)
    
    # 按题目 ID 分组，收集同组的所有奖励
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])  # 把第 i 个样本的奖励归到对应题目的组里
    
    # 计算每个题目组的均值和标准差
    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."  # 至少要有 2 个样本才能计算标准差
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))
    
    # 对每个样本做 z-score 规范化：(r_i - mean_group) / std_group
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)
    
    # 把标量 advantage 广播到每个有效 token 位置（unsqueeze + mask 相乘）
    # 同一个回答里所有有效 token 都共享相同的 advantage 值
    returns = scores.unsqueeze(-1) * response_mask  # (batch_size, response_length)
    
    return returns, returns


@torch.no_grad()
def compute_grpo_step_advantage(
    token_level_rewards: torch.Tensor,   # 形状：(batch_size, response_length)
    response_mask: torch.Tensor,         # 形状：(batch_size, response_length)
    index: torch.Tensor,                 # 形状：(batch_size,)
    eps: float = 1e-6,
    step_end_poss=None,   # List[List[int]]：每个样本各推理步骤结束位置的列表
    delim_poss=None,       # List[List[int]]：分隔符（如 "### Step x:"）的位置
    normalize: bool = True  # 是否做规范化（控制方差）
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO 的 Step-level Advantage 计算（比 Outcome 更细粒度，按推理步骤分配奖励）。
    
    与 compute_grpo_outcome_advantage 的区别：
    - Outcome：整个回答只有一个奖励，所有 token 共享
    - Step：每个推理步骤有各自的奖励，只在步骤结束位置放奖励，
           然后向左"填充"（fill_left_nonzero）覆盖这个步骤的所有 token
    
    步骤奖励的分组规范化：
    - 与 Outcome 类似，同一道题的同一步骤位置在组内做规范化
    - 额外乘以 1.78 / sqrt(rollout_n - 1) 以控制整体方差（保持与 GRPO 相似的 scale）
    
    参数：
    - step_end_poss：每个样本各步骤结束位置的列表，如 [[10, 25, 40], [8, 20, 38], ...]
    - delim_poss：分隔符 token 的位置（如 "### Step 1:" 这些 token 的位置）
                 这些位置的 advantage 设为 0（分隔符本身不用于学习）
    - normalize：是否做规范化（控制梯度幅度）
    
    返回：
    - returns：step-level 规范化 advantage 张量
    - returns：（同上）
    """
    
    # 直接操作 rewards 的 data（避免 autograd 追踪）
    scores = token_level_rewards.data  # (batch_size, response_length)
    
    # 按题目 ID 分组，收集同一道题在各步骤位置的奖励
    id2score = defaultdict(list)
    id2mean, id2std, id2len = {}, {}, {}
    
    bsz = scores.shape[0]
    for i in range(bsz):
        # 只取步骤结束位置的奖励值（而不是所有 token 的奖励）
        id2score[index[i]].append(scores[i][step_end_poss[i]])
    
    # 计算每组的均值、标准差和长度
    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        group_score = torch.cat(id2score[idx], dim=0)  # 把所有样本的步骤奖励拼接
        id2mean[idx] = torch.mean(group_score)
        id2std[idx] = torch.std(group_score)
        id2len[idx] = group_score.shape[0]  # 总步骤数（样本数 × 每个样本的步骤数）
    
    # 对每个样本的步骤奖励做规范化
    for i in range(bsz):
        scores[i][step_end_poss[i]] = (
            scores[i][step_end_poss[i]] - id2mean[index[i]]
        ) / (id2std[index[i]] + eps)
        
        if normalize:
            # 额外缩放：1.78 / sqrt(rollout_n - 1)
            # 这个系数来自论文中的方差分析，使 step-level advantage 的方差与 outcome-level 一致
            scores[i][step_end_poss[i]] = (
                scores[i][step_end_poss[i]] * 1.78 / (np.sqrt(id2len[index[i]] - 1) + eps)
            )
    
    # 把只在步骤结束位置有值的奖励，向左填充到整个步骤的所有 token 位置
    # 例如：[0, 0, 0, A, 0, 0, B, 0] → [A, A, A, A, B, B, B, 0]（最后乘以 response_mask）
    returns = fill_left_nonzero(scores) * response_mask
    
    # 把分隔符（"### Step x:"）位置的 advantage 设为 0
    # 这些 token 是格式标记，不应该参与策略学习
    max_len = returns.shape[1]
    for i in range(bsz):
        if len(delim_poss[i]) == 0:
            continue  # 没有分隔符，跳过
        
        # 对每个分隔符位置，及其后续 5 个 token 都设为 0
        # （因为分隔符可能跨几个 token：如 "### Step 1:" 是多个 token）
        for j in range(5):
            poss = (torch.tensor(delim_poss[i]) + j).clamp(max=max_len - 1)  # 防止越界
            returns[i][poss] = 0
    
    return returns, returns


def fill_left_nonzero(x: torch.Tensor) -> torch.Tensor:
    """
    把每一行中非零值向左填充（复制），直到被下一个非零值覆盖。
    
    用途：把"步骤结束位置的奖励值"填充到整个步骤的 token 上。
    
    示例：
    输入：[0, 0, A, 0, 0, B, 0, 0]
    输出：[A, A, A, B, B, B, 0, 0]  ← 每个非零值往左填充直到序列开头或上一个非零值
    
    注意：这里"向左填充"是针对翻转后的序列来实现的（等效于原序列的向右传播后再翻转）。
    
    参数：x - 形状为 (batch_size, seq_len) 的 2D 张量
    返回：填充后的张量（形状相同）
    
    异常：ValueError - 如果输入不是 2D 张量
    """
    if x.dim() != 2:
        raise ValueError("Only support 2-dim tensor")
    
    # 翻转序列（沿列方向），把"向左填充"变成"向右传播"（更容易实现）
    x_rev = torch.flip(x, dims=[1])  # (batch_size, seq_len)
    C = x_rev.size(1)
    
    # 列索引矩阵：每行都是 [0, 1, 2, ..., C-1]
    col_idx = torch.arange(C, device=x.device).expand_as(x_rev)
    
    # 找每个位置上是否有非零值：有则记录列索引，没有则记录 -1
    nz_idx = torch.where(x_rev != 0, col_idx, torch.full_like(col_idx, -1))
    
    # cummax 从左到右取最大值：-1 → 第一个非零列索引 → 保持不变（直到下一个非零）
    # 结果：last_nz_idx[i, j] = 到目前为止见过的最新非零值的列索引
    last_nz_idx, _ = nz_idx.cummax(dim=1)
    
    # clamp(min=0) 防止 -1 索引导致越界
    gather_idx = last_nz_idx.clamp(min=0)
    
    # 用 gather 取出每个位置应该填充的值
    filled_rev = x_rev.gather(1, gather_idx)
    
    # 对从未出现过非零值的位置（last_nz_idx == -1），填充为 0
    filled_rev = torch.where(
        last_nz_idx == -1,
        torch.zeros_like(filled_rev),
        filled_rev
    )
    
    # 翻转回原来的顺序
    return torch.flip(filled_rev, dims=[1])


@torch.no_grad()
def compute_grpo_latent_advantage(
    token_level_rewards: torch.Tensor,   # 形状：(batch_size, response_length)
    response_mask: torch.Tensor,         # 形状：(batch_size, response_length)
    index: torch.Tensor,                 # 形状：(batch_size,)
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    VLPO 的 Latent Advantage 计算（专门用于 latent token 位置的优势估计）。
    
    与 compute_grpo_outcome_advantage 实现完全相同，但在语义上专门针对 latent token：
    - response_mask 只覆盖 latent token 的位置（非 latent token 位置为 0）
    - 这样 advantage 只会被应用到 latent token 的梯度更新上
    - 允许 latent token 和文字 token 有不同的 advantage 权重
    
    参数、返回值与 compute_grpo_outcome_advantage 相同。
    """
    # 求和得到总奖励（与 outcome 版本相同）
    scores = token_level_rewards.sum(dim=-1)
    
    # 按题目 ID 分组
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    
    # 组内 z-score 规范化
    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))
    
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)
    
    # 广播到 token 级别（只有 response_mask 为 1 的位置才有非零 advantage）
    returns = scores.unsqueeze(-1) * response_mask
    
    return returns, returns


@torch.no_grad()
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,   # 形状：(batch_size, response_length)
    response_mask: torch.Tensor,         # 形状：(batch_size, response_length)
    index: torch.Tensor,                 # 形状：(batch_size,)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RLOO（Leave-One-Out）Outcome Advantage 计算。
    
    RLOO 与 GRPO 的区别：
    - GRPO：基线 = 同组所有样本的均值（包括自己）→ baseline_i = mean(group)
    - RLOO：基线 = 同组其他样本的均值（排除自己）→ baseline_i = (sum(group) - r_i) / (n-1)
    
    RLOO 的优点：
    - 基线估计是无偏的（因为排除了自己）
    - 理论上方差比 GRPO 略低
    
    参数、返回值格式与 compute_grpo_outcome_advantage 相同。
    """
    scores = token_level_rewards.sum(dim=-1)  # (batch_size,)
    
    # 按题目 ID 分组并计算组内总和
    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    
    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))  # 组内所有奖励之和
    
    # 对每个样本，基线 = (组内总和 - 自己的奖励) / (组内样本数 - 1)
    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)  # 排除自己的均值
        scores[i] = scores[i] - baseline  # advantage = 奖励 - 基线
    
    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@torch.no_grad()
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor,   # 形状：(batch_size, response_length)
    response_mask: torch.Tensor,         # 形状：(batch_size, response_length)
    gamma: torch.Tensor,                 # 折扣因子
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    REINFORCE++ 的 Outcome Advantage 计算。
    
    REINFORCE++ 来自论文：https://arxiv.org/abs/2501.03262
    
    与 GRPO 的区别：
    - GRPO：同组均值作为基线（需要多次采样）
    - REINFORCE++：使用折扣累积回报（discounted return）+ whitening 规范化
    
    折扣回报公式（从后往前）：
    G_t = r_t + γ × G_{t+1}
    
    在遇到 EOS 时重置（response_mask[t]=0 时 G 归零）
    
    参数：
    - gamma：折扣因子（0~1，越小越重视近期奖励）
    
    返回：
    - advantages：whitening 规范化后的 advantage（形状同输入）
    - returns：未规范化的折扣累积回报
    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0  # 从后往前的累积回报
    
    for t in reversed(range(token_level_rewards.shape[1])):
        # 折扣累积回报：G_t = r_t + γ × G_{t+1}
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        
        # 在 EOS 之后重置（response_mask[t] = 0 时 running_return 归零）
        # 防止 EOS 之后的"奖励"污染之前的步骤
        running_return = running_return * response_mask[:, t]
    
    # 对 returns 做 whitening（规范化）
    advantages = VF.masked_whiten(returns, response_mask)
    
    return advantages, returns


@torch.no_grad()
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,   # 形状：(batch_size, response_length)
    reward_baselines: torch.Tensor,      # 形状：(batch_size,)，贪婪解码的奖励
    response_mask: torch.Tensor,         # 形状：(batch_size, response_length)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ReMax 的 Outcome Advantage 计算。
    
    ReMax 来自论文：https://arxiv.org/abs/2310.10505
    
    与 GRPO 的区别：
    - GRPO：基线 = 同组采样结果的均值
    - ReMax：基线 = 贪婪解码（greedy decoding）的奖励
    
    ReMax 的直觉：
    - 贪婪解码是"现有策略的最优结果"
    - advantage = 采样结果奖励 - 贪婪奖励
    - 正 advantage：这次采样比贪婪还好（要强化）
    - 负 advantage：这次采样比贪婪差（要抑制）
    
    参数：
    - token_level_rewards：采样结果的 token 级奖励
    - reward_baselines：贪婪解码结果的奖励（已提前计算好）
    - response_mask：有效 token 掩码
    
    返回：
    - returns：advantage 张量
    - returns：（同上）
    """
    # 每个采样结果的总奖励 - 贪婪解码的总奖励 = advantage
    scores = token_level_rewards.sum(dim=-1) - reward_baselines
    
    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


# ══════════════════════════════════════════════════════════════════════
# 损失函数计算
# ══════════════════════════════════════════════════════════════════════

def compute_rewards(
    token_level_scores: torch.Tensor,   # 奖励函数给出的 token 级分数
    log_probs: torch.Tensor,            # 当前策略的对数概率
    ref_log_probs: torch.Tensor,        # 参考策略的对数概率
    kl_ratio: float,                    # KL 惩罚系数（即 kl_coef）
) -> torch.Tensor:
    """
    计算最终的 token 级奖励（原始奖励 - KL 惩罚）。
    
    KL 惩罚项：KL(π||π_ref) = log π(a|s) - log π_ref(a|s) = log_probs - ref_log_probs
    
    公式：reward = token_score - kl_ratio × KL
    
    这样，KL 惩罚直接内嵌在奖励信号里，使模型在追求高奖励的同时不会偏离参考策略太远。
    
    参数：
    - token_level_scores：奖励函数计算的每个 token 的原始得分（通常只有最后一个 token 不为零）
    - log_probs：当前策略对每个 token 的对数概率
    - ref_log_probs：参考策略对每个 token 的对数概率
    - kl_ratio：KL 惩罚权重
    
    返回：
    - 调整后的 token 级奖励
    """
    kl = log_probs - ref_log_probs  # 近似 KL 散度（逐 token 的对数概率差）
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(
    old_log_probs: torch.Tensor,   # 旧策略的对数概率（形状：(batch_size, response_length)）
    log_probs: torch.Tensor,       # 当前策略的对数概率（形状同上）
    advantages: torch.Tensor,      # 优势值（形状同上）
    response_mask: torch.Tensor,   # 有效 token 掩码（形状同上）
    clip_ratio_low: float,         # PPO clip 的下界（通常 0.2）
    clip_ratio_high: float,        # PPO clip 的上界（通常 0.2 或 0.3，DAPO 引入非对称 clip）
    clip_ratio_dual: float,        # Dual-clip PPO 的下界（用于 advantage < 0 的情况）
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 PPO（Proximal Policy Optimization）的策略损失。
    
    PPO 的核心思想：
    - 限制每次更新的幅度，防止策略更新太激进导致不稳定
    - 通过"clipping"把概率比例 r = π(a)/π_old(a) 限制在 [1-ε, 1+ε] 范围内
    - 取 unclipped 和 clipped 目标中的较小值（更保守的更新）
    
    本实现额外支持：
    - 非对称 clip（DAPO，https://arxiv.org/pdf/2503.14476）：上下界可以不同
    - Dual-clip（https://arxiv.org/pdf/1912.09729）：当 advantage < 0 时额外限制
    
    参数：
    - clip_ratio_low：ratio 的下界 = 1 - clip_ratio_low（通常 0.2，即 ratio 不低于 0.8）
    - clip_ratio_high：ratio 的上界 = 1 + clip_ratio_high（通常 0.2，即 ratio 不高于 1.2）
    - clip_ratio_dual：当 advantage < 0 时，-advantages 不超过 clip_ratio_dual（防止过强惩罚）
    
    返回：
    - final_pg_loss：最终的策略损失（标量）
    - pg_clipfrac_higher：被"上界 clip"的比例（监控用，> 0.5 说明更新幅度太大）
    - pg_clipfrac_lower：被"下界 clip"的比例（监控用）
    - ppo_kl：近似 KL 散度（监控策略变化幅度）
    """
    
    # 计算近似 KL 散度：log(π/π_old) = log_probs - old_log_probs
    # clamp 防止极端值（log ratio 太大时 exp 会溢出）
    negative_approx_kl = torch.clamp(log_probs - old_log_probs, min=-10.0, max=10.0)
    
    # 概率比例 r = π/π_old = exp(log π - log π_old)
    ratio = torch.exp(negative_approx_kl)
    
    # 裁剪后的概率比例（限制在 [1-clip_low, 1+clip_high] 范围内）
    clipped_ratio = torch.exp(
        torch.clamp(
            negative_approx_kl,
            np.log(1.0 - clip_ratio_low),    # 下限：log(1-ε)
            np.log(1.0 + clip_ratio_high)     # 上限：log(1+ε)
        )
    )
    
    # 三种策略损失（取最保守的一个）
    pg_loss = -advantages * ratio           # 未裁剪的策略梯度损失
    pg_loss2 = -advantages * clipped_ratio  # 裁剪后的策略梯度损失
    pg_loss3 = -advantages * clip_ratio_dual  # Dual-clip 损失（用于 advantage < 0 的情况）
    
    # ── 上界 clip（PPO 标准操作） ──
    # 当 advantage > 0（好动作），clipped_ratio 比 ratio 小 → pg_loss2 比 pg_loss 大
    # 取 max(pg_loss, pg_loss2) 相当于取 min(-advantage × ratio, -advantage × clipped_ratio)
    # = 用较小的比例更新，防止对好动作过度强化
    clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)
    pg_clipfrac_higher = (pg_loss < pg_loss2).float()  # 监控被 clip 的比例
    
    # ── 下界 clip（Dual-clip PPO） ──
    # 当 advantage < 0（坏动作），如果比例 r 很小（策略已经减少做这个动作），
    # pg_loss3 提供了一个下界，防止对坏动作过度惩罚
    clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)
    
    # 只在 advantage < 0 时才应用 Dual-clip
    final_pg_loss = torch.where(
        advantages < 0,
        clipped_pg_loss_lower,    # 坏动作：应用 Dual-clip
        clipped_pg_loss_higher    # 好动作：只应用上界 clip
    )
    pg_clipfrac_lower = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()
    
    # 在有效 token 位置取均值得到标量损失
    final_pg_loss = VF.masked_mean(final_pg_loss, response_mask)
    pg_clipfrac_higher = VF.masked_mean(pg_clipfrac_higher, response_mask)
    pg_clipfrac_lower = VF.masked_mean(pg_clipfrac_lower, response_mask)
    
    # 近似 KL 散度（用于监控策略变化幅度，而不是作为损失）
    ppo_kl = VF.masked_mean(-negative_approx_kl, response_mask)
    
    return final_pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl


def compute_value_loss(
    vpreds: torch.Tensor,      # 当前价值网络的预测值（形状：(batch_size, response_length)）
    returns: torch.Tensor,     # 真实回报（形状同上）
    values: torch.Tensor,      # 旧价值网络的预测值（形状同上）
    action_mask: torch.Tensor, # 有效 action 的掩码（形状同上）
    cliprange_value: float,    # 价值函数 clip 范围
) -> Tuple[torch.Tensor, float]:
    """
    计算 PPO 的价值函数损失（MSE + clip）。
    
    PPO 对价值函数也做 clipping，防止价值网络更新幅度过大：
    - 限制 vpreds 在 [old_values - clip, old_values + clip] 范围内
    - 取 unclipped 和 clipped MSE 损失中的较大值（更保守）
    
    注意：GRPO 不需要 Critic，所以通常不调用这个函数。
    
    参数：
    - vpreds：当前价值网络对响应各位置的价值预测
    - returns：目标回报值（由 advantage + old_values 计算得到）
    - values：旧价值网络的预测（用于 clip 的参考）
    - action_mask：有效 action 的掩码
    - cliprange_value：价值函数的 clip 范围
    
    返回：
    - vf_loss：价值函数损失（标量）
    - vf_clipfrac：被 clip 的比例（监控用）
    """
    # 把 vpreds 限制在 [values - clip, values + clip] 范围内
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    
    # 未裁剪的 MSE 损失
    vf_loss1 = torch.square(vpreds - returns)
    # 裁剪后的 MSE 损失
    vf_loss2 = torch.square(vpredclipped - returns)
    
    # 取两者中较大的（更保守的更新），乘以 0.5 是 MSE 的惯例
    vf_loss = 0.5 * VF.masked_mean(torch.max(vf_loss1, vf_loss2), action_mask)
    
    # clip 的比例（vf_loss1 < vf_loss2 时说明被 clip 了）
    vf_clipfrac = VF.masked_mean((vf_loss1 < vf_loss2).float(), action_mask)
    
    return vf_loss, vf_clipfrac


def compute_kl(log_probs: torch.FloatTensor, ref_log_probs: torch.FloatTensor, kl_penalty: str) -> torch.Tensor:
    """
    计算 KL 散度（用于奖励惩罚或监控）。
    
    支持多种 KL 估计方法，各有不同的方差和计算复杂度权衡：
    
    1. "kl"：原始近似 KL = log π - log π_ref（有偏估计，方差低）
    2. "abs"：绝对值 KL = |log π - log π_ref|（对称，更鲁棒）
    3. "mse"：均方误差近似 = 0.5 × (log π - log π_ref)²（非负，平滑）
    4. "low_var_kl"：低方差 KL 估计（来自 http://joschu.net/blog/kl-approx.html）
                     = exp(log π_ref - log π) - (log π_ref - log π) - 1
    5. "full"：精确 KL 散度（在离散分布上精确计算，但需要所有 token 的概率）
    
    参数：
    - log_probs：当前策略的对数概率
    - ref_log_probs：参考策略的对数概率
    - kl_penalty：KL 估计方法（见上方列表）
    
    返回：
    - 每个 token 位置的 KL 估计值（张量，形状同输入）
    """
    # 转为 float32 确保精度
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    
    if kl_penalty == "kl":
        # 最简单的 KL 近似：log π - log π_ref
        return log_probs - ref_log_probs
    
    if kl_penalty == "abs":
        # 绝对值版本（比 "kl" 更对称）
        return (log_probs - ref_log_probs).abs()
    
    if kl_penalty == "mse":
        # 均方误差版本（非负，平滑）
        return 0.5 * (log_probs - ref_log_probs).square()
    
    if kl_penalty == "low_var_kl":
        # 低方差 KL 估计（Schulman 2020）
        # 估计器：r - log r - 1，其中 r = π_ref / π = exp(log π_ref - log π)
        # 优点：无偏，方差比直接 log ratio 低
        kl = torch.clamp(ref_log_probs - log_probs, min=-10.0, max=10.0)
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)
    
    if kl_penalty == "full":
        # 精确 KL 散度（使用 PyTorch 内置的 KL div 函数）
        # log_target=True 表示 ref_log_probs 已经是对数概率（不用再 log）
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)
    
    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")
