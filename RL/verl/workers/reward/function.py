"""
本文件的作用：定义 RL 训练中的奖励函数管理器（Reward Manager）和规则判断管理器（Rule-Based Judge Manager）。

在 RL 训练中，每次 rollout 后需要为每个生成的回答计算奖励分数。
本文件通过"动态加载外部 Python 文件"的方式，把奖励函数和规则判断函数解耦为可配置的插件。

设计模式："策略模式 + 插件化"
- 奖励函数文件（如 monet_reward_function.py）是"插件"
- 本文件的 Manager 类是"宿主"，负责加载并调用插件
- 只需要修改配置文件里的 reward_function 路径，就能切换不同的奖励函数

类层次结构：
    FunctionRewardManager（抽象基类）
    ├── SequentialFunctionRewardManager（逐样本处理，速度慢但简单）
    └── BatchFunctionRewardManager（批量处理，支持 Monet 的多种奖励模式）
    
    FunctionRuleBasedJudgeManager（抽象基类）
    ├── SingleFunctionRuleBasedJudgeManager（单样本判断）
    └── BatchFunctionRuleBasedJudgeManager（批量判断）

奖励计算的数据流：
    rollout 输出（token IDs）
    → tokenizer.decode → response_str（文字）
    → reward_fn(response_str, ground_truth) → RewardScore
    → 把 overall 分数填入 reward_tensor 的最后一个有效 token 位置
    → reward_tensor 传给 PPO/GRPO 的 advantage 计算
"""

# 用于动态加载外部 Python 文件（奖励函数插件）
import importlib.util
# 文件路径检查
import os
# 系统模块注册（让动态加载的模块可以被其他代码 import）
import sys
# 抽象基类工具
from abc import ABC, abstractmethod
# 用于按 key 分组收集奖励指标
from collections import defaultdict
# 用于给奖励函数绑定默认参数（kwargs）
from functools import partial
# 类型注解工具
from typing import Callable, Dict, List, Optional, Tuple, TypedDict
# 正则表达式（用于清理 latent token 的不可读内容）
import re
# PyTorch（创建奖励张量）
import torch
# HuggingFace tokenizer 基类
from transformers import PreTrainedTokenizer

# 导入 verl 的数据协议类（DataProto 是 batch 数据的标准包装格式）
from ...protocol import DataProto, DataProtoItem
# 导入奖励函数的配置类
from .config import RewardConfig, RuleBasedJudgeConfig
# 数值计算
import numpy as np
# 调试工具（实际训练时不使用）
import pdb
# 用于检查函数的参数签名（动态传参）
import inspect


# ══════════════════════════════════════════════════════════════════════
# 类型定义（奖励分数格式）
# ══════════════════════════════════════════════════════════════════════

class RewardScore(TypedDict):
    """
    奖励分数的标准格式（TypedDict 提供类型提示但不强制校验）。
    
    - overall：综合总分（传递给 PPO/GRPO 的奖励信号）
    - format：格式分（回答是否有 \\boxed{} 等格式要求）
    - accuracy：准确率（回答是否正确）
    """
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


# ─── 函数类型别名（便于类型注解和文档说明）───

# 逐样本奖励函数：(response_str, ground_truth) → RewardScore
SequentialRewardFunction = Callable[[str, str], RewardScore]

# 批量奖励函数：(List[response_str], List[ground_truth]) → List[RewardScore]
BatchRewardFunction = Callable[[List[str], List[str]], List[RewardScore]]

# 逐样本规则判断函数：(response_str, ground_truth) → bool
SingleRuleBasedJudgeFunction = Callable[[str, str], RewardScore]

# 批量规则判断函数：(List[response_str], List[ground_truth]) → List[bool]
BatchRuleBasedJudgeFunction = Callable[[List[str], List[str]], List[RewardScore]]


# ══════════════════════════════════════════════════════════════════════
# 奖励函数管理器（Reward Manager）
# ══════════════════════════════════════════════════════════════════════

class FunctionRewardManager(ABC):
    """
    奖励函数管理器的抽象基类。
    
    职责：
    1. 从外部 Python 文件（配置的 reward_function 路径）动态加载奖励函数
    2. 对一批 rollout 数据调用奖励函数，计算奖励分数
    3. 把分数填入 reward_tensor（形状 = 回答的 token 序列长度）
    
    动态加载奖励函数的原因：
    - 允许不重启训练就切换奖励函数（只需修改配置文件）
    - 奖励函数和训练框架解耦，便于独立开发和测试
    """
    
    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        """
        初始化奖励函数管理器。
        
        参数：
        - config：奖励配置，包含：
            - reward_function：外部奖励函数文件的路径（如 "RL/examples/reward_function/monet_reward_function.py"）
            - reward_function_name：文件里要使用的函数名（如 "compute_score"）
            - reward_function_kwargs：传给奖励函数的额外关键字参数（如 format_weight）
        - tokenizer：用于把 token ID 解码为文字
        """
        # 检查奖励函数路径是否已配置
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")
        
        # 检查奖励函数文件是否存在
        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")
        
        # 动态加载奖励函数文件（与 apply_qwen2_5_monet.py 的原理相同）
        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        
        try:
            # 注册到 sys.modules，让模块内的相对导入也能正常工作
            sys.modules["custom_reward_fn"] = module
            # 执行模块代码（等价于 import）
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")
        
        # 检查指定的函数是否存在于加载的模块中
        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")
        
        # 获取函数对象
        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        
        # 用 partial 预先绑定额外的关键字参数
        # 例如：config.reward_function_kwargs = {"format_weight": 0.1}
        # 则 self.reward_fn 调用时会自动传入 format_weight=0.1
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        
        self.config = config
        self.tokenizer = tokenizer
    
    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        计算一批 rollout 数据的奖励分数。
        
        参数：
        - data：DataProto 对象，包含：
            - data.batch["responses"]：生成回答的 token ID 张量（形状：(batch_size, seq_len)）
            - data.batch["response_mask"]：有效 token 的掩码
            - data.non_tensor_batch["ground_truth"]：标准答案列表
        
        返回：
        - reward_tensor：奖励张量（形状同 responses），非零值在最后一个有效 token 位置
        - reward_metrics：各指标（overall/format/accuracy）的值列表（用于日志记录）
        """
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    """
    逐样本执行奖励函数的管理器（顺序执行版本）。
    
    特点：
    - 简单，每次调用奖励函数只处理一个样本
    - 适合调试或奖励函数不支持批量调用的情况
    - 速度比 BatchFunctionRewardManager 慢
    """
    
    reward_fn: SequentialRewardFunction  # 类型注解（表明这里用的是逐样本版本的函数）
    
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """逐样本计算奖励，顺序处理 batch 里的每个样本。"""
        
        # 初始化奖励张量（全零，形状与 responses 相同）
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        # 初始化指标字典
        reward_metrics = defaultdict(list)
        
        response_ids = data.batch["responses"]
        # 每个样本的实际有效长度（sum 计算 response_mask 中 1 的数量）
        response_length = data.batch["response_mask"].sum(dim=-1)
        
        for i in range(len(data)):
            # 取出第 i 个样本的有效 token ID（去掉 padding）
            valid_response_ids = response_ids[i][: response_length[i]]
            
            # 把 token ID 解码为文字
            response_str = self.tokenizer.decode(
                valid_response_ids,
                skip_special_tokens=self.config.skip_special_tokens
            )
            
            # 取出对应的标准答案
            ground_truth = data.non_tensor_batch["ground_truth"][i]
            
            # 调用奖励函数（逐样本）
            score = self.reward_fn(response_str, ground_truth)
            
            # 把总分填入最后一个有效 token 的位置
            # 这是"结果奖励"（outcome reward）的标准做法：只在序列末尾有非零奖励
            reward_tensor[i, response_length[i] - 1] = score["overall"]
            
            # 收集各指标用于日志
            for key, value in score.items():
                reward_metrics[key].append(value)
        
        return reward_tensor, reward_metrics


def replace_abs_vis_token_content(s: str) -> str:
    """
    清理回答文本中 <abs_vis_token>...</abs_vis_token> 之间的不可读 latent 内容。
    
    在计算奖励之前调用，避免 latent 向量的不可见字符影响答案提取和比较。
    
    参数：s - 原始输出字符串
    返回：latent 内容替换为 <latent> 占位符后的字符串
    """
    # 匹配 <abs_vis_token>（任意内容）</abs_vis_token>
    # flags=re.DOTALL 使 . 匹配换行符
    pattern = re.compile(r'(<abs_vis_token>)(.*?)(</abs_vis_token>)', flags=re.DOTALL)
    
    # 把中间内容替换为 <latent>
    return pattern.sub(r'\1<latent>\3', s)


class BatchFunctionRewardManager(FunctionRewardManager):
    """
    批量执行奖励函数的管理器（Monet 主要使用这个版本）。
    
    特点：
    - 批量处理所有样本（一次性传入整个 batch），支持并行评判（API 调用等）
    - 支持 Monet 的多种奖励模式（基于 data.non_tensor_batch 里的字段判断）
    
    支持的奖励模式：
    1. 普通模式（只有 ground_truth）：compute_score(responses, ground_truths)
    2. MC 模式（single_step_rewards）：使用预计算的单步奖励
    3. MC2 模式（full_step_rewards）：使用预计算的多步奖励（含步骤位置信息）
    4. 预判断模式（correctness）：使用预先判断好的正确性（rule_then_api 的结果）
    """
    
    reward_fn: BatchRewardFunction  # 批量版本的奖励函数
    
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        批量计算整个 batch 的奖励分数。
        
        根据 data.non_tensor_batch 里的字段，自动选择合适的奖励计算模式。
        """
        
        # ── 解码所有样本的回答 ──
        response_str, ground_truth = [], []
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            
            if "monet" in self.config.reward_function:
                # Monet 奖励函数需要特殊处理：
                # 1. skip_special_tokens=False（保留 <abs_vis_token> 标记，用于 use_latent_reward）
                # 2. 替换 latent 内容（把不可见字符变成 <latent>）
                # 3. 去掉 EOS token（避免干扰答案提取）
                response_str_ = replace_abs_vis_token_content(
                    self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                ).replace("<|endoftext|>", "").replace("<|im_end|>", "")
            else:
                # 普通奖励函数：按配置决定是否跳过特殊 token
                response_str_ = self.tokenizer.decode(
                    valid_response_ids,
                    skip_special_tokens=self.config.skip_special_tokens
                )
            
            response_str.append(response_str_)
            ground_truth.append(data.non_tensor_batch["ground_truth"][i])
        
        # ── 动态检查奖励函数是否接受 length_penalty_weight 参数 ──
        # （允许老版奖励函数不修改也能兼容）
        extra_kwargs = {}
        try:
            sig = inspect.signature(self.reward_fn)
            if "length_penalty_weight" in sig.parameters:
                extra_kwargs["length_penalty_weight"] = self.config.length_penalty_weight
        except Exception:
            pass
        
        # ── 根据 non_tensor_batch 里的字段选择奖励计算模式 ──
        
        if "single_step_rewards" in data.non_tensor_batch:
            # MC 模式（Monte Carlo per-step）：使用预计算的单步奖励
            # single_step_rewards 是 List[float]（每个样本一个整体奖励，而不是 per-token）
            scores = self.reward_fn(response_str, data.non_tensor_batch["single_step_rewards"])
        
        elif "full_step_rewards" in data.non_tensor_batch:
            # MC2 模式：使用预计算的多步奖励（含步骤位置）
            # full_step_rewards 是 List[List[float]]（每个样本的每个步骤有独立奖励）
            scores = self.reward_fn(
                response_str,
                data.non_tensor_batch["full_step_rewards"],
                resp_lengths=response_length,
                ref_resp_lengths=data.non_tensor_batch["ref_resp_lengths"],
                **extra_kwargs,
            )
        
        elif "correctness" in data.non_tensor_batch:
            # 预判断模式：正确性已通过 rule_then_api_batch_judge 预先计算好
            # correctness 是 List[float]（1.0 正确，0.0 错误，-1.0 重复惩罚）
            scores = self.reward_fn(
                response_str,
                data.non_tensor_batch["correctness"],
                resp_lengths=response_length,
                ref_resp_lengths=data.non_tensor_batch["ref_resp_lengths"],
                **extra_kwargs,
            )
        
        else:
            # 普通模式：直接用 ground_truth 计算奖励（内部做规则判断）
            scores = self.reward_fn(
                response_str,
                ground_truth,
                resp_lengths=response_length,
                ref_resp_lengths=data.non_tensor_batch["ref_resp_lengths"],
                **extra_kwargs,
            )
        
        # ── 把分数填入奖励张量 ──
        
        # 初始化奖励张量（全零）
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        
        for i, score in enumerate(scores):
            if "overall" in score:
                # 普通奖励（outcome reward）：填入最后一个有效 token 位置
                reward_tensor[i, response_length[i] - 1] = score["overall"]
            
            elif "overall_step_wise" in score:
                # 步骤级奖励（MC2）：把每个步骤的奖励填入对应的步骤结束位置
                poss = data.non_tensor_batch["step_end_positions"][i]  # List[int]：步骤结束位置
                reward_tensor[i, poss] = torch.tensor(
                    score["overall_step_wise"], dtype=reward_tensor.dtype
                )
            
            # 收集数值类指标用于日志
            for key, value in score.items():
                if not (isinstance(value, np.floating) or isinstance(value, float)):
                    continue  # 跳过非数值类型的字段
                reward_metrics[key].append(value)
        
        return reward_tensor, reward_metrics


# ══════════════════════════════════════════════════════════════════════
# 规则判断管理器（Rule-Based Judge Manager）
# ══════════════════════════════════════════════════════════════════════

class FunctionRuleBasedJudgeManager(ABC):
    """
    规则判断管理器的抽象基类。
    
    职责：
    - 动态加载外部规则判断函数（与 FunctionRewardManager 类似的插件化设计）
    - 在 rollout 后进行"先规则、后 API"的两阶段正确性判断
    - 判断结果存入 data.non_tensor_batch["correctness"]，供 BatchFunctionRewardManager 使用
    
    与 FunctionRewardManager 的关系：
    - RuleBasedJudgeManager 只判断"对/错"（布尔值）
    - FunctionRewardManager 计算综合奖励分数（含格式分、长度惩罚等）
    - 分两步是为了支持"先规则判断 + API 补充"的流水线，避免 API 在奖励计算关键路径上
    """
    
    def __init__(self, config: RuleBasedJudgeConfig, tokenizer: PreTrainedTokenizer):
        """
        初始化规则判断管理器（动态加载外部判断函数）。
        
        参数：
        - config：规则判断配置，包含判断函数文件路径和函数名
        - tokenizer：用于把 token ID 解码为文字
        """
        # 检查判断函数路径是否配置
        if config.judge_function is None:
            raise ValueError("RuleBasedJudge function is not provided.")
        
        # 检查判断函数文件是否存在
        if not os.path.exists(config.judge_function):
            raise FileNotFoundError(f"RuleBasedJudge function file {config.judge_function} not found.")
        
        # 动态加载判断函数文件
        spec = importlib.util.spec_from_file_location("custom_rule_based_judge_fn", config.judge_function)
        module = importlib.util.module_from_spec(spec)
        
        try:
            sys.modules["custom_rule_based_judge_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load rule_based_judge function: {e}")
        
        # 检查指定函数是否存在
        if not hasattr(module, config.judge_function_name):
            raise AttributeError(f"Module {module} does not have function {config.judge_function_name}.")
        
        # 获取判断函数
        rule_based_judge_fn = getattr(module, config.judge_function_name)
        print(f"Using rule_based_judge function `{config.judge_function_name}` from `{config.judge_function}`.")
        
        # 注意：不用 partial，判断函数通常不需要预先绑定参数
        self.rule_based_judge_fn = rule_based_judge_fn
        self.config = config
        self.tokenizer = tokenizer
    
    @abstractmethod
    def compute_rule_based_judge(self, data: DataProto) -> bool:
        """
        计算一批或单个样本的正确性。
        
        参数：data - 包含 responses 和 ground_truth 的数据
        返回：正确性结果（单个 bool 或 List[bool]）
        """
        ...
    
    def compute_rule_based_judge_with_string(self, response_str: str, ground_truth: str) -> bool:
        """
        直接用文字字符串（而不是 token ID）计算正确性。
        
        这是一个便捷接口，当已经有文字字符串时可以跳过 decode 步骤。
        
        参数：
        - response_str：模型回答的文字
        - ground_truth：标准答案
        
        返回：True（正确）或 False（错误）
        """
        ...


class SingleFunctionRuleBasedJudgeManager(FunctionRuleBasedJudgeManager):
    """
    单样本规则判断管理器（逐样本处理）。
    
    调用方通常在 rollout 阶段，对每个生成结果单独判断是否正确，
    以便即时收集"这个推理步骤是否能导向正确答案"的信号。
    """
    
    rule_based_judge_fn: SingleRuleBasedJudgeFunction
    
    def compute_rule_based_judge(self, data: DataProtoItem) -> bool:
        """
        对单个样本（DataProtoItem，不是 DataProto 批次）进行正确性判断。
        
        参数：data - 单个样本的 DataProtoItem（没有批次维度）
        返回：(correctness, response_str) 元组
        """
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        
        # 取出有效 token（单个样本没有批次维度，所以直接 [:response_length]）
        valid_response_ids = response_ids[:response_length]
        
        # 解码为文字
        response_str = self.tokenizer.decode(
            valid_response_ids,
            skip_special_tokens=self.config.skip_special_tokens
        )
        ground_truth = data.non_tensor_batch["ground_truth"]
        
        try:
            # 调用判断函数
            correctness = self.rule_based_judge_fn(response_str, ground_truth)
        except Exception as e:
            # 判断失败时，保守处理：视为不正确
            print(f"Rule-based judge error: {e}")
            correctness = False
        
        return correctness, response_str
    
    def compute_rule_based_judge_with_string(self, response_str: str, ground_truth: str) -> bool:
        """
        用文字字符串（已解码）直接判断正确性。
        
        参数：
        - response_str：已解码的回答文字
        - ground_truth：标准答案
        
        返回：正确性（True/False）
        """
        try:
            correctness = self.rule_based_judge_fn(response_str, ground_truth)
        except Exception as e:
            print(f"Rule-based judge error: {e}")
            correctness = False
        return correctness


class BatchFunctionRuleBasedJudgeManager(FunctionRuleBasedJudgeManager):
    """
    批量规则判断管理器（批量处理整个 batch）。
    
    注意：当前实现实际上是逐样本循环处理，不是真正的批量调用（可能是遗留问题）。
    批量 API 判断由外部的 rule_then_api_batch_judge 函数完成。
    """
    
    rule_based_judge_fn: BatchRuleBasedJudgeFunction
    
    def compute_rule_based_judge(self, data: DataProto) -> List[bool]:
        """
        对一批样本（DataProto）进行正确性判断。
        
        注意：这里实际上是逐样本循环调用 judge_fn，因此 judge_fn 应该是单样本版本。
        
        参数：data - 批次数据
        返回：每个样本的正确性列表
        """
        correctness = []
        response_strs = []
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            # 解码第 i 个样本的回答
            response_str = self.tokenizer.decode(
                valid_response_ids,
                skip_special_tokens=self.config.skip_special_tokens
            )
            response_strs.append(response_str)
            ground_truth = data.non_tensor_batch["ground_truth"][i]
            
            try:
                # 调用判断函数（注意：这里把 correctness 变量覆盖了，有 bug：
                # 循环内的 correctness = judge_fn(...) 覆盖了外部的列表变量，
                # 然后 correctness.append(correctness) 会报错。这是一个已知的代码问题。）
                correctness = self.rule_based_judge_fn(response_str, ground_truth)
            except Exception as e:
                print(f"Rule-based judge error: {e}")
                correctness = False
            
            correctness.append(correctness)  # 注意：这里有 bug（见上方注释）
        
        return correctness
