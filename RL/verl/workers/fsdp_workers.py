"""
本文件的作用：定义 FSDPWorker——RL 训练中所有 GPU 工作单元的统一封装。

FSDPWorker 是整个 RL 训练的核心 Worker，一个 FSDPWorker 实例可以同时承担多种角色：
- actor（策略网络，负责 PPO/GRPO 梯度更新）
- rollout（vLLM 推理，负责生成候选回答）
- ref（参考策略，负责计算 KL 散度的 baseline）
- critic（价值网络，负责 PPO 的价值估计）

在 Monet/VLPO 中，一般使用 "actor_rollout_ref" 角色，即：
- 同一组 GPU 承担 Actor、Rollout、Ref Policy 三个功能
- 通过 FSDPVLLMShardingManager 在 FSDP 训练模式和 vLLM 推理模式之间动态切换

FSDP（Fully Sharded Data Parallel）的作用：
- 把模型参数分片到多个 GPU 上，节省单卡显存
- 支持梯度检查点、混合精度、CPU 卸载等优化
- 对 7B 参数模型，8 卡 FSDP 每卡只需存储 1/8 的参数

与 Ray 框架的关系：
- FSDPWorker 在训练脚本里被 @ray.remote 装饰为 Ray Actor（见 RL/verl/trainer/main.py）
- Ray 负责把训练指令（如 update_actor、generate_sequences）调度到正确的 GPU 上
- Worker 内部通过 dist.init_process_group（NCCL）完成 GPU 间的梯度同步

关键方法：
    init_model()         → 初始化 FSDP 模型、优化器、vLLM 推理引擎
    generate_sequences() → 调用 vLLMRollout 生成候选回答
    update_actor()       → PPO/GRPO 梯度更新（包含 MFU 统计）
    compute_log_probs()  → 计算当前策略的对数概率（重要：用旧模型参数计算）
    compute_ref_log_probs() → 计算参考策略的对数概率（用于 KL 散度）
    compute_rule_based_judge() → 规则判断（含 rule_then_api 两阶段）
"""

# 必须在最前面导入 Monet RL 补丁（在加载模型之前注入到 sys.modules）
import monet_rl_patch

# 类型注解工具
from typing import Literal, Optional, Union, List

# 数值计算（用于处理 numpy 格式的指标数据）
import numpy as np
# 系统资源监控（CPU 内存使用量）
import psutil
# PyTorch 核心
import torch
# PyTorch 分布式通信（NCCL 多卡通信）
import torch.distributed as dist
# HuggingFace 加速库：空模型初始化（节省内存）
from accelerate import init_empty_weights
# 计时工具（用于统计每次更新的耗时）
from codetiming import Timer
# PyTorch 分布式设备网格（定义 FSDP/TP/SP 的并行拓扑）
from torch.distributed.device_mesh import init_device_mesh
# FSDP 相关组件
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# HuggingFace 模型加载工具
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
    PreTrainedModel,
)
# 无初始化权重的上下文管理器（空模型 shell，节省内存）
from transformers.modeling_utils import no_init_weights

# verl 内部模块：序列并行（Ulysses）补丁
from ..models.monkey_patch import apply_ulysses_patch
# verl 数据协议格式
from ..protocol import DataProto
# Worker 基类（定义 rank、world_size 等属性）
from ..single_controller.base import Worker
# Dispatch 装饰器（定义如何把远程调用分发到 GPU）
from ..single_controller.base.decorator import Dispatch, register
# FSDP 检查点管理器（保存/加载模型权重）
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
# FLOP 统计器（计算 MFU，模型利用率）
from ..utils.flops_counter import FlopsCounter
# FSDP 工具函数（wrap policy、模型卸载/加载等）
from ..utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
)
# 模型信息打印工具
from ..utils.model_utils import print_gpu_memory_usage, print_model_size
# Tokenizer/Processor 加载工具
from ..utils.tokenizer import get_processor, get_tokenizer
# 精度类型工具（"bfloat16" ↔ torch.bfloat16）
from ..utils.torch_dtypes import PrecisionType
# 自定义 AdamW（支持 bf16 参数更新）和学习率调度器
from ..utils.torch_functional import AnyPrecisionAdamW, get_constant_schedule_with_warmup
# 各种 Worker 配置类
from .config import ActorConfig, CriticConfig, FSDPConfig, ModelConfig, OptimConfig, RefConfig, WorkerConfig
# vLLM Rollout（推理端）
from .rollout import vLLMRollout
# FSDP ↔ vLLM 切换管理器
from .sharding_manager import FSDPVLLMShardingManager
# FSDP ↔ Ulysses（序列并行）切换管理器
from .sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

# 操作系统接口（环境变量、文件操作）
import os

# Ray 分布式框架（调用远程 Actor）
import ray
# 以下两行是从 transformers 重复导入（IDE 可能会警告，但不影响运行）
from transformers import AutoTokenizer, AutoModel, AutoConfig, GenerationConfig
# PretrainedConfig：用于手动构造 config 实例
from transformers.configuration_utils import PretrainedConfig
# PyTorch 函数式接口
import torch.nn.functional as F
# 进度条
from tqdm import tqdm

# ── 奖励函数导入（直接从 examples 目录加载）──
# 注意：这里硬编码了具体函数的导入路径，与配置文件中的 judge_function 路径对应
from examples.reward_function.monet_reward_function import extract_and_check as easyr1_monet_extract_and_check
from examples.reward_function.monet_reward_function import extract_and_check_api as easyr1_monet_extract_and_check_api
from examples.reward_function.monet_reward_function import rule_then_api_batch_judge

# API 客户端构建工具（Gemini / DeepSeek）
from tools.custom_api import build_deepseek_client, build_gemini_client
# vLLM 推理引擎（用于 embed_model 嵌入服务）
from vllm import LLM
# OpenAI 风格的客户端（DeepSeek API 兼容）
from openai import OpenAI
# 时间工具（NCCL 超时设置）
import datetime


def _to_config(sub: object, parent_model_type: str):
    """
    把 dict 类型的子配置转换为对应的 PretrainedConfig 实例。
    
    背景：从磁盘加载的 config.json 中，某些嵌套字段（如 text_config）可能是 dict，
    而 HuggingFace 的模型初始化需要这些字段是 PretrainedConfig 对象。
    
    参数：
    - sub：可能是 dict 或已经是 Config 对象
    - parent_model_type：父配置的 model_type（用于确定子配置的类型）
    
    返回：PretrainedConfig 的子类实例
    """
    if not isinstance(sub, dict):
        return sub
    
    # 从 dict 中读取 model_type，如果没有则用父类的 model_type
    mt = sub.get("model_type", parent_model_type or "auto")
    
    try:
        # 尝试用精确的 Config 类加载（如 Qwen2Config、LlamaConfig 等）
        conf_cls = AutoConfig.for_model(mt)
        return conf_cls.from_dict(sub)
    except Exception:
        # 失败时回退到通用 PretrainedConfig
        return PretrainedConfig.from_dict(sub)


def _sanitize_mm_config(cfg, torch_dtype):
    """
    对多模态模型（Qwen2.5-VL）的配置做兼容性处理。
    
    解决的问题：
    1. 强制 is_encoder_decoder=False（解码器专用模型）
    2. 把 text_config、vision_config 等嵌套 dict 转换为 Config 对象
    3. 处理 generation_config 的格式
    4. 设置 flash_attention_2 和 torch_dtype 提示
    
    参数：
    - cfg：从 AutoConfig.from_pretrained 加载的配置
    - torch_dtype：训练时使用的精度（如 torch.bfloat16）
    
    返回：处理后的配置对象
    """
    # 1. 强制解码器专用模式（编码器-解码器模式会导致 FSDP 初始化出错）
    if getattr(cfg, "is_encoder_decoder", None):
        cfg.is_encoder_decoder = False
    
    parent_mt = getattr(cfg, "model_type", None) or "auto"
    
    # 2. 把嵌套 dict 转换为 Config 对象
    for key in ("text_config", "vision_config", "decoder", "encoder"):
        sub = getattr(cfg, key, None)
        if isinstance(sub, dict):
            setattr(cfg, key, _to_config(sub, parent_mt))
        
        # 如果 decoder/encoder 是 None，从配置中删除（防止 HF 误判为 seq2seq）
        if key in ("decoder", "encoder") and getattr(cfg, key, None) is None:
            try:
                delattr(cfg, key)
            except Exception:
                pass
    
    # 3. 处理 generation_config 字段
    gen = getattr(cfg, "generation_config", None)
    if isinstance(gen, dict):
        try:
            setattr(cfg, "generation_config", GenerationConfig.from_model_config(cfg))
        except Exception:
            setattr(cfg, "generation_config", GenerationConfig.from_dict(gen))
    
    # 4. 设置 flash_attention_2 提示（HF 后续初始化会读取）
    setattr(cfg, "attn_implementation", "flash_attention_2")
    if not hasattr(cfg, "torch_dtype"):
        setattr(cfg, "torch_dtype", torch_dtype)
    
    return cfg


class FSDPWorker(Worker):
    """
    RL 训练的核心 Worker，使用 FSDP 并行策略。
    
    一个 FSDPWorker 实例对应一组 GPU（由 device_mesh 定义），
    可以承担 Actor、Rollout、Ref Policy、Critic 等多种角色。
    
    主要能力：
    1. 模型初始化（FSDP wrap + 优化器）
    2. vLLM 推理生成（rollout 角色）
    3. 对数概率计算（actor/ref 角色）
    4. 梯度更新（actor/critic 角色）
    5. 规则判断（rollout 角色，用于计算正确性）
    6. 检查点保存/加载
    
    与 Ray 的关系：
    - 在 main.py 里，FSDPWorker 被 ray.remote 包装为 Ray Actor
    - Ray 负责把 init_model、update_actor 等调用路由到正确的 GPU 进程
    - 多个 FSDPWorker 实例通过 NCCL 完成梯度同步
    """
    
    def __init__(
        self,
        config: WorkerConfig,
        role: Literal["actor", "critic", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        """
        初始化 FSDPWorker。
        
        注意：这里只做轻量级初始化（设置角色标志、初始化 NCCL 进程组）。
        实际的模型初始化在 init_model() 方法里（由 Ray 调度后调用）。
        
        参数：
        - config：Worker 的完整配置（包含 actor/critic/ref/rollout 的子配置）
        - role：这个 Worker 承担的角色
            - "actor_rollout_ref"：同时承担训练、推理、参考策略（最常用）
            - "critic"：只做价值网络
        """
        super().__init__()
        
        self.config = config
        self.role = role
        
        # ── 初始化 NCCL 分布式进程组 ──
        
        # 设置当前进程使用的 GPU（LOCAL_RANK 是当前节点内的 GPU 索引）
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        
        if not dist.is_initialized():
            # 初始化 NCCL 进程组（用于多卡间的梯度同步和参数 allreduce）
            dist.init_process_group(
                backend="nccl",          # 使用 NVIDIA NCCL 通信库（最快的 GPU 通信方案）
                init_method="env://",    # 从环境变量（MASTER_ADDR, MASTER_PORT）读取通信地址
                world_size=int(os.environ["WORLD_SIZE"]),  # 总 GPU 数
                rank=int(os.environ["RANK"]),              # 当前进程的全局 rank
                timeout=datetime.timedelta(minutes=240),   # NCCL 超时（4小时，防止长时间卡死）
                pg_options=dist.ProcessGroupNCCL.Options(is_high_priority_stream=False)
            )
        
        # 关闭 TF32 精度（提高数值稳定性，避免梯度计算出现精度损失）
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        
        # ── 根据 role 设置角色标志 ──
        
        # 是否参与 Actor 梯度更新
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        # 是否是 Critic（价值网络）
        self._is_critic = self.role == "critic"
        # 是否承担 Rollout（推理生成）职责
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        # 是否承担 Reference Policy（KL 散度基线）职责
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        
        # 嵌入模型（用于步骤相似度计算，可选）
        self.embed_model = None
        self.embed_tokenizer = None
        
        # ── 初始化 API 客户端（如果判断策略使用 API）──
        if self._is_rollout and "api" in self.config.rule_based_judge.judge_function_name:
            # "rule_then_api" 策略：规则判断失败时调用 LLM API 做二次判断
            if self.config.rule_based_judge.api_name == 'deepseek-chat':
                self.api_client = build_deepseek_client()
            elif self.config.rule_based_judge.api_name == 'gemini-2.5-pro':
                self.api_client = build_gemini_client()
            else:
                raise ValueError(f"API {self.config.rule_based_judge.api_name} not supported.")
        
        # ── 参数/优化器卸载标志（节省 GPU 显存）──
        
        # param offload：把模型参数卸载到 CPU，只在计算时加载回 GPU
        self._use_param_offload = False
        # optimizer offload：把优化器状态（momentum等）卸载到 CPU
        self._use_optimizer_offload = False
        
        # 根据角色读取对应的 offload 配置
        if self._is_actor:
            self._use_param_offload = self.config.actor.offload.offload_params
            self._use_optimizer_offload = self.config.actor.offload.offload_optimizer
            self._init_config(self.config.actor, "actor")
        elif self._is_critic:
            self._use_param_offload = self.config.critic.offload.offload_params
            self._use_optimizer_offload = self.config.critic.offload.offload_optimizer
            self._init_config(self.config.critic, "critic")
        elif self._is_ref:
            # Ref Policy 通常不做梯度更新，只需参数卸载（不需要优化器卸载）
            self._use_param_offload = self.config.ref.offload.offload_params
            self._init_config(self.config.ref, "ref")
    
    def _init_config(
        self, config: Union[ActorConfig, CriticConfig, RefConfig], role: Literal["actor", "critic", "ref"]
    ):
        """
        初始化 FSDP 设备网格（device mesh）和 batch size 计算。
        
        device mesh 定义了 GPU 的并行拓扑：
        - 纯 FSDP：所有 GPU 在一个 "fsdp" 维度上 → (world_size,)
        - HSDP（Hybrid Shard）：两个维度 (ddp_size, fsdp_size)，
          同一个 fsdp group 内完全分片，不同 fsdp group 之间做 DDP
        
        参数：
        - config：Actor/Critic/Ref 的具体配置
        - role：角色名（用于日志打印）
        """
        world_size = dist.get_world_size()
        fsdp_size = config.fsdp.fsdp_size  # FSDP group 内的 GPU 数（≤0 表示使用所有 GPU）
        
        if fsdp_size <= 0 or fsdp_size >= world_size:
            # 纯 FSDP：所有 GPU 在同一个分片组
            self.device_mesh = init_device_mesh(
                "cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",)
            )
        else:
            # HSDP（Hybrid Sharded Data Parallel）：内部 FSDP + 外部 DDP
            # 例如：world_size=16, fsdp_size=8 → 2 个 FSDP 组，每组 8 卡
            self.device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(world_size // fsdp_size, fsdp_size),
                mesh_dim_names=("ddp", "fsdp")
            )
        
        # 配置 Ulysses 序列并行（如果启用）
        if config.ulysses_sequence_parallel_size > 1:
            # 序列并行：把一个长序列切分到多个 GPU 上并行计算
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(
                    world_size // config.ulysses_sequence_parallel_size,
                    config.ulysses_sequence_parallel_size,
                ),
                mesh_dim_names=("dp", "sp"),  # dp：数据并行，sp：序列并行
            )
        else:
            self.ulysses_device_mesh = None
        
        # 初始化 Ulysses 分片管理器
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        
        # Ref 模型没有 global_batch_size（不需要批次大小配置）
        if not hasattr(config, "global_batch_size"):
            return
        
        # 如果每道题采样 n 个回答（rollout.n > 1），global_batch_size 需要扩展 n 倍
        if self.config.rollout.n > 1:
            config.global_batch_size *= self.config.rollout.n
            self.print_rank0(f"{role} will use global batch size {config.global_batch_size}.")
        
        # 计算每个 GPU 的 batch size
        # global_batch_size_per_device = global_batch_size / num_gpus * ulysses_sp_size
        config.global_batch_size_per_device = (
            config.global_batch_size // self.device_mesh.size() * config.ulysses_sequence_parallel_size
        )
        
        if config.global_batch_size_per_device == 0:
            raise ValueError(f"{role} global batch size * ulysses size must be larger than num gpus.")
        
        # 检查 batch size 能否被 micro batch size 整除（梯度累积要求）
        if config.global_batch_size_per_device % config.micro_batch_size_per_device_for_update != 0:
            raise ValueError(f"{role} global batch size per device must be divisible by the micro batch size.")
        
        # FSDP CPU offload 与梯度累积不兼容（因为 CPU offload 需要完整的前向/反向配对）
        if (
            config.fsdp.enable_cpu_offload
            and config.global_batch_size_per_device != config.micro_batch_size_per_device_for_update
        ):
            raise ValueError(f"{role} cannot use FSDP's CPU offload when gradient accumulation is enabled.")
    
    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: Optional[OptimConfig],
        padding_free: bool = False,
    ) -> None:
        """
        初始化 FSDP 模型和优化器。
        
        流程：
        1. 加载 tokenizer 和 processor
        2. 从 pretrained 路径加载 AutoConfig
        3. 选择正确的 AutoModel 类（Vision2Seq / CausalLM / TokenClassification）
        4. 根据 enable_rank0_init 决定加载策略：
           - True（默认）：只有 rank 0 加载完整模型权重，其他 rank 创建空壳
           - False：所有 rank 都从磁盘加载（内存开销大但更简单）
        5. 用 FSDP wrap 模型
        6. 初始化优化器（AdamW 或 AnyPrecisionAdamW）
        
        参数：
        - model_config：模型路径、tokenizer 路径等配置
        - fsdp_config：FSDP 分片策略、混合精度等配置
        - optim_config：优化器类型、学习率等配置（ref 模型为 None）
        - padding_free：是否使用 Ulysses 序列并行补丁（去掉 padding 提升效率）
        """
        
        # ── 加载 tokenizer 和 processor ──
        
        self.tokenizer = get_tokenizer(
            model_config.tokenizer_path,
            trust_remote_code=model_config.trust_remote_code,
            use_fast=True,
        )
        self.processor = get_processor(
            model_config.tokenizer_path,
            trust_remote_code=model_config.trust_remote_code,
            use_fast=True,
        )
        
        # ── 加载模型配置 ──
        
        self.model_config = AutoConfig.from_pretrained(
            model_config.model_path,
            trust_remote_code=model_config.trust_remote_code,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_config.override_config,  # 允许覆盖特定配置字段
        )
        
        # 加载生成配置（temperature、top_p 等默认值）
        try:
            self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
        except Exception:
            self.generation_config = GenerationConfig.from_model_config(self.model_config)
        
        self.print_rank0(f"Model config: {self.model_config}")
        
        # ── 应用 Ulysses 序列并行补丁（如果启用）──
        if padding_free:
            apply_ulysses_patch(self.model_config.model_type)
            self.print_rank0("Ulysses patch applied!")
        
        # ── 确定模型精度 ──
        
        if fsdp_config.torch_dtype is None:
            # 默认精度：Actor/Critic 用 float32，Ref Policy 用 bfloat16
            torch_dtype = torch.float32 if self._is_actor or self._is_critic else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)
        
        # ── 选择 AutoModel 类 ──
        
        if self._is_critic:
            # Critic 使用 TokenClassification（输出每个 token 一个标量价值）
            auto_class = AutoModelForCausalLM
        elif type(self.model_config) in AutoModelForVision2Seq._model_mapping.keys():
            # 多模态模型（Qwen2.5-VL）
            auto_class = AutoModelForVision2Seq
            print("Auto class is AutoModelForVision2Seq")
        else:
            # 纯文本模型
            auto_class = AutoModelForCausalLM
            print("Auto class is AutoModelForCausalLM")
        
        # ── 初始化模型 ──
        
        cfg = self.model_config
        
        if (not fsdp_config.enable_rank0_init) or self.device_mesh.get_local_rank("fsdp") == 0:
            # Rank 0（或禁用 rank0_init 时所有 rank）：直接从磁盘加载完整模型
            cfg = _sanitize_mm_config(cfg, torch_dtype)
            model = auto_class.from_pretrained(
                model_config.model_path,
                config=cfg,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",  # 使用 Flash Attention 2 提速
                # enable_rank0_init 时只有 rank 0 加载到 CPU（之后 FSDP 会广播到其他 rank）
                device_map="cpu" if fsdp_config.enable_rank0_init else "cuda",
                low_cpu_mem_usage=True,              # 分块加载，减少峰值内存
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            # 非 rank 0（enable_rank0_init 时）：创建空模型 shell，等待 FSDP 广播
            # 这样可以节省大量内存（7B 模型 = ~14GB，8 卡如果都加载需要 112GB CPU 内存）
            setattr(cfg, "torch_dtype", torch_dtype)
            setattr(cfg, "attn_implementation", "flash_attention_2")
            
            cfg = _sanitize_mm_config(cfg, torch_dtype)
            
            # 调试：打印各字段类型（已注释掉）
            for k in ("text_config", "vision_config", "decoder", "encoder", "generation_config"):
                v = getattr(cfg, k, None)
            
            # 创建空 shell（参数未初始化）
            with no_init_weights(), init_empty_weights():
                model = auto_class.from_config(cfg)  # 只创建结构，不初始化参数
        
        # 类型检查
        assert isinstance(model, PreTrainedModel)
        
        # 绑定权重（lm_head 和 embedding 共享）
        model.tie_weights()
        
        # 转换到目标精度
        model = model.to(torch_dtype)
        
        # 启用梯度检查点（牺牲计算时间换取显存）
        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        # Ref Policy 和 Rollout 不需要梯度
        if not (self._is_actor or self._is_critic):
            model.requires_grad_(False)
        
        # 冻结视觉塔（如果配置了，只训练语言模型部分）
        if model_config.freeze_vision_tower:
            if hasattr(model, "visual"):
                model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True  # 冻结部分参数时需要 use_orig_params
                self.print_rank0("Vision tower is set to not trainable.")
            else:
                self.print_rank0("No vision tower found.")
        
        # 同步屏障（所有 rank 都完成模型初始化后再继续）
        dist.barrier(device_ids=[torch.cuda.current_device()])
        print_model_size(model)
        print_gpu_memory_usage("After huggingface model init")
        
        # ── 配置 FSDP 混合精度 ──
        
        # MixedPrecision：参数精度、梯度 reduce 精度、buffer 精度分别控制
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),   # 参数精度（通常 bf16）
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype), # allreduce 精度（通常 fp32 保证数值精度）
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype), # buffer 精度（通常 bf16）
        )
        
        # 自动确定 FSDP wrap 策略（按 TransformerBlock 的边界分层分片）
        auto_wrap_policy = get_fsdp_wrap_policy(model)
        self.print_rank0(f"FSDP wrap policy: {auto_wrap_policy}.")
        
        # ── 确定分片策略 ──
        
        if self.device_mesh.ndim == 2:
            # HSDP（Hybrid Shard Data Parallel）模式
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.HYBRID_SHARD      # 完全分片
            else:
                sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2  # ZeRO2 风格（不分片梯度）
        else:
            # 纯 FSDP 模式
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.FULL_SHARD   # ZeRO3：参数+梯度+优化器全分片
            else:
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP  # ZeRO2：只分片参数，不分片梯度
        
        # CPU offload 配置
        if fsdp_config.enable_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)  # 把模型参数卸载到 CPU
        else:
            cpu_offload = None
        
        # rank 0 init 时，其他 rank 需要在 FSDP 初始化时从 CPU 搬到 GPU
        if fsdp_config.enable_rank0_init:
            sync_module_states = True       # FSDP 初始化时从 rank 0 广播参数到其他 rank
            param_init_fn = get_init_fn(model, device="cuda") if self.rank != 0 else None  # 非 rank 0 用 lazy init
        else:
            sync_module_states = False
            param_init_fn = None
        
        # ── 用 FSDP 包装模型 ──
        self.fsdp_module = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,  # 自动决定哪些子模块需要独立分片
            mixed_precision=mixed_precision,
            param_init_fn=param_init_fn,        # 空模型 shell 的参数初始化函数
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            forward_prefetch=False,             # 禁用预取（与 vLLM 集成时可能冲突）
            use_orig_params=fsdp_config.use_orig_params,  # 是否保留原始参数（冻结层时需要）
            device_mesh=self.device_mesh,       # FSDP/HSDP 的拓扑描述
        )
        print_gpu_memory_usage("After FSDP module init")
        
        # ── 初始化优化器和学习率调度器（只有 Actor/Critic 需要）──
        
        if self._is_actor or self._is_critic:
            if optim_config.strategy == "adamw":
                # 标准 AdamW（fused=True 使用 CUDA fused 实现，更快）
                self.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                    fused=True,
                )
            elif optim_config.strategy == "adamw_bf16":
                # AnyPrecision AdamW（优化器状态用 bf16 存储，进一步节省显存）
                self.optimizer = AnyPrecisionAdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                )
            else:
                raise NotImplementedError(f"Optimizer {optim_config.strategy} not supported.")
            
            # Warmup + 常数学习率调度（先从 0 线性增加到 lr，然后保持不变）
            num_warmup_steps = int(optim_config.lr_warmup_ratio * optim_config.training_steps)
            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
            )
            print_gpu_memory_usage("After optimizer init")
        else:
            # Ref Policy 不需要优化器
            self.optimizer, self.lr_scheduler = None, None
    
    def _build_rollout(self) -> None:
        """
        初始化 vLLM Rollout 引擎和 FSDP ↔ vLLM 切换管理器。
        
        rollout_device_mesh 定义了 vLLM 的 tensor parallel 拓扑：
        - dp_size：数据并行（几道题可以并行处理）
        - tp_size：张量并行（一个 vLLM 实例占用几块 GPU）
        
        FSDPVLLMShardingManager 负责：
        - 在 FSDP 训练完一步后，把最新的模型权重同步到 vLLM
        - 切换 GPU 显存的使用模式（FSDP 参数 ↔ vLLM KV cache）
        """
        tp_size = self.config.rollout.tensor_parallel_size  # vLLM 的张量并行大小
        dp_size = self.world_size // tp_size                # 数据并行大小
        
        assert self.world_size % tp_size == 0, (
            f"rollout world size: {self.world_size} is not divisible by tp size: {tp_size}"
        )
        
        # 创建 rollout 的设备网格（dp × tp）
        rollout_device_mesh = init_device_mesh(
            "cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp")
        )
        
        # 如果使用 Monet 采样策略，获取哈希服务器（Ray Actor 引用）
        if self.config.rollout.sampling_strategy in ["monet"]:
            self.hash_server = ray.get_actor(self.config.rollout.monet.hash_server_name)
        else:
            self.hash_server = None
        
        # 获取规则判断服务器（如果配置了）
        self.rule_based_judge_server = (
            ray.get_actor(self.config.rule_based_judge.judge_server_name)
            if self.config.rule_based_judge.judge_server_name else None
        )
        
        self.embed_tokenizer = None
        
        # ── 初始化 vLLM Rollout ──
        self.rollout = vLLMRollout(
            model_path=self.config.actor.model.model_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            processor=self.processor,
            hash_server=self.hash_server,
            rule_based_judge_server=self.rule_based_judge_server,
            embed_model=self.embed_model,
            embed_tokenizer=self.embed_tokenizer,
        )
        
        # ── 初始化 FSDP ↔ vLLM 切换管理器 ──
        # 当进入 rollout 模式时：
        #   - 把 FSDP 模型的最新权重同步到 vLLM
        #   - vLLM 的 GPU 显存被激活（从 sleep 状态唤醒）
        # 当退出 rollout 模式时：
        #   - vLLM 进入 sleep 状态（释放 KV cache 显存）
        #   - GPU 显存重新分配给 FSDP 训练
        self.rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.fsdp_module,
            inference_engine=self.rollout.inference_engine,
            device_mesh=rollout_device_mesh,
        )
        
        print_gpu_memory_usage("After vllm init")
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        初始化模型（被 Ray 以 ONE_TO_ALL 模式调用，所有 Worker 进程都会执行）。
        
        根据当前 Worker 的角色，初始化对应的组件：
        - Actor/Critic/Ref：调用 _build_model_optimizer，初始化 FSDP 模型和优化器
        - Rollout：调用 _build_rollout，初始化 vLLM 推理引擎
        - Actor/Critic：初始化 DataParallelPPO* 包装类、FLOP 计数器、检查点管理器
        """
        
        # 根据角色选择对应的配置
        if self._is_critic:
            model_config = self.config.critic.model
            fsdp_config = self.config.critic.fsdp
            optim_config = self.config.critic.optim
            padding_free = self.config.critic.padding_free
            role = "critic"
        elif self._is_actor:
            model_config = self.config.actor.model
            fsdp_config = self.config.actor.fsdp
            optim_config = self.config.actor.optim
            padding_free = self.config.actor.padding_free
            role = "actor"
        elif self._is_ref:
            # Ref 共享 Actor 的模型权重（但用 Ref Policy 的 FSDP 配置）
            model_config = self.config.actor.model
            fsdp_config = self.config.ref.fsdp
            optim_config = None  # Ref 不需要优化器
            padding_free = self.config.ref.padding_free
            role = "ref"
        else:
            raise ValueError(f"Unknown role {role}.")
        
        # 初始化 FSDP 模型（所有有模型的角色都需要）
        if self._is_actor or self._is_critic or self._is_ref:
            self._build_model_optimizer(
                model_config=model_config,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                padding_free=padding_free,
            )
            
            # 如果启用了参数卸载，初始化后立即把模型卸载到 CPU
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")
            
            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)
                print_gpu_memory_usage(f"After offload {role} optimizer during init")
        
        # 初始化 Actor 训练封装类
        if self._is_actor:
            from .actor.dp_actor import DataParallelPPOActor  # 延迟导入避免循环依赖
            
            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.fsdp_module,
                actor_optimizer=self.optimizer,
            )
        
        # 初始化 Critic 训练封装类
        if self._is_critic:
            from .critic.dp_critic import DataParallelPPOCritic  # 延迟导入
            
            self.critic = DataParallelPPOCritic(
                config=self.config,
                critic_module=self.fsdp_module,
                critic_optimizer=self.optimizer,
            )
        
        # 初始化 vLLM Rollout
        if self._is_rollout:
            self._build_rollout()
        
        # 初始化 Ref Policy
        if self._is_ref:
            from .actor.dp_actor import DataParallelPPOActor
            
            # Ref Policy 复用 DataParallelPPOActor（只用其 compute_log_prob 方法，不做梯度更新）
            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.fsdp_module,
            )
        
        # 初始化辅助工具（只有 Actor/Critic 需要）
        if self._is_actor or self._is_critic:
            # FLOP 计数器（用于计算 MFU，监控 GPU 利用率）
            self.flops_counter = FlopsCounter(self.model_config)
            
            # 检查点管理器（保存/恢复训练状态）
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
            )
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str):
        """
        保存检查点（被 Ray 以 ONE_TO_ALL 模式调用，所有 Worker 都保存各自的分片）。
        
        参数：path - 检查点保存路径
        """
        assert self._is_actor or self._is_critic
        
        # 如果参数在 CPU 上，先加载回 GPU（FSDP 需要 GPU 参数才能保存）
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        
        self.checkpoint_manager.save_checkpoint(path)
        dist.barrier()  # 等待所有 rank 都完成保存
        
        # 保存完成后再卸载
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        """
        加载检查点（从上次训练中断处恢复）。
        
        参数：path - 检查点加载路径
        """
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        
        self.checkpoint_manager.load_checkpoint(path)
        dist.barrier()
        
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
        
        # 避免恢复时 OOM：优化器状态可能很大，加载完后立即卸载
        if self._use_optimizer_offload:
            offload_fsdp_optimizer(self.optimizer)
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        """
        执行一次 PPO/GRPO 梯度更新（Actor 的核心训练步骤）。
        
        被 Ray 以 DP_COMPUTE_PROTO 模式调用：
        - data 会按 DP（数据并行）维度分片，每个 Worker 处理一部分
        - 内部通过 FSDP allreduce 同步梯度
        
        流程：
        1. 把 data 移到 GPU
        2. 如有需要，从 CPU 加载模型和优化器
        3. 用 Ulysses 分片管理器包裹（处理序列并行的数据重排）
        4. 调用 actor.update_policy 计算 PPO loss 并反向传播
        5. 计算 MFU、显存使用量等性能指标
        6. 更新学习率调度器
        7. 卸载模型和优化器（如果启用了 offload）
        
        参数：data - 包含 old_log_probs、advantages、response_mask 等的 DataProto
        返回：包含各种训练指标的 DataProto（非张量批次格式）
        """
        assert self._is_actor
        
        # 把数据移到当前 GPU
        data = data.to(torch.cuda.current_device())
        
        # 从 CPU 加载回 GPU（如果启用了 param offload）
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        
        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)
        
        with self.ulysses_sharding_manager:
            # 序列并行数据预处理（如果启用 Ulysses）
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            
            # 执行 PPO/GRPO 策略更新（含前向传播 + 反向传播 + 优化器步骤）
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            
            # 计算 MFU（模型计算利用率）
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )
            
            # 记录显存使用量
            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
            
            # 更新学习率（每次 update 后 step 一次）
            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            
            # 打包指标为 DataProto 格式（non_tensor_batch 存放标量/列表指标）
            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value)
                    for key, value in metrics.items()
                }
            )
        
        # 训练完成后卸载模型
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
        
        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)
        
        # 移到 CPU（返回给 Ray 主进程）
        output = output.to("cpu")
        return output
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        """
        生成候选回答（Rollout 阶段）。
        
        根据 prompts.meta_info["mode"] 选择生成模式：
        - "test"：标准推理（只生成，不收集 latent）
        - "train_pre_gen"：Monet 的离线预生成模式（只收集回答，不用 LatentRecorder）
        - "train_pre_gen_online"：Monet 的在线预生成模式（用 LatentRecorder 收集 latent）
        - "train_rl_gen"：RL 训练的主生成（greedy 或 monet）
        
        流程：
        1. 同步 FSDP 模型的最新参数到 vLLM（通过 rollout_sharding_manager）
        2. 调用 vLLMRollout.generate_sequences / generate_sequences_monet 生成
        3. 后处理数据格式
        
        参数：prompts - 包含题目的 DataProto
        返回：包含完整序列（prompt + response）的 DataProto
        """
        assert self._is_rollout
        
        # 加载 FSDP 模型到 GPU（如果在 CPU 上）
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        
        # 准备 meta_info（EOS token ID、pad token ID）
        meta_info = {
            "eos_token_id": (
                self.generation_config.eos_token_id
                if self.generation_config is not None
                else self.tokenizer.eos_token_id
            ),
            "pad_token_id": (
                self.generation_config.pad_token_id
                if self.generation_config is not None
                else self.tokenizer.pad_token_id
            ),
        }
        prompts.meta_info.update(meta_info)
        
        with self.rollout_sharding_manager:
            # 进入 rollout_sharding_manager 的 __enter__：
            # 1. 把 FSDP 最新参数同步到 vLLM
            # 2. 如有需要，卸载 FSDP 模型释放显存给 vLLM
            
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
            
            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)
            
            # 按 vLLM 的 tensor parallel 分片重排 prompts
            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            
            mode = prompts.meta_info["mode"]  # 获取生成模式
            
            if mode == "test":
                # 测试模式：标准生成
                output = self.rollout.generate_sequences(prompts=prompts)
            
            elif mode == "train_pre_gen":
                # 训练预生成（离线）：只生成回答，不记录 latent
                if self.config.rollout.sampling_strategy in ["monet"]:
                    output = self.rollout.generate_sequences(prompts=prompts)
                else:
                    raise NotImplementedError(
                        f"Sampling strategy {self.config.rollout.sampling_strategy} not supported for {mode} mode."
                    )
            
            elif mode == "train_pre_gen_online":
                # 训练预生成（在线）：生成回答 + 用 LatentRecorder 收集 latent
                if self.config.rollout.sampling_strategy in ["monet"]:
                    output = self.rollout.generate_sequences_monet(prompts=prompts)
                else:
                    raise NotImplementedError(
                        f"Sampling strategy {self.config.rollout.sampling_strategy} not supported for {mode} mode."
                    )
            
            elif mode == "train_rl_gen":
                # RL 训练主生成
                if self.config.rollout.sampling_strategy == "greedy":
                    # 贪婪解码（用于计算 ReMax 的基线）
                    output = self.rollout.generate_sequences(prompts=prompts)
                elif self.config.rollout.sampling_strategy in ["monet"]:
                    # VLPO 采样（含 LatentRecorder）
                    output = self.rollout.generate_sequences_monet(prompts=prompts)
                else:
                    raise NotImplementedError(
                        f"Sampling strategy {self.config.rollout.sampling_strategy} not supported for {mode} mode."
                    )
            else:
                raise NotImplementedError(f"Mode {mode} not supported.")
            
            # 后处理（从 vLLM 的 tp 分片格式转换回标准格式）
            output = self.rollout_sharding_manager.postprocess_data(output)
        
        # 移到 CPU 返回
        output = output.to("cpu")
        return output
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_probs(self, data: DataProto):
        """
        用当前策略（Actor）计算生成序列的对数概率。
        
        用途：PPO 中需要"旧策略的 log_probs"（生成时的概率）和"当前策略的 log_probs"（更新时的概率）
        来计算 ratio = π/π_old。这里计算的是"旧策略"的概率（在 rollout 后立即调用）。
        
        参数：data - 包含完整序列（input_ids、attention_mask、position_ids）的 DataProto
        返回：包含 old_log_probs 的 DataProto
        """
        assert self._is_actor
        
        data = data.to(torch.cuda.current_device())
        
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        
        # 设置 temperature（对数概率计算时需要）
        data.meta_info["temperature"] = self.config.rollout.temperature
        
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data)  # FSDP 前向传播，不计算梯度
            output = DataProto.from_dict(
                tensors={"old_log_probs": output},
                meta_info={"temperature": self.config.rollout.temperature}
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)
        
        # FSDP 注意：需要在返回前重新分片根模块（防止内存泄漏）
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)
        
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
        
        output = output.to("cpu")
        return output
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_probs(self, data: DataProto):
        """
        用参考策略（Ref Policy）计算对数概率（用于 KL 散度惩罚）。
        
        Ref Policy 是 RL 训练开始时固定的模型（不更新参数），
        其对数概率作为 KL 散度计算的基线。
        
        KL 散度：KL(π||π_ref) = log π(a|s) - log π_ref(a|s)
        
        参数：data - 包含完整序列的 DataProto
        返回：包含 ref_log_probs 的 DataProto
        """
        assert self._is_ref
        
        data = data.to(torch.cuda.current_device())
        
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        
        data.meta_info["temperature"] = self.config.rollout.temperature
        
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)  # 不更新参数（ref 是冻结的）
            output = DataProto.from_dict(tensors={"ref_log_probs": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)
        
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)
        
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
        
        output = output.to("cpu")
        return output
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        """
        用价值网络（Critic）计算每个 token 的价值估计（PPO 专用，GRPO 不需要）。
        
        价值估计用于 GAE advantage 计算：
        A_t = r_t + γ × V_{t+1} - V_t
        
        参数：data - 包含完整序列的 DataProto
        返回：包含 values 的 DataProto
        """
        assert self._is_critic
        
        data = data.to(torch.cuda.current_device())
        
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
        
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
        
        output = output.to("cpu")
        return output
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        """
        执行价值网络的梯度更新（Critic 训练步骤，PPO 专用）。
        
        流程与 update_actor 类似，但使用 critic.update_critic 方法。
        
        参数：data - 包含 returns 等的 DataProto
        返回：包含 Critic 训练指标的 DataProto
        """
        data = data.to(torch.cuda.current_device())
        
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        
        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)
        
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            
            # 计算 MFU
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_critic"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )
            
            # 更新 Critic 学习率
            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr
            
            output = DataProto(
                non_tensor_batch={
                    metric: np.array([value] if np.isscalar(value) else value)
                    for metric, value in metrics.items()
                }
            )
        
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
        
        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)
        
        output = output.to("cpu")
        return output
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_embed_service(self):
        """返回嵌入服务的引用（用于步骤语义聚类，供外部调用）。"""
        return self.embed_service
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_embeds(self, data: DataProto):
        """
        计算推理步骤的嵌入向量（用于步骤语义聚类）。
        
        VLPO 的 StepHashDict 需要用嵌入向量来判断"两个步骤是否语义等价"。
        这个方法对 batch 里的每组步骤文字批量计算嵌入向量。
        
        参数：data.non_tensor_batch["steps"] - 每个样本的步骤文字列表
        返回：包含 embeds 字段的 DataProto（嵌入向量数组）
        """
        assert self._is_rollout
        
        batch_steps: np.array = data.non_tensor_batch["steps"]  # List of List[str]
        batch_embeds = []
        
        for steps in batch_steps:
            batch_embeds.append(self.compute_embeds_fn(steps))
        
        return DataProto(non_tensor_batch={"embeds": np.array(batch_embeds, dtype=object)})
    
    def compute_embeds_fn(self, texts):
        """
        对文字列表计算嵌入向量（调用 vLLM embed 模型）。
        
        参数：texts - 文字字符串列表
        返回：嵌入向量数组（形状：(len(texts), embed_dim)）
        """
        # 调用 vLLM embed 模型的 embed 接口
        outputs = self.embed_model.embed(texts, use_tqdm=False)
        
        # 提取嵌入向量并转换格式
        return torch.tensor(
            [o.outputs.embedding for o in outputs]
        ).detach().half().cpu().numpy().copy()
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rule_based_judge(self, data: DataProto):
        """
        对一批回答进行正确性判断（规则判断 + 可选 API 判断）。
        
        这是 RL 训练中"两阶段判断"的主要入口：
        1. 先用规则（数学等价性）快速判断
        2. 规则判断失败的答案，再调用 LLM API 做补充判断（可选）
        
        支持的判断策略：
        - "rule_then_api_batch_judge"：规则先判，错的才送 API（推荐）
        - "extract_and_check"：只用规则
        - "extract_and_check_api"：每个都调用 API
        
        结果存入 DataProto.non_tensor_batch["correctness"]：
        - 1.0：正确
        - 0.0：错误
        - -1.0：重复惩罚（repetition_penalty 模式）
        
        参数：data - 包含 responses、ground_truth、problem 的 DataProto
        返回：包含 correctness 和 response_strs 的 DataProto
        """
        assert self._is_rollout
        
        correctness = []
        response_strs = []
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        
        # 确定进度条位置（避免多进程进度条重叠）
        if self.config.rollout.offline_difficulty_sampling:
            position = 2
        elif self.config.rollout.online_difficulty_sampling:
            position = 3
        
        if self.config.rule_based_judge.judge_function_name == "rule_then_api_batch_judge":
            # ── 两阶段批量判断（推荐模式）──
            
            # 先解码所有回答为文字
            for i in range(len(data)):
                valid_response_ids = response_ids[i][: response_length[i]]
                # skip_special_tokens=True：去掉 <abs_vis_token> 等特殊标记
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                response_strs.append(response_str)
            
            # 批量两阶段判断
            correctness = rule_then_api_batch_judge(
                questions=data.non_tensor_batch["problem"],  # 题目文字
                preds=response_strs,                         # 预测回答
                gts=data.non_tensor_batch["ground_truth"],   # 标准答案
                api_name=self.config.rule_based_judge.api_name,
                api_kwargs=self.config.rule_based_judge.api_kwargs,
                client=self.api_client,
                repetition_penalty=self.config.reward.repetition_penalty,  # 是否启用重复惩罚
            )
        else:
            # ── 逐样本规则判断（兼容旧版配置）──
            for i in tqdm(
                range(len(data)), desc="Rule-based judge",
                position=position, disable=self.rank != 0  # 只在 rank 0 显示进度条
            ):
                valid_response_ids = response_ids[i][: response_length[i]]
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                response_strs.append(response_str)
                ground_truth = data.non_tensor_batch["ground_truth"][i]
                question = data.non_tensor_batch["problem"][i]
                
                if self.config.rule_based_judge.judge_function == \
                        "./examples/reward_function/monet_reward_function.py:extract_and_check":
                    # 纯规则判断
                    correctness.append(easyr1_monet_extract_and_check(response_str, ground_truth))
                
                elif self.config.rule_based_judge.judge_function == \
                        "./examples/reward_function/monet_reward_function.py:extract_and_check_api":
                    # 每个都调用 API 判断（慢但更准确）
                    correctness.append(
                        easyr1_monet_extract_and_check_api(question, response_str, ground_truth, self.api_client)
                    )
                else:
                    raise NotImplementedError(
                        f"Rule-based judge function {self.config.rule_based_judge.judge_function} not supported."
                    )
        
        return DataProto(non_tensor_batch={
            "correctness": correctness,      # 正确性列表（1.0/0.0/-1.0）
            "response_strs": response_strs   # 解码后的回答文字列表
        })
