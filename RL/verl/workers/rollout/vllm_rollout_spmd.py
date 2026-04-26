"""
本文件的作用：vLLM Rollout Worker（RL 训练中的"生成回答"阶段）

在 RL 训练中，"Rollout"指的是用当前策略（模型）对一批题目生成回答的过程。
这个文件定义了 vLLMRollout 类，负责：
1. 加载 vLLM 推理引擎（含 Monet 的 latent 推理补丁）
2. 对每批题目生成 n 个候选回答（sampling_params.n > 1）
3. 收集每个回答对应的 latent 向量（Monet 专用）
4. 把生成结果打包成 DataProto 格式，传给奖励计算和 FSDP 训练 Worker

两个主要生成函数：
- generate_sequences：标准 GRPO 模式（不收集 latent）
- generate_sequences_monet：VLPO 模式（用 LatentRecorder 收集 latent 向量）

工作流程（对应 RayPPOTrainer.fit() 里的一个训练步骤）：
    题目 batch
    → vLLMRollout.generate_sequences_monet()
       → vLLM.generate()（用 LatentRecorder 包裹，收集 latent 向量）
       → 查询 StepHashServer 获取参考回答长度（用于长度惩罚）
    → 返回 DataProto（含 responses、latents、ref_resp_lengths 等）
    → 传给奖励计算 Worker（计算 reward）
    → 传给 FSDP Actor（计算 policy loss 并更新参数）
"""

# 操作系统接口（读取环境变量、文件操作）
import os
# 上下文管理器工具（用于 update_sampling_params 的临时修改）
from contextlib import contextmanager
# 调试工具
import pdb
# 类型注解
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence

# 数值计算（用于处理 numpy 数组格式的 non_tensor_batch 字段）
import numpy as np
# PyTorch 张量操作
import torch
# PyTorch 分布式通信（多 GPU 协调）
import torch.distributed
# TensorDict：一种支持嵌套张量的字典格式（用于 DataProto）
from tensordict import TensorDict
# HuggingFace tokenizer 和 processor 基类
from transformers import PreTrainedTokenizer, ProcessorMixin
# vLLM 核心类：推理引擎、输出格式、采样参数
from vllm import LLM, RequestOutput, SamplingParams

# verl 数据格式
from ...protocol import DataProto
# 张量工具函数（pad、masked_mean 等）
from ...utils import torch_functional as VF
# processor 加载工具（用于获取图像 token ID）
from ...utils.tokenizer import get_processor
# 精度类型转换工具（如 "bfloat16" ↔ torch.bfloat16）
from ...utils.torch_dtypes import PrecisionType
# BaseRollout 抽象基类
from .base import BaseRollout
# Rollout 配置类
from .config import RolloutConfig
# Ray Actor：步骤哈希服务器 + 样本哈希服务器
from tools.actors import StepHashServer, SampleHashServer
# 规则判断管理器基类
from verl.workers.reward.function import FunctionRuleBasedJudgeManager
# Ray 分布式框架（用于调用 StepHashServer 等远程 Actor）
import ray
# 正则表达式（未直接使用，遗留导入）
import re

# ── Monet 专用导入 ──
# LatentRecorder：收集 vLLM 推理过程中生成的 latent 向量的记录器
from monet_models.vllm.latent_recorder import LatentRecorder
# 文件操作工具
import os, json, shutil, tempfile, pathlib


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    """
    把张量或 numpy 数组在第 0 维上进行"交错复制"。
    
    示例（repeats=2）：
    输入：[A, B, C]
    输出：[A, A, B, B, C, C]
    
    用途：当每道题采样 n 个回答时，需要把 prompt 也扩展 n 倍，
    使 prompt 和回答一一对应（每个 prompt 对应 n 个回答）。
    
    参数：
    - value：待扩展的张量或 numpy 数组
    - repeats：每个元素复制的次数
    
    返回：
    - 扩展后的张量或 numpy 数组
    """
    if isinstance(value, torch.Tensor):
        # PyTorch 张量：使用 repeat_interleave
        return value.repeat_interleave(repeats, dim=0)
    else:
        # NumPy 数组：使用 np.repeat（行为与 repeat_interleave 相同）
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(model_path: str, trust_remote_code: bool) -> Optional[Dict[int, float]]:
    """
    获取需要偏置（logit bias）的 token ID 映射。
    
    用途：对图像 token 施加极大负偏置（-100），防止模型在回答时主动生成图像占位符。
    
    参数：
    - model_path：模型路径（用于加载 processor）
    - trust_remote_code：是否允许加载自定义代码
    
    返回：
    - {image_token_id: -100}（有图像 token 时）或 None（无图像 token 时）
    """
    # 加载 processor，从中获取图像 token 的 ID
    processor = get_processor(model_path, trust_remote_code=trust_remote_code)
    
    if processor is not None and hasattr(processor, "image_token"):
        # 获取图像占位符（如 "<image>"）对应的 token ID
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        # 对该 token 施加 -100 的 logit 偏置（相当于无限禁止）
        return {image_token_id: -100}
    else:
        return None


def remove_text_config_inplace(path: str) -> bool:
    """
    从模型目录的 config.json 中删除 'text_config' 字段。
    
    背景：某些版本的 Qwen2.5-VL 的 config.json 中有 'text_config' 字段，
    会导致 vLLM 解析出错。这个函数在加载模型前把这个字段删掉。
    
    使用原子写入（临时文件 + rename）防止写入过程中意外中断导致 config.json 损坏。
    
    参数：
    - path：模型目录路径（或直接指向 config.json 的路径）
    
    返回：
    - True：成功删除了 text_config 字段
    - False：原本就没有 text_config 字段，无需修改
    
    异常：
    - FileNotFoundError：如果 config.json 不存在
    """
    # 定位 config.json 文件路径
    cfg_path = path
    if os.path.isdir(cfg_path):
        cfg_path = os.path.join(cfg_path, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found at: {cfg_path}")
    
    # 加载 config.json
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    # 如果没有 text_config 字段，直接返回（无需修改）
    if "text_config" not in cfg:
        return False
    
    # 删除 text_config 字段
    del cfg["text_config"]
    
    # 原子写回：先写临时文件，再用 os.replace 替换原文件
    # 这样即使写入中途崩溃，原文件也不会损坏
    dir_name = os.path.dirname(cfg_path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=dir_name) as tmp:
        json.dump(cfg, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())  # 强制把内存缓冲写入磁盘
        tmp_name = tmp.name
    
    # 原子替换
    os.replace(tmp_name, cfg_path)
    return True


class vLLMRollout(BaseRollout):
    """
    基于 vLLM 的 Rollout Worker。
    
    在 RL 训练中，Rollout 是"用当前策略采样生成回答"的阶段。
    vLLMRollout 使用 vLLM 高效地生成多个候选回答（n 个/题），
    并在 Monet 模式下额外收集 latent 向量。
    
    与 FSDPWorker 的关系：
    - vLLMRollout 负责"生成"（inference，不计算梯度）
    - FSDPWorker 负责"训练"（training，计算梯度）
    - 同一个 FSDPWorker 可以切换 vLLM 和 FSDP 两种模式
    """
    
    def __init__(
        self,
        model_path: str,               # 模型目录路径
        config: RolloutConfig,         # Rollout 配置
        tokenizer: PreTrainedTokenizer, # 文本 tokenizer
        processor: Optional[ProcessorMixin],  # 多模态 processor（可选）
        hash_server: Optional[Union[StepHashServer, SampleHashServer]] = None,  # 步骤哈希服务器（Ray Actor 引用）
        rule_based_judge_server: Optional[FunctionRuleBasedJudgeManager] = None,  # 规则判断服务器（Ray Actor 引用）
        embed_model=None,              # 嵌入模型（用于步骤相似度计算，可选）
        embed_tokenizer=None           # 嵌入模型的 tokenizer（可选）
    ):
        """
        初始化 vLLM 推理引擎和所有相关服务。
        
        初始化完成后，调用 generate_sequences() 或 generate_sequences_monet() 开始生成。
        """
        super().__init__()
        
        # 获取当前进程的 rank（在分布式环境中，rank 0 是主进程）
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.processor = processor
        self.tokenizer = tokenizer
        
        # 参数校验
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")
        
        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")
        
        # 修复 config.json 中可能的兼容性问题（删除 text_config 字段）
        remove_text_config_inplace(model_path)
        
        # ── 初始化 vLLM 推理引擎 ──
        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,            # 加载 tokenizer（vLLM 内部也需要 tokenizer）
            trust_remote_code=config.trust_remote_code,  # 允许自定义代码
            load_format="auto",                   # 自动检测模型格式
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),  # 精度（通常 bfloat16）
            seed=config.seed,                     # 随机种子
            # 最大序列长度（prompt + response），取配置中的 max_model_len 或自动计算
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            # "external_launcher" 表示使用外部的分布式启动（与 Ray FSDP 框架集成）
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,  # 张量并行 GPU 数
            gpu_memory_utilization=config.gpu_memory_utilization,  # GPU 显存使用比例
            max_num_batched_tokens=config.max_num_batched_tokens,  # 最大批次 token 数
            disable_log_stats=config.disable_log_stats,            # 禁用 vLLM 统计日志
            enforce_eager=config.enforce_eager,                     # 禁用 CUDA graph（Monet 补丁需要）
            disable_custom_all_reduce=True,                         # 禁用自定义 allreduce（与 Ray 不兼容）
            limit_mm_per_prompt={"image": config.limit_images},     # 每个 prompt 最多几张图片
            enable_chunked_prefill=config.enable_chunked_prefill,   # 分块预填充（提升长序列处理效率）
            enable_sleep_mode=True,                                  # 允许引擎空闲时休眠（释放显存）
        )
        
        # 让 vLLM 引擎进入休眠状态（level=1 = 卸载 KV cache，节省显存）
        # 在 FSDP 训练时，vLLM 不需要占用显存
        self.inference_engine.sleep(level=1)
        
        # ── 配置采样参数 ──
        
        # 基础采样参数
        sampling_kwargs = {
            "max_tokens": config.response_length,  # 最大生成 token 数
            "detokenize": False,                   # 不解码（让外部处理 token ID）
            # 对图像 token 施加负无穷偏置（防止模型生成图像占位符）
            "logit_bias": _get_logit_bias(model_path, trust_remote_code=config.trust_remote_code),
        }
        
        # 从 RolloutConfig 中提取所有与 SamplingParams 同名的字段（如 temperature、top_p 等）
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)
        
        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)
        
        # 保存服务器引用
        self.hash_server = hash_server                     # 步骤哈希/样本哈希服务器（Ray Actor 引用）
        self.rule_based_judge_server = rule_based_judge_server  # 规则判断服务器（Ray Actor 引用）
        self.embed_model = embed_model                     # 嵌入模型（用于步骤相似度）
        self.embed_tokenizer = embed_tokenizer
        
        # 从环境变量读取 latent 向量大小（每个 latent token 生成的向量数量）
        # 在 vlpo_train.sh 里通过 LATENT_SIZE=10 设置
        self.latent_size = int(os.getenv("ABS_VIS_LATENT_SIZE", '0'))
    
    @contextmanager
    def update_sampling_params(self, **kwargs):
        """
        临时修改采样参数的上下文管理器。
        
        进入 with 块时：把 kwargs 中的参数临时覆盖到 self.sampling_params
        退出 with 块时：恢复原来的参数值
        
        用途：某些 batch 可能需要特定的采样参数（如验证时 temperature=0）。
        
        示例：
        with self.update_sampling_params(temperature=0):
            completions = self.inference_engine.generate(...)  # 贪婪解码
        # 离开 with 块后，sampling_params.temperature 恢复原值
        """
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        
        yield  # 执行 with 块内的代码
        
        # 恢复原来的采样参数
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)
    
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        标准 GRPO 模式的生成函数（不收集 latent 向量）。
        
        流程：
        1. 从 prompts 中提取 input_ids 和图像数据，构建 vLLM 输入格式
        2. 用 vLLM 批量生成回答（每道题 n 个）
        3. 查询参考回答长度（用于长度惩罚）
        4. 把 prompt 扩展 n 倍，与 n 个回答对齐
        5. 打包成 DataProto 返回
        
        参数：prompts - 包含 input_ids、attention_mask、position_ids 等的 DataProto
        返回：包含完整序列（prompt + response）的 DataProto
        """
        # 提取基础信息
        input_ids: torch.Tensor = prompts.batch["input_ids"]   # (batch_size, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]  # EOS token 的 ID
        batch_size = input_ids.size(0)
        
        non_tensor_batch = prompts.non_tensor_batch
        
        # 如果是 "monet" 采样策略，取出题目的全局 ID（用于查询哈希服务器）
        if self.config.sampling_strategy in ["monet"]:
            batch_sample_idx = list(non_tensor_batch.pop("global_index"))
        
        # 检查 vLLM 分片管理是否正常（每个 rank 应该处理相同数量的样本）
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")
        
        # ── 构建 vLLM 输入格式 ──
        if "multi_modal_data" in non_tensor_batch:
            # 多模态输入（含图片）：每个样本包含 prompt_token_ids 和 multi_modal_data
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({
                    "prompt_token_ids": list(raw_prompt_ids),
                    "multi_modal_data": multi_modal_data
                })
        else:
            # 纯文本输入
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids)}
                for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]
        
        # 收集参考回答长度（用于长度惩罚）
        batch_min_mean_correct_resp_lens = []
        
        with self.update_sampling_params(**prompts.meta_info):
            # 调用 vLLM 生成
            completions: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=(self.rank == 0)  # 只在主进程显示进度条
            )
            
            # 提取所有生成的 token ID 序列
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            
            if self.config.sampling_strategy in ["monet"]:
                # Monet 策略：收集每道题的参考回答长度
                response_ids = []
                for completion, global_id in zip(completions, batch_sample_idx):
                    for output in completion.outputs:
                        response_ids.append(output.token_ids)
                    
                    # 从哈希服务器查询这道题历史正确回答的最短/均值长度
                    min_len, mean_len = ray.get(
                        self.hash_server.look_up_min_mean_correct_resp_len.remote(global_id)
                    )
                    # 为这道题的 n 个回答都填入相同的参考长度
                    batch_min_mean_correct_resp_lens.extend(
                        [min_len] * self.sampling_params.n if min_len < float("inf")
                        else [mean_len] * self.sampling_params.n
                    )
            
            # ── 把变长的 response_ids 列表 padding 成统一长度的张量 ──
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)
            
            # 如果每道题采样了 n > 1 个回答，把 prompt 也扩展 n 倍
            # 扩展后：batch_size = 原 batch_size × n
            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
        
        # ── 构建完整序列（prompt + response）──
        
        # 拼接 prompt 和 response 的 token ID
        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        
        response_length = response_ids.size(1)
        
        # 计算 response 的 position IDs（从 prompt 的最后一个 position 开始递增）
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        
        if position_ids.dim() == 3:
            # Qwen2.5-VL 使用 MRoPE（多模态旋转位置编码），position_ids 是 3 维的 (bs, 3, seq_len)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
        
        # 拼接 prompt 和 response 的 position IDs
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        # 计算 response 的 attention mask
        # response_mask：有效 token 为 1，EOS 之后的 padding 为 0
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        
        # 拼接 prompt 和 response 的 attention mask
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)
        
        # 保存参考回答长度
        if self.config.sampling_strategy in ["monet"]:
            non_tensor_batch["ref_resp_lengths"] = np.array(batch_min_mean_correct_resp_lens)
        
        # ── 打包成 DataProto 返回 ──
        batch = TensorDict(
            {
                "prompts": input_ids,            # 原始 prompt 的 token IDs
                "responses": response_ids,       # 生成的 response 的 token IDs
                "input_ids": sequence_ids,       # 完整序列（prompt + response）的 token IDs
                "attention_mask": attention_mask, # 完整序列的 attention mask
                "response_mask": response_mask,   # 只有 response 部分的 mask
                "position_ids": position_ids,    # 完整序列的 position IDs
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    
    @torch.no_grad()
    def generate_sequences_monet(self, prompts: DataProto) -> DataProto:
        """
        VLPO 模式的生成函数（收集 latent 向量）。
        
        与 generate_sequences 的区别：
        1. 使用 LatentRecorder 上下文管理器包裹 vLLM 生成，捕获每个 latent token 的向量
        2. 把捕获的 latent 向量存入 non_tensor_batch["latents"]
        3. latent 向量会用于 VLPO 的优势估计和反向传播
        
        LatentRecorder 的工作原理：
        - 在 Monet 补丁的 GPU model runner 里，每当生成一个 latent token，
          就把对应的隐状态向量发送给 LatentRecorder
        - LatentRecorder 通过 TCP 或 UNIX socket 收集这些向量
        - 生成结束后，to_object_array_auto() 把收集到的向量整理成 batch 格式
        
        参数：prompts - 同 generate_sequences
        返回：包含 latents 字段的 DataProto（比 generate_sequences 多一个字段）
        """
        # ── 提取输入信息 ──
        
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)
        
        non_tensor_batch = prompts.non_tensor_batch
        
        # Monet 模式必须有 global_index（用于查询哈希服务器）
        batch_sample_idx = list(non_tensor_batch.pop("global_index"))
        
        # 保存题目文字（用于 API 判断时传入 question）
        gts = list(non_tensor_batch["ground_truth"])       # 标准答案列表
        questions = list(non_tensor_batch["problem"])       # 题目文字列表
        
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")
        
        # ── 构建 vLLM 输入 ──
        
        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({
                    "prompt_token_ids": list(raw_prompt_ids),
                    "multi_modal_data": multi_modal_data
                })
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids)}
                for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]
        
        batch_min_mean_correct_resp_lens = []
        
        with self.update_sampling_params(**prompts.meta_info):
            # ── 关键：用 LatentRecorder 包裹生成，收集 latent 向量 ──
            with LatentRecorder(set_env=True, prefer_tcp=True, filter_rank=self.rank) as rec:
                # vLLM 生成（Monet 补丁会在生成 latent token 时向 LatentRecorder 发送向量）
                completions: List[RequestOutput] = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=self.sampling_params,
                    use_tqdm=(self.rank == 0)
                )
                # 提取所有生成的 token ID
                response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            
            # 找出最小的 request_id（用于 LatentRecorder 的对齐）
            min_req_id = 99999
            for completion in completions:
                min_req_id = min(min_req_id, int(completion.request_id))
            
            # 把 LatentRecorder 收集到的 latent 向量整理成对象数组
            # bsz 是原始 batch 大小，rollout_n 是每道题采样的回答数
            non_tensor_batch['latents'] = rec.to_object_array_auto(
                bsz=batch_size,
                rollout_n=self.sampling_params.n,
                min_req_id=min_req_id
            )
            
            # ── 收集 response_ids 和参考回答长度 ──
            response_ids = []
            for completion, global_id in zip(completions, batch_sample_idx):
                for output in completion.outputs:
                    response_ids.append(output.token_ids)
                
                # 查询这道题历史正确回答的最短/均值长度
                min_len, mean_len = ray.get(
                    self.hash_server.look_up_min_mean_correct_resp_len.remote(global_id)
                )
                batch_min_mean_correct_resp_lens.extend(
                    [min_len] * self.sampling_params.n if min_len < float("inf")
                    else [mean_len] * self.sampling_params.n
                )
            
            # Padding response_ids
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)
            
            # 扩展 prompt 到 n 倍
            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
        
        # ── 构建完整序列 ──
        
        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        
        if position_ids.dim() == 3:  # Qwen2.5-VL 的 MRoPE
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
        
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)
        
        # 保存参考回答长度和扩展后的 ground_truth / problem
        non_tensor_batch["ref_resp_lengths"] = np.array(batch_min_mean_correct_resp_lens)
        
        # 把 ground_truth 和 problem 也扩展 n 倍（与回答对应）
        non_tensor_batch["ground_truth"] = _repeat_interleave(np.array(gts), self.sampling_params.n)
        non_tensor_batch["problem"] = _repeat_interleave(np.array(questions), self.sampling_params.n)
        
        # ── 打包返回 ──
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
