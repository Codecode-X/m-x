"""
本文件的作用：RL 训练的主入口（Main Entry Point）

这是整个 RL 训练流程的起点，对应 vlpo_train.sh 脚本里的：
    python -m verl.trainer.main [配置参数]

执行顺序（从 main() 开始）：
1. 解析命令行参数 → 合并默认配置 + YAML 文件配置 + 命令行参数 → PPOConfig
2. 初始化 Ray 分布式计算框架（管理多 GPU 的任务调度）
3. 创建 Runner Actor 并调用 runner.run()，在 Ray 集群里启动实际的训练
4. run() 内部：
   - 加载 tokenizer 和 processor
   - 创建 GPU Worker 映射（Actor/Critic/RefPolicy → 哪些 GPU）
   - 创建奖励函数管理器（RewardManager）和规则判断管理器（RuleBasedJudgeManager）
   - 创建数据加载器（train/val）
   - 创建 RayPPOTrainer 并调用 trainer.init_workers() + trainer.fit()

关键组件关系图：
    main() 
      └─ Ray.init()：初始化分布式框架
      └─ Runner.remote()：在 Ray 里启动 Runner Actor
           └─ tokenizer/processor：加载模型的文本/视觉处理器
           └─ FSDPWorker → Actor/Critic/RefPolicy（FSDP 全参模型）
           └─ RewardManager → 奖励函数（计算回答得分）
           └─ RuleBasedJudgeManager → 规则判断服务（先规则、后 API）
           └─ DataLoader → 训练/验证数据集
           └─ RayPPOTrainer.fit() → 实际的 PPO/GRPO/VLPO 训练循环
"""

# 导入 Monet RL 补丁（必须最先导入，替换 verl/vLLM 里的标准实现）
# 等同于 SFT 里的 apply_qwen2_5_monet.py，但针对 verl/vLLM 的 RL 框架
import monet_rl_patch

# 导入 JSON 工具（用于打印配置，调试用）
import json
# 调试工具（实际训练时不使用）
import pdb
# 导入 Ray 分布式计算框架
import ray
# 导入 OmegaConf：结构化配置管理工具（支持 YAML + 命令行参数合并）
from omegaconf import OmegaConf

# 导入 Ray Worker 组管理类（负责在 Ray 集群里创建和调度 GPU Worker）
from ..single_controller.ray import RayWorkerGroup
# 导入 tokenizer/processor 加载工具
from ..utils.tokenizer import get_processor, get_tokenizer
# 导入 FSDP Worker 类（Actor/Critic/RefPolicy 都用这个，区别在于运行时的角色）
from ..workers.fsdp_workers import FSDPWorker
# 导入四种奖励/判断管理器：
# - BatchFunctionRewardManager：批量奖励函数管理器
# - SequentialFunctionRewardManager：顺序奖励函数管理器
# - BatchFunctionRuleBasedJudgeManager：批量规则判断管理器
# - SingleFunctionRuleBasedJudgeManager：单样本规则判断管理器
from ..workers.reward import (
    BatchFunctionRewardManager,
    SequentialFunctionRewardManager,
    BatchFunctionRuleBasedJudgeManager,
    SingleFunctionRuleBasedJudgeManager
)
# 导入 PPOConfig：整个训练流程的配置数据类
from .config import PPOConfig
# 导入数据加载器创建函数
from .data_loader import create_dataloader
# 导入 RayPPOTrainer（训练循环的核心）、资源池管理器、角色枚举
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
# 操作系统接口（读取环境变量）
import os
# 日志重定向工具（把 stdout 同时写入文件）
from verl.trainer.save_any_log import setup_tee_logger
# 日期时间工具（用于生成日志文件名）
import datetime


# ══════════════════════════════════════════════════════════════════════
# Runner：训练主逻辑的 Ray Actor 封装
# ══════════════════════════════════════════════════════════════════════

# @ray.remote(num_cpus=2)：把 Runner 变成 Ray Actor，在 Ray 集群里运行
# 注释说明"main_task is not scheduled on head"：
#   - Ray 集群有一个 head 节点（负责调度）和若干 worker 节点
#   - Runner 被调度到 worker 节点，避免占用 head 节点资源
@ray.remote(num_cpus=2)
class Runner:
    """RL 训练的执行者（Runner），负责把所有组件组装起来并启动训练。"""
    
    def run(self, config: PPOConfig):
        """
        实际的训练启动逻辑，在 Ray Actor 里运行。
        
        参数：
        - config：PPOConfig 对象，包含所有训练参数
        """
        # 局部导入（在 Ray Worker 进程里动态导入，避免序列化问题）
        import torch, os, sys
        
        # 打印环境信息（便于调试版本不兼容问题）
        print("Torch version :", torch.__version__)
        print("Torch path    :", torch.__file__)
        print("Python exec   :", sys.executable)
        print("CUDA_VISIBLE_DEVICES :", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("http_proxy =", os.getenv("http_proxy"))
        print("https_proxy=", os.getenv("https_proxy"))
        
        # ── 初始化 tokenizer 和 processor ──
        
        # 加载 tokenizer（文本 tokenization，用于编码输入文本）
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,            # 模型路径
            override_chat_template=config.data.override_chat_template,  # 可选：覆盖默认的对话模板
            trust_remote_code=config.worker.actor.model.trust_remote_code,  # 允许加载自定义代码
            use_fast=True,                                   # 使用 Rust 实现的快速 tokenizer
        )
        
        # 加载 processor（多模态处理器，包含 tokenizer + 图像预处理器）
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        
        # ── 配置 GPU Worker 和资源池 ──
        
        # 使用标准的 RayWorkerGroup（Ray 的多 GPU Worker 管理器）
        ray_worker_group_cls = RayWorkerGroup
        
        # 角色 → Worker 类型的映射
        # ActorRollout：负责生成 rollout（用 vLLM）和训练（用 FSDP）
        # Critic：价值网络（GRPO 不需要，但 PPO 需要）
        # RefPolicy：参考策略（用于 KL 散度惩罚，防止模型偏离太多）
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(FSDPWorker),   # 都用 FSDPWorker，区别在角色
            Role.Critic: ray.remote(FSDPWorker),
            Role.RefPolicy: ray.remote(FSDPWorker),
        }
        
        # 资源池 ID（所有角色共享同一个 GPU 池）
        global_pool_id = "global_pool"
        
        # 资源池规格：每个节点的 GPU 数 × 节点数
        # 例如：n_gpus_per_node=8, nnodes=1 → [8]（8 个 GPU 在 1 个节点上）
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        
        # 每个角色使用的资源池（都用 global_pool）
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        
        # 创建资源池管理器（负责在 GPU 间分配和调度 Worker）
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping
        )
        
        # ── 创建奖励函数管理器 ──
        
        # 根据配置选择奖励函数的执行方式：
        # - "sequential"：逐个样本执行（更简单，便于调试）
        # - "batch"：批量执行（更高效）
        if config.worker.reward.reward_type == "sequential":
            RewardManager = SequentialFunctionRewardManager
        elif config.worker.reward.reward_type == "batch":
            RewardManager = BatchFunctionRewardManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.worker.reward.reward_type}.")
        
        # ── 创建规则判断管理器 ──
        
        # 根据配置选择规则判断的执行方式：
        # - "single"：单样本判断（不批处理）
        # - "batch"：批量判断（更高效，支持 API 并发调用）
        if config.worker.rule_based_judge.judge_type == "single":
            RuleBasedJudgeManager = SingleFunctionRuleBasedJudgeManager
        elif config.worker.rule_based_judge.judge_type == "batch":
            RuleBasedJudgeManager = BatchFunctionRuleBasedJudgeManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.worker.rule_based_judge.judge_type}.")
        
        # 把 RewardManager 变成 Ray Actor，指定 CPU 数
        # .options(num_cpus=...) 控制这个 Actor 占用多少 CPU 核
        RemoteRewardManager = ray.remote(RewardManager).options(num_cpus=config.worker.reward.num_cpus)
        
        # 创建训练奖励函数（用于训练集的奖励计算）
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        # 创建验证奖励函数（用于验证集的奖励计算，配置相同但是独立实例）
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        
        # 创建规则判断服务（以命名 Actor 方式运行，便于跨 Worker 访问）
        RemoteRuleBasedJudgeManager = ray.remote(RuleBasedJudgeManager).options(
            num_cpus=config.worker.rule_based_judge.num_cpus,
            name="rule_based_judge_server"  # 命名 Actor，其他 Worker 可以通过名字找到它
        )
        # 把 Actor 名字存入配置（Worker 会用这个名字通过 ray.get_actor() 获取引用）
        config.worker.rule_based_judge.judge_server_name = "rule_based_judge_server"
        rule_based_judge_fn = RemoteRuleBasedJudgeManager.remote(config.worker.rule_based_judge, tokenizer)
        
        # ── 创建数据加载器 ──
        
        # 同时创建训练集和验证集的 DataLoader
        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)
        
        # ── 创建并启动训练器 ──
        
        # 把所有组件传给 RayPPOTrainer（训练循环的核心）
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            rule_based_judge=rule_based_judge_fn,
        )
        
        # 初始化所有 GPU Worker（分配 GPU、加载模型权重）
        trainer.init_workers()
        
        # 开始训练循环（fit = 拟合 = 训练）
        # 这个函数会一直运行直到训练结束（达到 max_steps 或 max_epochs）
        trainer.fit()


def main():
    """
    RL 训练的 Python 入口函数。
    
    被 vlpo_train.sh 通过 `python -m verl.trainer.main [参数]` 调用。
    
    配置加载优先级（从低到高）：
    1. PPOConfig 数据类的默认值
    2. YAML 配置文件（通过 --config=path/to/config.yaml 指定）
    3. 命令行参数（通过 key=value 格式指定，如 trainer.n_gpus_per_node=8）
    
    OmegaConf 会按优先级合并这三层配置。
    """
    
    # 从命令行解析参数（OmegaConf 格式：key=value）
    cli_args = OmegaConf.from_cli()
    
    # 加载 PPOConfig 的默认值（所有参数的默认配置）
    default_config = OmegaConf.structured(PPOConfig())
    
    # 如果命令行里指定了 config=path/to/yaml，加载该 YAML 文件
    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)  # 取出并删除 config 参数
        file_config = OmegaConf.load(config_path)   # 加载 YAML 文件
        # 把 YAML 文件配置合并到默认配置上（YAML 覆盖默认值）
        default_config = OmegaConf.merge(default_config, file_config)
    
    # 把命令行参数合并到配置上（命令行覆盖 YAML 和默认值）
    ppo_config = OmegaConf.merge(default_config, cli_args)
    
    # 把 OmegaConf 对象转换为 PPOConfig Python 对象
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    
    # 执行后处理初始化（检查参数合法性、计算派生参数等）
    ppo_config.deep_post_init()
    
    # ── 初始化 Ray 分布式框架 ──
    
    if not ray.is_initialized():
        # Ray 的运行时环境配置（所有 Worker 进程都会继承这些环境变量）
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",        # 允许 tokenizer 并行（避免 HuggingFace 警告）
                "NCCL_DEBUG": "WARN",                    # NCCL 日志级别（减少 GPU 通信日志）
                "VLLM_LOGGING_LEVEL": "WARN",            # vLLM 日志级别（减少 vLLM 日志）
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # 避免 NCCL 记录 CUDA stream（减少内存碎片）
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",  # 禁用 PyTorch 显存扩展（更稳定）
                "PYTHONUNBUFFERED": "1",                 # Python 输出不缓冲（实时显示日志）
                "RAY_DEBUG": os.getenv("RAY_DEBUG", "0"),              # Ray 调试级别
                "RAY_LOG_TO_STDERR": os.getenv("RAY_LOG_TO_STDERR", "0"),  # Ray 日志是否输出到 stderr
            }
        }
        
        # 是否使用单进程本地模式（local_mode）用于调试
        # 设置环境变量 RAY_LOCAL_MODE=1 可以强制单进程运行
        local_mode = os.getenv("RAY_LOCAL_MODE", "0").lower() in ("1", "true", "yes")
        
        # 确定 Ray 集群地址：
        # - USE_RAY_LOCAL=1（默认）→ 使用 "local"（在本机启动新的 Ray 集群）
        # - RAY_ADDRESS=xxx → 连接到已有的 Ray 集群
        address = None
        try:
            _force_local = os.getenv("USE_RAY_LOCAL", "1").lower() in ("1", "true", "yes")
        except Exception:
            _force_local = True
        
        if _force_local:
            address = "local"  # 在本机启动 Ray（最常用的单机多卡配置）
        else:
            address = os.getenv("RAY_ADDRESS", None)  # 连接到外部 Ray 集群
        
        # Ray 的临时文件目录配置
        # 使用短路径避免 AF_UNIX socket 107 字节限制（路径太长会导致 socket 连接失败）
        default_spill = "/tmp/ray_spill"    # Ray 对象溢出（内存不足时 spill 到磁盘）目录
        default_temp = "/tmp/ray_tmp"       # Ray 临时文件目录
        spill_dir = os.path.abspath(os.getenv("RAY_SPILL_DIR", default_spill))
        temp_dir = os.path.abspath(os.getenv("RAY_TMPDIR", default_temp))
        
        # 创建目录（如果不存在）
        os.makedirs(spill_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Ray 共享对象存储的内存大小（默认 128MB）
        # 训练时 GPU 张量通过 Ray 的共享内存在 Worker 间传递
        try:
            object_store_mem = int(os.getenv("RAY_OBJECT_STORE_MEMORY", str(128 * 1024 ** 2)))
        except Exception:
            object_store_mem = 128 * 1024 ** 2  # 128 MB
        
        # Worker 注册超时时间（默认 300 秒）
        # 第一次加载大模型可能需要较长时间，设置较长的超时防止 Ray 误判 Worker 死亡
        try:
            register_timeout = int(os.getenv("RAY_WORKER_REGISTER_TIMEOUT_SECONDS", "300"))
        except Exception:
            register_timeout = 300
        
        # 单进程调试模式：设置分布式训练环境变量（模拟单卡环境）
        if local_mode:
            os.environ.update({
                "RANK": "0",           # 当前进程在分布式训练中的排名（0 = 主进程）
                "WORLD_SIZE": "1",     # 总进程数（单进程 = 1）
                "MASTER_ADDR": "127.0.0.1",  # 分布式训练主节点地址
                "MASTER_PORT": "29500"       # 分布式训练主节点端口
            })
        
        # Ray 启动时声明的 CPU/GPU 资源数量
        # 这些值限制了 Ray 会创建多少 Worker，防止资源竞争
        try:
            advertised_cpus = int(os.getenv("RAY_NUM_CPUS", "16"))
        except Exception:
            advertised_cpus = 16
        
        try:
            advertised_gpus = int(os.getenv(
                "RAY_NUM_GPUS",
                str(getattr(ppo_config.trainer, "n_gpus_per_node", 1))
            ))
        except Exception:
            advertised_gpus = getattr(ppo_config.trainer, "n_gpus_per_node", 1)
        
        # 初始化 Ray（这里会启动 Ray 集群或连接到已有集群）
        ray.init(
            address=address,                     # Ray 集群地址（"local" 或远程地址）
            runtime_env=runtime_env,             # 所有 Worker 共享的环境变量
            local_mode=local_mode,               # 是否单进程调试模式
            include_dashboard=False,             # 不启动 Ray Dashboard（减少资源占用）
            _temp_dir=temp_dir,                  # 临时文件目录
            object_store_memory=object_store_mem, # 共享对象存储大小
            object_spilling_directory=spill_dir, # 对象溢出目录
            _system_config={
                "worker_register_timeout_seconds": register_timeout,  # Worker 注册超时
            },
            num_cpus=advertised_cpus,   # 向 Ray 声明的 CPU 数量
            num_gpus=advertised_gpus,   # 向 Ray 声明的 GPU 数量
        )
    
    # ── 启动训练 ──
    
    # 创建 Runner Actor（在 Ray 集群里的 Worker 节点上运行）
    runner = Runner.remote()
    
    # 调用 runner.run 并等待完成
    # ray.get() 阻塞当前进程，直到训练结束
    ray.get(runner.run.remote(ppo_config))


# Python 标准入口：当文件被直接运行时执行 main()
# 通过 `python -m verl.trainer.main` 调用时也会触发这里
if __name__ == "__main__":
    main()
