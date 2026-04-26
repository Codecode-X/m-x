# Monet 项目代码结构详解

> 本文档面向初学者，帮助你在向老师讲解代码时快速定位每一个文件的功能。

---

## 一、项目是什么？

**Monet**（**M**ultimodal **O**n-latent **Net**work）是一个发表在 **CVPR 2026** 的视觉推理框架。

它的核心创新是：让多模态大语言模型（如 Qwen2.5-VL-7B）不只用文字推理，还能在**隐空间**（latent space）里进行**视觉思考**——模型在回答问题时，可以多次生成一组"看不见的视觉思维向量"（latent embeddings），这些向量代替了传统的 `<observation>` 图文描述，让推理更紧凑、更高效。

**推理链示意：**
```
[用户问题 + 图像]
  → 文字推理...
  → <abs_vis_token> [latent思考向量×10个] </abs_vis_token>  ← 视觉思维步骤1
  → 继续文字推理...
  → <abs_vis_token> [latent思考向量×10个] </abs_vis_token>  ← 视觉思维步骤2
  → \boxed{最终答案}
```

**训练分两大阶段：**
- **SFT（监督微调）**：3个子阶段，逐步教会模型生成 latent 向量
- **RL（强化学习）**：用 GRPO 算法进一步提升模型推理能力

---

## 二、整体目录结构

```
Monet-main/
│
├── 📄 README.md                    ← 官方英文说明文档
├── 📄 CODE_GUIDE.md                ← 本文件（中文代码结构详解）
├── 📄 requirements.txt             ← SFT 环境依赖包列表
├── 📄 .gitignore                   ← Git 忽略文件配置
│
├── 📁 monet_qwen_model/            ← 【核心】SFT 训练用的修改版模型
├── 📁 src/                         ← 【核心】SFT 训练的主逻辑代码
├── 📁 script_examples/             ← SFT 三阶段的训练启动脚本
├── 📁 deepspeed/                   ← DeepSpeed 分布式训练配置
│
├── 📁 RL/                          ← 【核心】强化学习训练的全部代码
│   ├── 📁 examples/                ← RL 训练配置文件和启动脚本
│   ├── 📁 monet_models/            ← RL 用的修改版 Transformers/vLLM 模型
│   ├── 📁 tools/                   ← RL 辅助工具（API调用、奖励评判）
│   └── 📁 verl/                    ← RL 训练框架（基于 EasyR1/verl）
│
├── 📁 inference/                   ← 推理（测试模型效果）相关代码
├── 📁 images/                      ← 文档用图片（overview图、示例图）
└── 📁 transformers/                ← 空目录占位（可忽略）
```

---

## 三、SFT 训练相关文件详解

### 3.1 `monet_qwen_model/` — 修改版 Qwen2.5-VL 模型（SFT 用）

这个目录是 SFT 训练的**模型定义核心**，对官方 Qwen2.5-VL-7B 进行了定制修改，加入了 latent 推理逻辑。

| 文件 | 功能说明 |
|------|---------|
| `apply_qwen2_5_monet.py` | **补丁入口**。只有13行，运行时把 `modeling_qwen2_5_vl_monet.py` 注入到 Python 的 `transformers` 模块中，替换官方代码。相当于一个"偷梁换柱"的开关。|
| `modeling_qwen2_5_vl_monet.py` | **最核心的文件（2584行）**。在原版 Qwen2.5-VL 基础上，主要修改了 `Qwen2_5_VLModel.forward()` 和 `Qwen2_5_VLForConditionalGeneration.forward()`，加入了 `latent_mode` 分支：当 `latent_mode=True` 时，模型会逐步处理序列，对每个 latent token 位置单独做 forward，并把上一步的 `last_hidden_state` 作为下一步 latent 的输入。还包含对齐损失（alignment loss）、仿射子空间对齐（affine subspace alignment）等损失计算函数。|

**关键概念速查（在 `modeling_qwen2_5_vl_monet.py` 中）：**
- `latent_mode`（第1689行）：是否启用 latent 推理模式的开关
- `latent_token_id`：特殊 token `<abs_vis_token>` 的 ID
- `for pos in latent_pos`（第1799行）：多步 latent 推理的循环，**证明是多步而非单步**
- `alignment_loss()`（第244行）：计算 teacher-student 隐状态对齐的余弦相似度损失
- `affine_subspace_alignment_loss()`（第149行）：SFT Stage 3 用的仿射子空间对齐损失

---

### 3.2 `src/` — SFT 训练主逻辑

| 文件 | 功能说明 |
|------|---------|
| `main.py` | **SFT 训练入口（424行）**。加载模型和数据，根据 `--stage` 参数决定训练阶段（stage1/stage2/stage3），调用 trainer 执行训练循环。包含 DDP 分布式训练初始化、tokenizer 特殊 token 注册等。|
| `task.py` | **数据预处理（97行）**。定义 `Monet_single_input_images_preprocess_function()` 函数，负责解析 SFT 训练数据的对话格式，验证 `<abs_vis_token>` 和 `<observation>` 标签的对应关系，过滤不合格样本。|
| `trainer.py` | **训练步骤逻辑（351行）**。定义 `compute_latents_only_loss()`（对 latent 向量计算梯度代理损失）和 `load_offline_tensor()`（从磁盘加载预计算的 teacher 表示）。是 SFT 损失计算的关键文件。|
| `utils.py` | **工具函数集合（约1200行）**。包含：命令行参数解析 `get_args()`、随机种子设置 `seed_everything()`、数据集加载、4D attention mask 生成、数据 collate 函数等。是整个 SFT 训练的"工具箱"。|
| `precompute_teacher_reps.py` | **SFT Stage 2 预计算脚本**。在 Stage 2 训练之前运行，用 Stage 1 训练好的模型，提前计算训练集中每个样本的 `<observation>` token 的隐状态，保存到磁盘。这样 Stage 2 训练时直接读取，避免重复计算。|
| `precompute_teacher_latents.py` | **SFT Stage 3 预计算脚本**。在 Stage 3 训练之前运行，用 Stage 2 训练好的模型，提前计算训练集中每个 latent token 位置的目标隐状态，保存到磁盘供 Stage 3 的蒸馏对齐使用。|

---

### 3.3 `script_examples/` — SFT 训练启动脚本

三个 shell 脚本，分别对应 SFT 的三个训练阶段，直接用 `torchrun` 启动分布式训练。

| 文件 | 对应阶段 | 功能说明 |
|------|---------|---------|
| `sft_stage1.sh` | **Stage 1：文本 CoT 热身**。从 Qwen2.5-VL-7B-Instruct 出发，用含 `<observation>` 文字标注的数据（Monet-SFT-125K）做标准 SFT，让模型学会"先观察后推理"的思维格式。使用 **8 卡**，4 epochs。|
| `sft_stage2.sh` | **Stage 2：引入 Latent Token**。分两步：先运行 `precompute_teacher_reps.py` 预计算 teacher 表示；再训练，让模型用 `<abs_vis_token>` latent 向量替代文字 observation，通过对齐损失向 Stage 1 模型的隐状态靠近。**8 卡**，2 epochs。|
| `sft_stage3.sh` | **Stage 3：Latent 蒸馏精炼**。分两步：先运行 `precompute_teacher_latents.py` 预计算目标 latent；再用仿射子空间对齐损失进一步优化 latent 向量的质量。**8 卡**，2 epochs。|

---

### 3.4 `deepspeed/` — 分布式训练优化配置

| 文件 | 功能说明 |
|------|---------|
| `ds_zero2_gpu.json` | DeepSpeed ZeRO Stage 2 配置。启用 BF16 精度，将优化器状态分片到各 GPU（ZeRO-2），不做 CPU offload（全在 GPU 上），以加速 SFT 训练。SFT 三个阶段都用这个配置。|

---

## 四、RL 训练相关文件详解

RL 训练代码位于 `RL/` 目录，基于开源框架 [EasyR1](https://github.com/hiyouga/EasyR1)（verl）构建，用 Ray 做分布式调度。

### 4.1 `RL/examples/` — RL 训练的配置与脚本

| 文件/目录 | 功能说明 |
|----------|---------|
| `vlpo_train.sh` | **RL 训练启动脚本**。设置 8 卡 GPU 环境，配置 Ray 集群参数，调用 `python -m verl.trainer.main` 启动 GRPO 训练。关键参数：`sampling_strategy=monet`（启用 latent 推理）、`LATENT_SIZE=10`（每次生成10个 latent token）。|
| `config_monet.yaml` | **RL 训练主配置文件**。定义了所有超参数：数据路径、GRPO 算法参数（kl_coef、adv_estimator）、actor/rollout/ref 模型的显存配置、GPU 数量（`n_gpus_per_node: 4` 为调试值，实际用8卡）等。|
| `merge_model.sh` | **模型合并脚本**。RL 训练完成后，用于把 FSDP 分片的模型参数合并为一个完整的模型文件。|
| `runtime_env.yaml` | Ray 集群运行时环境配置（Python 路径等）。|
| `format_prompt/monet_format.jinja` | Jinja2 模板，定义 RL 训练时输入数据的 prompt 格式（如何组织问题和图像）。|
| `reward_function/monet_reward_function.py` | **奖励函数（核心，250行）**。定义 RL 的奖励信号：`format_reward()`（答案是否有`\boxed{}`格式）、`accuracy_reward()`（答案是否正确）、`use_latent_reward()`（是否使用了 latent 推理）、`rule_then_api_batch_judge()`（先规则判断，再调 Gemini API 判断）。|
| `reward_function/eval_grader.py` | 封装答案评估逻辑（调用 mathruler 库）。|
| `reward_function/answer_transformation.py` | 答案格式转换工具（如分数、百分比等格式统一化）。|
| `reward_function/r1v.py` | R1V 基线的奖励函数，用于对比实验。|
| `dataset_valid_ids/` | 各数据集的有效样本 ID 列表（`valid_ids.txt`），用于过滤数据。包含 Thyme-RL 训练集/验证集和 Geometry3K 数据集的有效 ID。|

---

### 4.2 `RL/monet_models/` — RL 用的修改版模型

RL 训练需要同时运行 **Transformers 版**（用于 actor 梯度更新）和 **vLLM 版**（用于 rollout 推理），因此有两套修改。

#### `RL/monet_models/transformers/`

| 文件 | 功能说明 |
|------|---------|
| `monet_modeling_qwen2_5_vl.py` | RL 训练专用的修改版 Qwen2.5-VL（约100KB）。与 SFT 版的 `modeling_qwen2_5_vl_monet.py` 功能类似，但针对 RL 场景做了适配：支持 VLPO 损失（`monet_rl_sigma` 参数）、与 verl 框架的 FSDP 接口对接。|

#### `RL/monet_models/vllm/`

| 文件 | 功能说明 |
|------|---------|
| `monet_gpu_model_runner.py` | **RL rollout 用的修改版 vLLM 推理引擎（94KB）**。替换 vLLM 官方的 `gpu_model_runner.py`，在每个 decoding step 检测是否生成了 `<abs_vis_token>` token（ID=151666），若是则切换到 latent 推理模式，把上一层的 hidden state 作为下一个 token 的输入，实现多步 latent 推理的自回归生成。|
| `latent_hook.py` | **Latent 向量传输模块（555行）**。在 vLLM 每个 decoding step 生成 latent 向量后，通过 UDP/TCP socket 把向量发送出去，供外部 `LatentRecorder` 收集，用于后续的 VLPO 损失计算。|
| `latent_recorder.py` | **Latent 向量收集器（289行）**。接收 `latent_hook.py` 发来的 latent 向量，按 `request_id` 归档，支持多种输出格式（list/numpy/tensor）。RL 训练时用它积累每个生成样本的完整 latent 轨迹。|
| `__init__.py` | 模块初始化文件，暴露 vLLM 模型的接口。|

---

### 4.3 `RL/tools/` — RL 辅助工具

| 文件 | 功能说明 |
|------|---------|
| `api_judge.py` | **API 评判工具（9.5KB）**。封装调用 Gemini/DeepSeek API 进行答案正确性判断的逻辑。RL 训练时对于规则难以判断的题目，调用大模型 API 来打分。|
| `custom_api.py` | **自定义 API 接口（3.5KB）**。统一封装 Gemini 和 DeepSeek 两种 API 的调用方式，提供统一的接口供 `api_judge.py` 调用。|
| `actors.py` | Ray actor 工具，用于分布式 API 调用的并发管理。|
| `hash_dict.py` | 哈希字典工具，用于缓存 API 调用结果，避免重复调用相同问题浪费费用。|

---

### 4.4 `RL/verl/` — RL 训练框架（基于 EasyR1）

这个目录是 RL 训练框架的核心，包含了 GRPO 算法的完整实现。

#### `RL/verl/trainer/` — 训练调度层

| 文件 | 功能说明 |
|------|---------|
| `main.py` | **RL 训练入口（223行）**。解析 `config_monet.yaml` 配置，初始化 Ray 集群，创建 `RayPPOTrainer` 并启动训练。|
| `ray_trainer.py` | **RL 训练主循环（约50KB）**。实现 `RayPPOTrainer`，协调 actor、rollout、ref、reward 四个 worker 之间的交互，执行标准的 PPO/GRPO 训练循环：rollout → 计算奖励 → 计算优势 → 更新 actor。|
| `core_algos.py` | **RL 核心算法（19KB）**。实现 GRPO 的优势估计（advantage estimation）、KL 散度惩罚（`low_var_kl`）、PPO-clip 损失等数学计算。|
| `config.py` | 定义训练配置的数据类 `PPOConfig`，对应 `config_monet.yaml` 的结构。|
| `data_loader.py` | RL 训练数据加载器，读取 Thyme-RL 数据集。|
| `metrics.py` | 训练指标计算（奖励均值/方差、KL散度等），用于 wandb 日志记录。|
| `save_any_log.py` | 日志保存工具，把训练日志同时输出到控制台和文件。|

#### `RL/verl/workers/` — 分布式 Worker

| 文件/目录 | 功能说明 |
|----------|---------|
| `fsdp_workers.py` | **FSDP Worker 核心（35KB）**。实现 `FSDPWorker`，用 PyTorch FSDP 对大模型进行分片，支持 actor 和 ref 模型的 forward/backward 操作。|
| `actor/dp_actor.py` | Actor worker 实现（21KB）。负责 actor 模型的前向传播、损失计算和梯度更新。`dp_actor.py` 是数据并行版本。|
| `rollout/vllm_rollout_spmd.py` | **Rollout Worker（17KB）**。用 vLLM 引擎生成样本（rollout），调用修改后的 `monet_gpu_model_runner.py` 实现 latent 推理。|
| `reward/function.py` | 奖励 Worker（10KB），调用 `monet_reward_function.py` 为每个生成的样本计算奖励分数。|
| `critic/` | Critic worker（PPO 中用于估计状态价值，GRPO 模式下不需要）。|
| `sharding_manager/` | 管理 FSDP 和 vLLM 之间的权重同步（actor 参数更新后需要同步到 rollout 引擎）。|

#### `RL/verl/utils/` — 工具函数

| 文件 | 功能说明 |
|------|---------|
| `dataset.py` | 数据集工具，处理多模态数据的序列化和批处理。|
| `fsdp_utils.py` | FSDP 相关工具（模型分片、收集等）。|
| `seqlen_balancing.py` | 序列长度均衡工具，让各 GPU 负载尽量平衡。|
| `torch_functional.py` | PyTorch 张量操作工具函数集合。|
| `ulysses.py` | Ulysses 序列并行工具（长序列并行训练）。|
| `checkpoint/` | 模型和优化器的 checkpoint 保存/加载管理。|
| `logger/` | 训练日志工具（支持 wandb、console 等多种输出）。|

#### `RL/verl/single_controller/` — Ray 分布式控制器

| 文件 | 功能说明 |
|------|---------|
| `ray/base.py` | **Ray Worker Group（19KB）**。实现基于 Ray 的分布式 worker 调度，支持 actor、rollout、ref 等模型在多节点多 GPU 上的协同运行。|
| `base/worker_group.py` | Worker 组抽象基类。|
| `base/worker.py` | 单个 Worker 抽象基类。|

#### `RL/` 根目录其他文件

| 文件 | 功能说明 |
|------|---------|
| `monet_rl_patch.py` | **RL 补丁入口（2.5KB）**。类似 SFT 的 `apply_qwen2_5_monet.py`，在 RL 训练启动时把修改版的 Transformers 模型和 vLLM 推理引擎注入到对应的 Python 模块中。|
| `requirements.txt` | RL 环境的依赖包列表（与 SFT 环境不同，需要单独安装）。|
| `setup.py` | RL 模块的 Python 包安装配置。|
| `Dockerfile` | Docker 容器构建文件（最新版）。|
| `Dockerfile.legacy` | Docker 容器构建文件（旧版，备用）。|

---

## 五、推理相关文件详解

### `inference/` — 推理（使用模型生成答案）

| 文件 | 功能说明 |
|------|---------|
| `vllm_inference_example.py` | **推理示例脚本（41行）**。演示如何用 vLLM 加载 Monet-7B 模型，输入一张图片和问题，生成带 latent 推理的答案，并把 latent token 替换为 `<latent>` 占位符显示。**这是给用户看效果最直接的入口。**|
| `apply_vllm_monet.py` | 推理用的 vLLM 补丁注入文件（类似 `apply_qwen2_5_monet.py`，但用于推理时的 vLLM 引擎）。|
| `load_and_gen_vllm.py` | 封装 vLLM 模型的加载（`vllm_mllm_init()`）、输入处理（`vllm_mllm_process_batch_from_messages()`）和生成（`vllm_generate()`）函数。|
| `example.sh` | 推理运行示例命令（几行 bash 命令）。|
| `vllm/monet_gpu_model_runner.py` | 推理专用的修改版 vLLM 引擎（132KB）。与 RL 版本类似但针对纯推理场景优化，不需要记录 latent 轨迹。|

---

## 六、三阶段 SFT 训练流程图

```
原始模型: Qwen2.5-VL-7B-Instruct
        │
        ▼
┌─────────────────────────────────────────────┐
│ SFT Stage 1: 文字 CoT 热身                   │
│ 脚本: script_examples/sft_stage1.sh         │
│ 数据: Monet-SFT-125K (含 <observation> 标注) │
│ 损失: 标准交叉熵 CE loss                     │
│ GPU:  8卡, 4 epochs                         │
└─────────────────┬───────────────────────────┘
                  │ 产出: Stage1 模型
                  ▼
        ┌──────────────────────┐
        │ 预计算 Teacher 表示  │
        │ src/precompute_      │
        │ teacher_reps.py      │
        └──────────┬───────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ SFT Stage 2: 引入 Latent Token               │
│ 脚本: script_examples/sft_stage2.sh         │
│ 数据: Monet-SFT-125K                        │
│ 损失: CE loss + 对齐损失 (对齐观察token)     │
│ GPU:  8卡, 2 epochs                         │
└─────────────────┬───────────────────────────┘
                  │ 产出: Stage2 模型
                  ▼
        ┌──────────────────────┐
        │ 预计算 Latent 目标   │
        │ src/precompute_      │
        │ teacher_latents.py   │
        └──────────┬───────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ SFT Stage 3: Latent 蒸馏精炼                 │
│ 脚本: script_examples/sft_stage3.sh         │
│ 数据: Monet-SFT-125K                        │
│ 损失: CE loss + 仿射子空间对齐损失           │
│ GPU:  8卡, 2 epochs                         │
└─────────────────┬───────────────────────────┘
                  │ 产出: Monet-SFT-7B
                  ▼
┌─────────────────────────────────────────────┐
│ RL 训练: GRPO + VLPO                        │
│ 脚本: RL/examples/vlpo_train.sh             │
│ 数据: Thyme-RL                              │
│ 奖励: 格式奖励 + 准确性奖励 + Latent 使用奖励│
│ GPU:  8卡                                   │
└─────────────────┬───────────────────────────┘
                  │ 产出: Monet-7B（最终模型）
                  ▼
              推理使用
        inference/vllm_inference_example.py
```

---

## 七、关键概念速查表

| 概念 | 解释 | 所在文件 |
|------|------|---------|
| `<abs_vis_token>` / `</abs_vis_token>` | Latent 推理的开始/结束标记 token | `src/main.py` 第64-66行 |
| `latent_mode` | 是否启用 latent 推理模式的布尔开关 | `monet_qwen_model/modeling_qwen2_5_vl_monet.py` 第1689行 |
| `LATENT_SIZE` | 每次触发 latent 推理时生成的向量数量（默认10） | `RL/examples/vlpo_train.sh` 第41行 |
| `stage` | SFT 训练阶段（sft_stage1/sft_stage2/avt_v5_stage2） | `script_examples/*.sh` |
| `alignment_loss` | Teacher-student 余弦相似度对齐损失 | `monet_qwen_model/modeling_qwen2_5_vl_monet.py` 第244行 |
| `affine_subspace_alignment_loss` | Stage 3 的仿射子空间对齐损失 | `monet_qwen_model/modeling_qwen2_5_vl_monet.py` 第149行 |
| `GRPO` | Group Relative Policy Optimization，RL 算法 | `RL/verl/trainer/core_algos.py` |
| `VLPO` | Visual Latent Policy Optimization，Monet 提出的 RL 变体 | `RL/examples/vlpo_train.sh` 中 `sampling_strategy=monet` |
| `FSDP` | Fully Sharded Data Parallel，大模型分布式训练方式 | `RL/verl/workers/fsdp_workers.py` |
| `DeepSpeed ZeRO-2` | SFT 训练的优化器状态分片 | `deepspeed/ds_zero2_gpu.json` |
| `emit_latents_step` | 每步 decoding 时发送 latent 向量的钩子函数 | `RL/monet_models/vllm/latent_hook.py` 第104行 |
| `LatentRecorder` | 收集并归档每个请求的多步 latent 向量 | `RL/monet_models/vllm/latent_recorder.py` |

---

## 八、如何快速找到代码

**如果你想看模型是如何生成 latent 向量的：**
→ `monet_qwen_model/modeling_qwen2_5_vl_monet.py`，第1689行开始的 `latent_mode` 分支

**如果你想看 SFT 训练怎么启动：**
→ 先看 `script_examples/sft_stage1.sh`（最简单），再看 `src/main.py`

**如果你想看损失函数怎么计算：**
→ `src/trainer.py`（辅助损失） + `monet_qwen_model/modeling_qwen2_5_vl_monet.py`（主损失）

**如果你想看 RL 训练的奖励怎么设计：**
→ `RL/examples/reward_function/monet_reward_function.py`

**如果你想看 RL 训练怎么启动：**
→ 先看 `RL/examples/vlpo_train.sh`，再看 `RL/examples/config_monet.yaml`，再看 `RL/verl/trainer/main.py`

**如果你想看推理（inference）怎么跑：**
→ `inference/vllm_inference_example.py`（只有41行，最简单）

**如果你想看 latent 推理在 vLLM 中如何实现：**
→ `RL/monet_models/vllm/monet_gpu_model_runner.py`（搜索 `latent_token_id` 或 `LATENT_START_ID`）
