"""
本文件的作用：定义三个 Ray Actor 类，用于 RL 训练时的分布式状态管理。

在 GRPO/VLPO 强化学习训练中，Ray 用于协调多 GPU 并行工作。
这三个 Actor 类以"服务"的方式运行在独立进程中，为所有 GPU Worker 提供共享状态访问。

三个 Actor 类：
1. StepHashServer：步骤相似性缓存服务器
   - 维护每个题目（sample_id）的"推理步骤语义聚类"
   - 通过比较向量相似度，判断新生成的步骤是否与历史步骤"相同"
   - 可以告诉训练器：这个推理步骤之前见过，且历史上它能/不能导向正确答案
   
2. EmbedServer：文本嵌入服务器
   - 运行在专用 GPU 上，负责把推理步骤文本转换为向量嵌入
   - 供 StepHashServer 使用（先嵌入，再做相似度聚类）

3. SampleHashServer：样本级状态缓存服务器
   - 维护每个题目（sample_id）是否曾经被正确回答过
   - 记录每个题目观测到的最短/均值正确回答长度
   - 用于 VLPO 的 advantage 计算（如果某道题模型从未答对，不做 advantage 规范化）

使用方式（Ray 远程 Actor）：
    server = StepHashServer.remote(config)     # 在 Ray 集群里启动服务
    result = server.update_sample_step_hash_dict.remote(...)  # 远程调用
    result = ray.get(result)                   # 等待并获取结果
"""

# 导入 PyTorch（用于 EmbedServer 的张量操作）
import torch
# 导入 HuggingFace transformers 的 tokenizer 和模型加载工具
# （此处未直接使用，遗留导入）
from transformers import AutoTokenizer, AutoModel
# 导入两个哈希字典实现（核心数据结构，定义在 hash_dict.py）
from tools.hash_dict import StepHashDict, SampleHashDict
# 导入 Ray 框架（用于分布式 Actor 管理）
import ray
# 导入 PyTorch 函数式接口（此处未直接使用，遗留导入）
import torch.nn.functional as F
# 导入 NumPy（此处未直接使用，遗留导入）
import numpy as np
# 导入类型注解工具
from typing import List, Union
# 导入进程监控、操作系统接口、垃圾回收工具
import psutil, os, gc
# 导入 sentence-transformers 库（此处未直接使用，遗留导入，实际用 vLLM 做嵌入）
from sentence_transformers import SentenceTransformer
# 导入 vLLM 推理引擎（用于 EmbedServer）
from vllm import LLM


# ══════════════════════════════════════════════════════════════════════
# StepHashServer：推理步骤相似性缓存服务
# ══════════════════════════════════════════════════════════════════════

# @ray.remote 装饰器：把这个类变成一个 Ray 远程 Actor
# num_cpus=4：这个 Actor 占用 4 个 CPU 核（不用 GPU，纯 CPU 操作）
# num_gpus=0：不需要 GPU
@ray.remote(num_cpus=4, num_gpus=0)
class StepHashServer:
    """
    推理步骤相似性缓存服务器（Ray Actor）。
    
    核心能力：
    1. 记录每个题目（sample_id）被解题时，每个推理步骤的语义向量和正确性
    2. 当新生成了一个推理步骤时，快速判断这个步骤是否与历史中的某个步骤"语义相同"
    3. 如果语义相同，还能告诉你这类步骤历史上是否能导向正确答案
    
    用途：
    - VLPO 训练中，避免多次采样时对完全相同的步骤重复计算奖励
    - 通过步骤正确性的历史统计，提供更稳定的 per-step 优势估计（advantage）
    """
    
    def __init__(self, config):
        """
        初始化 StepHashServer。
        
        参数：
        - config：verl 训练配置对象，从中读取步骤哈希相关参数：
            - config.rollout.mc.step_hash_threshold：相似度阈值（高于此值认为是"同一步骤"）
            - config.rollout.mc.correct_cluster_threshold：正确性阈值（簇中超过此比例的成员能导向正确答案，才认为这个簇是"正确"的）
        """
        # 创建 StepHashDict 实例
        # rep_mode="all"：判断是否加入某个簇时，要求与簇内所有成员的相似度都超过阈值（最严格）
        self.step_hash_dict = StepHashDict(
            similarity_threshold=config.rollout.mc.step_hash_threshold,
            correct_cluster_threshold=config.rollout.mc.correct_cluster_threshold,
            rep_mode="all"
        )
    
    def update_sample_step_hash_dict(self, sample_id, steps, embeds, lead_to_correct_list):
        """
        更新某个题目的步骤哈希字典。
        
        在每次 rollout 后调用：把新生成的推理步骤（及其向量、正确性）加入字典。
        
        参数：
        - sample_id：题目 ID
        - steps：推理步骤的文本列表（如 ["First, I observe...", "Then, I count..."]）
        - embeds：对应每个步骤的向量嵌入（numpy 数组，形状为 (N, D)）
        - lead_to_correct_list：每个步骤是否最终导向了正确答案（bool 列表）
        
        返回：
        - correctness：每个步骤的正确性估计（基于其所在簇的历史统计）
        """
        return self.step_hash_dict.update_sample_step_hash_dict(
            sample_id=sample_id,
            embeddings=embeds,
            texts=steps,
            lead_correct_list=lead_to_correct_list
        )
    
    def look_up_step_correctness(
        self,
        sample_id: int,
        texts: Union[str, List[str]]  # 待查询的步骤文本
    ) -> List[bool]:
        """
        查询某个推理步骤文本的历史正确性。
        
        参数：
        - sample_id：题目 ID
        - texts：待查询的步骤文本（单个字符串或列表）
        
        返回：
        - 每个步骤文本对应的正确性（True/False 列表）
        
        异常：
        - ValueError：如果文本在历史记录中找不到
        """
        return self.step_hash_dict.look_up_step_correctness(
            sample_id=sample_id,
            texts=texts
        )
    
    def update_min_mean_correct_resp_len(self, sample_id: int, resp_len: int):
        """
        更新某个题目的"最短/均值正确回答长度"统计。
        
        在某次 rollout 正确回答了这道题后调用，记录该次回答的 token 长度。
        用于后续的长度惩罚计算（避免模型在能正确回答时仍生成冗余内容）。
        
        参数：
        - sample_id：题目 ID
        - resp_len：本次正确回答的 token 长度
        """
        return self.step_hash_dict.update_min_mean_correct_resp_len(
            sample_id=sample_id,
            resp_len=resp_len
        )
    
    def look_up_min_mean_correct_resp_len(self, sample_id: int):
        """
        查询某个题目历史上观测到的最短/均值正确回答长度。
        
        参数：
        - sample_id：题目 ID
        
        返回：
        - (min_len, mean_len)：最短长度和均值长度（如果从未正确回答过，返回 inf 和 0.0）
        """
        return self.step_hash_dict.look_up_min_mean_correct_resp_len(sample_id=sample_id)
    
    def get_step_dict_info(self, verbose_info: bool = False, print_info: bool = False):
        """
        获取整个步骤哈希字典的统计信息（用于监控和调试）。
        
        参数：
        - verbose_info：是否包含每个簇的详细信息（成员文本、正确率等）
        - print_info：是否同时打印到控制台
        
        返回：
        - info_dict：包含每个 sample_id 的统计信息的字典
        """
        return self.step_hash_dict.get_step_dict_info(verbose_info, print_info)
    
    def get_rss(self):
        """
        获取当前进程的内存使用量（GB）。
        
        用于监控 StepHashServer 的内存占用，防止随训练进行内存无限增长。
        调用前先触发 GC，获取更准确的数字。
        
        返回：
        - rss_gb：物理内存使用量（GB）
        """
        gc.collect()
        # psutil.Process(os.getpid()) 获取当前进程对象
        # .memory_info().rss 是 Resident Set Size（实际物理内存占用，字节）
        # 除以 2^30 转换为 GB
        rss_gb = psutil.Process(os.getpid()).memory_info().rss / 2 ** 30
        return rss_gb
    
    def save_info(self, filepath: str, overwrite: bool = True):
        """
        把步骤哈希字典保存到磁盘（用于断点续训）。
        
        参数：
        - filepath：保存目录路径
        - overwrite：是否覆盖已有文件
        """
        return self.step_hash_dict.save_info(filepath, overwrite)
    
    def load_info(self, filepath: str):
        """
        从磁盘加载步骤哈希字典（断点续训时恢复状态）。
        
        参数：
        - filepath：之前保存的目录路径
        """
        return self.step_hash_dict.load_info(filepath)
    
    def ping(self):
        """
        心跳检测：用于确认这个 Actor 还活着，可以正常通信。
        调用方可以用 ray.get(server.ping.remote()) 来检测 Actor 是否正常。
        """
        return


# ══════════════════════════════════════════════════════════════════════
# EmbedServer：文本嵌入服务（GPU Actor）
# ══════════════════════════════════════════════════════════════════════

# num_gpus=1：这个 Actor 需要 1 块 GPU
# resources={"embed_gpu": 1}：自定义资源标签（需要在 Ray 集群启动时指定）
# 这样可以把 EmbedServer 调度到专用的"embed GPU"上，不占用训练用的 GPU
@ray.remote(num_gpus=1, resources={"embed_gpu": 1})
class EmbedServer:
    """
    文本嵌入服务器（GPU Actor）。
    
    职责：把推理步骤的文本转换为向量嵌入，供 StepHashServer 做相似度聚类。
    
    为什么需要独立服务器：
    - 嵌入计算需要 GPU，但不需要和训练模型共享 GPU
    - 作为独立 Actor，可以在不干扰训练 GPU 的情况下异步做嵌入计算
    - StepHashServer 可以把嵌入任务委托给 EmbedServer，自己不需要 GPU
    """
    
    def __init__(self, model_path: str):
        """
        初始化嵌入模型。
        
        参数：
        - model_path：嵌入模型的路径（支持任何 vLLM 可以加载的嵌入模型）
        
        注意：在 Ray Actor 里，GPU 编号从 0 开始（是 Ray 分配给这个 Actor 的那块 GPU）
        """
        # 把 CUDA 设备设为 0（在 Ray Actor 里 0 就是 Ray 分配的那块 GPU）
        torch.cuda.set_device(0)
        
        # 用 vLLM 加载嵌入模型（task="embed" 表示只用于嵌入，不做生成）
        # 也可以用 SentenceTransformer(...)，但 vLLM 支持更多模型和更快的批量推理
        self.model = LLM(model=model_path, task="embed")
    
    @torch.no_grad()
    def encode(self, sentences, use_tqdm):
        """
        批量编码文本为向量嵌入。
        
        参数：
        - sentences：待编码的文本列表
        - use_tqdm：是否显示进度条
        
        返回：
        - 嵌入向量列表（vLLM embed() 的输出格式）
        """
        # @torch.no_grad()：不计算梯度（推理模式，节省显存）
        return self.model.embed(sentences, use_tqdm=use_tqdm)
    
    def ping(self):
        """心跳检测"""
        return "ok"


# ══════════════════════════════════════════════════════════════════════
# SampleHashServer：样本级状态缓存服务
# ══════════════════════════════════════════════════════════════════════

# num_cpus=2：只需要 2 个 CPU（纯字典操作，轻量级）
# num_gpus=0：不需要 GPU
@ray.remote(num_cpus=2, num_gpus=0)
class SampleHashServer:
    """
    样本（题目）级状态缓存服务器（Ray Actor）。
    
    是对 SampleHashDict 的 Ray Actor 封装，提供以下功能：
    1. 记录某道题是否曾被正确回答（一旦为 True 永不清除）
    2. 记录每道题的最短/均值正确回答长度
    3. 支持保存/加载（断点续训）
    
    与 StepHashServer 的区别：
    - StepHashServer 记录的是推理步骤级别的信息（步骤向量、正确性）
    - SampleHashServer 记录的是样本级别的信息（题目整体的正确性历史、长度统计）
    """
    
    def __init__(self):
        """初始化样本字典"""
        # 创建 SampleHashDict 实例（纯 CPU 操作）
        self.sample_dict = SampleHashDict()
    
    def set_correct_answered(self, sample_id: int, value: bool):
        """
        标记某道题是否被正确回答。
        
        这是"单调递增"的操作：一旦某道题被标记为正确（True），
        后续再调用 set_correct_answered(sample_id, False) 也不会改回 False。
        
        参数：
        - sample_id：题目 ID
        - value：是否正确回答了
        """
        return self.sample_dict.set_correct_answered(sample_id, value)
    
    def get_info(self, sample_id: int):
        """
        获取某道题的完整状态信息。
        
        参数：
        - sample_id：题目 ID
        
        返回：
        - info_dict：包含 'corret_answered'（bool）和 'min_len'（float）的字典
        """
        return self.sample_dict.get_info(sample_id)
    
    def update_min_mean_correct_resp_len(self, sample_id: int, resp_len: int):
        """
        更新某道题的正确回答长度统计。
        
        参数：
        - sample_id：题目 ID
        - resp_len：本次正确回答的 token 长度
        """
        return self.sample_dict.update_min_mean_correct_resp_len(sample_id, resp_len)
    
    def look_up_min_mean_correct_resp_len(self, sample_id: int):
        """
        查询某道题的最短/均值正确回答长度。
        
        参数：
        - sample_id：题目 ID
        
        返回：
        - (min_len, mean_len)：最短和均值长度
        """
        return self.sample_dict.look_up_min_mean_correct_resp_len(sample_id)
    
    def ping(self):
        """心跳检测"""
        return
    
    def save_info(self, filepath: str, overwrite: bool = True):
        """
        把样本状态字典保存到磁盘（断点续训）。
        
        参数：
        - filepath：保存目录
        - overwrite：是否覆盖已有文件
        """
        return self.sample_dict.save_info(filepath, overwrite)
    
    def load_info(self, filepath: str):
        """
        从磁盘加载样本状态字典（恢复训练状态）。
        
        参数：
        - filepath：之前保存的目录路径
        """
        return self.sample_dict.load_info(filepath)
