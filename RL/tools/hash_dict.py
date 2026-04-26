"""
本文件的作用：定义两个用于 RL 训练的哈希字典数据结构。

这两个类负责"记忆"训练过程中的关键统计信息，供奖励计算和 advantage 估计使用：

1. StepHashDict：推理步骤语义聚类字典
   - 对每道题，把历史上见过的推理步骤按语义相似度聚成若干"簇"
   - 每个簇记录：代表向量、成员文本列表、能否导向正确答案的历史统计
   - 新生成的步骤来了，找最近的簇；超过相似度阈值就加入（更新正确率）；否则新建簇
   - 用于"Monte Carlo per-step advantage"估计（VLPO 的核心创新之一）
   
2. SampleHashDict：样本级状态字典
   - 对每道题，记录它是否曾被正确回答过、历史正确回答的最短/均值长度
   - 用于 advantage 规范化和长度惩罚的基准

设计思路（为什么需要"语义聚类"而不是精确匹配）：
- RL 训练中会对同一道题采样多次（n=8 或更多）
- 不同次采样的推理步骤文字措辞可能略有不同，但语义相同
- 语义聚类避免了把"语义相同但文字不同"的步骤当成全新步骤处理
- 使得 per-step 正确率估计更稳定（样本量更大）
"""

# 导入 NumPy，用于向量运算（相似度计算、均值等）
import numpy as np
# 导入 defaultdict：访问不存在的 key 时自动返回默认值，不抛异常
from collections import defaultdict
# 导入类型注解工具
from typing import List, Dict, Union
# 导入 pickle：用于把字典序列化保存到磁盘
import pickle
# 导入操作系统接口（文件路径操作）
import os


class StepHashDict:
    """
    推理步骤语义聚类字典。
    
    数据结构概览：
    self.dicts = {
        sample_id_1: {
            0: {  # 簇 0
                "rep_embedding": np.array(D,),       # 这个簇的"代表向量"（用于快速相似度比较）
                "rep_text": "First, observe...",      # 代表向量对应的文本
                "members_texts": ["...", "..."],       # 所有成员的文本
                "members_idx": [0, 3, 5],             # 所有成员在 rollout batch 里的索引
                "member_embeddings": np.array(M, D), # 所有成员的向量（M 是成员数）
                "correct_cnt": 2                      # 有多少成员最终导向了正确答案
            },
            1: { ... },  # 簇 1
            ...
        },
        sample_id_2: { ... },
        ...
    }
    
    self.resp_len_stats = {
        sample_id: {"min_len": ..., "mean_len": ..., "cnt": ...},
        ...
    }
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,       # 相似度阈值：高于此值才视为"同一步骤"
        correct_cluster_threshold: float = 0.5,  # 正确率阈值：超过此比例的成员能导向正确答案，该簇视为"正确"
        rep_mode: str = "all",                    # 代表向量的选取策略（见下方说明）
    ):
        """
        初始化 StepHashDict。
        
        参数：
        - similarity_threshold：余弦相似度阈值（0~1），高于此值认为两个步骤语义相同
        - correct_cluster_threshold：正确率阈值（0~1），簇内超过此比例的成员能导向正确答案时，
          认为这个簇整体是"正确"的（用于判断新步骤的历史正确性）
        - rep_mode：代表向量策略，支持：
            "first"    - 固定用簇的第一个成员作为代表向量（最快）
            "centroid" - 代表向量 = 所有成员向量的均值（最准确，但需要更新）
            "medoid"   - 代表向量 = 离均值最近的成员（比 centroid 更鲁棒）
            "all"      - 用第一个成员作为代表，但判断归属时要求与所有成员相似（最严格）
        """
        # 主字典：sample_id → {cluster_id → cluster_info}
        # defaultdict(dict) 表示：访问不存在的 sample_id 时自动创建空字典
        self.dicts: Dict[int, Dict[int, dict]] = defaultdict(dict)
        
        # 响应长度统计字典：sample_id → {"min_len": ..., "mean_len": ..., "cnt": ...}
        self.resp_len_stats: Dict[int, dict] = defaultdict(
            lambda: {"min_len": float("inf"), "mean_len": 0.0, "cnt": 0}
        )
        
        # 保存配置参数
        self.similarity_threshold = similarity_threshold
        self.correct_cluster_threshold = correct_cluster_threshold
        self.rep_mode = rep_mode.lower()  # 统一转小写
        
        # 检查 rep_mode 是否合法
        assert self.rep_mode in {"first", "centroid", "medoid", "all"}
    
    # ─────────────────────── 私有辅助方法 ───────────────────────
    
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        """
        L2 归一化一个向量（使其模长为 1）。
        
        归一化后，两个向量的点积等于它们的余弦相似度（值域 [-1, 1]）。
        加 1e-8 防止除以零（零向量的情况）。
        
        参数：v - 待归一化的向量
        返回：归一化后的向量
        """
        return v / (np.linalg.norm(v) + 1e-8)
    
    def _build_rep_matrix(self, clusters: Dict[int, dict]) -> np.ndarray:
        """
        把当前所有簇的"代表向量"拼成一个矩阵 (K, D)，便于批量计算相似度。
        
        K = 当前簇数量，D = 向量维度
        
        参数：clusters - 当前 sample 的簇字典
        返回：(K, D) 的 numpy 矩阵，如果没有任何簇则返回 None
        """
        if not clusters:
            # 还没有任何簇（第一次见这个 sample_id）
            return None
        
        # 从每个簇中取出代表向量，拼成矩阵
        reps = [info["rep_embedding"] for info in clusters.values()]
        reps = np.vstack(reps).copy()         # copy() 确保内存连续
        reps.setflags(write=True)             # 确保可写（numpy 有时返回只读数组）
        return reps
    
    # ─────────────────────── 主要接口 ───────────────────────
    
    def update_sample_step_hash_dict(
        self,
        sample_id: int,
        embeddings: np.ndarray,              # (N, D) 形状，每行是一个步骤的向量嵌入（已 L2 归一化）
        texts: List[str],                    # 每个步骤对应的文本
        lead_correct_list: List[bool] | None = None  # 每个步骤是否最终导向了正确答案
    ):
        """
        把新一批推理步骤（embeddings + texts + correctness）更新到哈希字典中。
        
        对每个新步骤：
        1. 先找最相似的现有簇（通过矩阵乘法计算余弦相似度）
        2. 如果找到满足阈值的簇，就把这个步骤加入该簇（更新成员列表和正确率统计）
        3. 如果没找到，就新建一个簇
        4. 返回每个步骤的"历史正确性估计"（基于其所在簇的历史正确率）
        
        参数：
        - sample_id：题目 ID
        - embeddings：每个步骤的向量（形状 N×D，已 L2 归一化）
        - texts：每个步骤的原始文本
        - lead_correct_list：每个步骤是否最终导向了正确答案（None 时不更新正确率）
        
        返回：
        - correctness：每个步骤的正确性估计（True/False 列表，基于历史统计）
        """
        # 参数校验：embeddings 和 texts 数量必须一致
        assert len(embeddings) == len(texts), "embeddings 和 texts 数量不一致"
        
        # 取出该 sample 的簇字典（不存在时自动创建空字典）
        clusters = self.dicts[sample_id]
        
        # 构建代表向量矩阵（用于批量相似度计算）
        rep_matrix = self._build_rep_matrix(clusters)
        
        # 每个步骤的正确性估计结果
        correctness = []
        
        # 遍历每个步骤，依次更新
        for idx, (emb, txt) in enumerate(zip(embeddings, texts)):
            # 取出这个步骤是否能导向正确答案（None 时用 None）
            lead_to_correct = lead_correct_list[idx] if lead_correct_list else None
            
            # ──────── 特殊情况：第一个步骤（还没有任何簇） ────────
            if rep_matrix is None:
                # 创建第一个簇（ID = 0）
                clusters[0] = dict(
                    rep_embedding=emb,                         # 代表向量 = 第一个成员
                    rep_text=txt,                              # 代表文本
                    members_texts=[txt],                       # 成员文本列表
                    members_idx=[idx],                         # 成员在 batch 里的索引
                    member_embeddings=[emb],                   # 成员向量列表（初始是 Python list）
                    correct_cnt=1 if lead_to_correct else 0   # 正确成员计数
                )
                
                # 用这个第一个向量初始化代表向量矩阵（形状 (1, D)）
                rep_matrix = emb[None, :].copy()
                rep_matrix.setflags(write=True)
                
                # 第一个步骤的正确性：基于 lead_to_correct（还没有历史统计，直接用当前值）
                correctness.append(True if lead_to_correct else False)
                continue  # 处理下一个步骤
            
            # ──────── 正常情况：找最合适的簇 ────────
            
            insert_cid = None  # 要插入的簇 ID（None 表示需要新建簇）
            
            if self.rep_mode == "all":
                # "all" 模式：判断是否加入簇时，要求与簇内所有成员的相似度都超过阈值
                
                # 第一步：用代表向量矩阵做粗筛（快速，过滤掉明显不相似的簇）
                sims_rep = rep_matrix @ emb  # (K,)：与每个簇代表向量的余弦相似度
                
                # 找出代表向量相似度超过阈值的候选簇
                cand_cids = np.where(sims_rep > self.similarity_threshold)[0]
                
                if cand_cids.size:  # 有候选簇才继续细筛
                    best_avg, insert_cid = -1.0, None
                    for cid in cand_cids:
                        cinfo = clusters[cid]
                        member_embs = cinfo["member_embeddings"]  # (M, D) 所有成员向量
                        
                        # 计算与簇内所有成员的相似度
                        sims = member_embs @ emb  # (M,)
                        
                        # "all" 要求：必须与所有成员都相似（最严格）
                        if np.all(sims > self.similarity_threshold):
                            # 用平均相似度作为"最合适"的度量
                            avg_sim = sims.mean()
                            if avg_sim > best_avg:
                                # 更新最佳候选簇
                                insert_cid, best_avg = cid, avg_sim
            
            else:
                # "first" / "centroid" / "medoid" 模式：只与代表向量比较
                sims = np.dot(rep_matrix, emb)       # (K,) 与所有代表向量的相似度
                best_row = int(np.argmax(sims))       # 找相似度最高的簇
                if float(sims[best_row]) > self.similarity_threshold:
                    insert_cid = best_row  # 相似度超过阈值，加入这个簇
            
            # ──────── 根据 insert_cid 决定：插入现有簇 or 新建簇 ────────
            
            if insert_cid is not None:
                # ── 插入现有簇 ──
                cinfo = clusters[insert_cid]
                
                # 更新成员列表
                cinfo["members_texts"].append(txt)
                cinfo["members_idx"].append(idx)
                
                # 更新正确率计数
                cinfo["correct_cnt"] += 1 if lead_to_correct else 0
                
                # 计算这个步骤的历史正确性（基于所在簇的正确率）
                # 注意：correct_cnt / len(members_texts) 是该簇的历史正确率
                correctness.append(
                    True if cinfo["correct_cnt"] / len(cinfo["members_texts"]) > self.correct_cluster_threshold
                    else False
                )
                
                # 把新成员向量加入成员矩阵（在第 0 轴上拼接）
                cinfo["member_embeddings"] = np.concatenate(
                    (cinfo["member_embeddings"], emb[None, :]), axis=0
                )
                
                # 根据 rep_mode 更新代表向量
                if self.rep_mode == "centroid":
                    # centroid 模式：代表向量更新为所有成员的均值（归一化后）
                    new_rep = self._normalize(np.mean(cinfo["member_embeddings"], 0))
                    cinfo["rep_embedding"] = new_rep
                    rep_matrix[insert_cid] = new_rep  # 同步更新矩阵
                
                elif self.rep_mode == "medoid":
                    # medoid 模式：代表向量更新为"离均值最近的成员"
                    centroid = np.mean(cinfo["member_embeddings"], 0)
                    sims_centroid = np.dot(cinfo["member_embeddings"], centroid)
                    best_idx = int(np.argmax(sims_centroid))
                    new_rep = cinfo["member_embeddings"][best_idx]
                    cinfo["rep_embedding"] = new_rep
                    cinfo["rep_text"] = cinfo["members_texts"][best_idx]
                    rep_matrix[insert_cid] = new_rep  # 同步更新矩阵
                
                # "first" 或 "all" 模式：代表向量固定为第一个成员，不需要更新
            
            else:
                # ── 新建簇（没有找到足够相似的现有簇） ──
                
                # 新簇的 ID = 当前簇数量（0-indexed 连续分配）
                new_cid = len(clusters)
                clusters[new_cid] = dict(
                    rep_embedding=emb,                         # 代表向量 = 这个新步骤
                    rep_text=txt,
                    members_texts=[txt],
                    members_idx=[idx],
                    member_embeddings=emb[None, :].copy(),     # (1, D) 形状的 ndarray
                    correct_cnt=1 if lead_to_correct else 0
                )
                
                # 把新簇的代表向量加入代表向量矩阵（在第 0 轴拼接，矩阵变大一行）
                rep_matrix = np.vstack([rep_matrix, emb[None, :]]).copy()
                rep_matrix.setflags(write=True)
                
                # 新簇的正确性：基于 lead_to_correct（第一个成员的实际正确性）
                correctness.append(True if lead_to_correct else False)
        
        return correctness
    
    def update_min_mean_correct_resp_len(self, sample_id: int, resp_len: int):
        """
        更新某道题的"正确回答长度统计"（最小值和均值）。
        
        每次这道题被正确回答后调用，记录该次回答的 token 长度。
        用于计算长度惩罚的基准值（奖励函数里的 ref_resp_lengths）。
        
        参数：
        - sample_id：题目 ID
        - resp_len：本次正确回答的 token 长度
        """
        # 更新最小长度（取历史最小值和当前值中的较小者）
        self.resp_len_stats[sample_id]["min_len"] = min(
            self.resp_len_stats[sample_id]["min_len"], resp_len
        )
        
        # 更新均值长度（在线均值计算公式：new_mean = (old_mean * cnt + new_val) / (cnt + 1)）
        self.resp_len_stats[sample_id]["mean_len"] = (
            self.resp_len_stats[sample_id]["mean_len"] * self.resp_len_stats[sample_id]["cnt"] + resp_len
        ) / (self.resp_len_stats[sample_id]["cnt"] + 1)
        
        # 更新计数器
        self.resp_len_stats[sample_id]["cnt"] += 1
    
    def look_up_min_mean_correct_resp_len(self, sample_id: int) -> int:
        """
        查询某道题历史上观测到的"最短正确回答长度"和"均值正确回答长度"。
        
        参数：
        - sample_id：题目 ID
        
        返回：
        - (min_len, mean_len)：最短长度和均值长度
          如果这道题从未被正确回答过，返回 (inf, 0.0)
        """
        # 用 .get() 防止 KeyError（没有记录则返回默认值）
        return (
            self.resp_len_stats.get(sample_id, {"min_len": float("inf"), "mean_len": 0.0})["min_len"],
            self.resp_len_stats.get(sample_id, {"min_len": float("inf"), "mean_len": 0.0})["mean_len"]
        )
    
    def look_up_step_correctness(
        self,
        sample_id: int,
        texts: Union[str, List[str]]
    ) -> List[bool]:
        """
        按字符串精确匹配，查询某个推理步骤文本的历史正确性。
        
        注意：这是精确文本匹配（不是向量相似度匹配），
        所以只能查询之前通过 update_sample_step_hash_dict 加入的步骤文本。
        
        参数：
        - sample_id：题目 ID
        - texts：待查询的步骤文本（单个字符串或字符串列表）
        
        返回：
        - 每个文本对应的正确性估计（True/False 列表）
        
        异常：
        - KeyError：如果该 sample_id 没有任何记录
        - ValueError：如果某个文本在所有簇里都找不到
        """
        # 统一成列表格式（便于统一处理单个和多个文本）
        if isinstance(texts, str):
            texts = [texts]
        
        # 获取该 sample 的所有簇
        clusters = self.dicts.get(sample_id, {})
        if not clusters:
            raise KeyError(f"No clusters found for sample_id {sample_id}")
        
        results: List[bool] = []
        
        for query in texts:
            found = False
            
            # 在所有簇中查找包含这个文本的簇
            for cinfo in clusters.values():
                if query in cinfo["members_texts"]:
                    # 找到了：根据所在簇的历史正确率判断
                    results.append(
                        True if cinfo['correct_cnt'] / len(cinfo["members_texts"]) > self.correct_cluster_threshold
                        else False
                    )
                    found = True
                    break
            
            if not found:
                # 在所有簇里都找不到这个文本
                raise ValueError(
                    f'Text "{query}" not found in any cluster for sample_id {sample_id}'
                )
        
        return results
    
    def get_step_dict_info(self, verbose_info: bool = False, print_info: bool = False):
        """
        获取整个字典的统计摘要（用于监控训练过程中的"步骤多样性"）。
        
        参数：
        - verbose_info：是否包含每个簇的详细信息（成员文本、正确率等）
        - print_info：是否同时打印统计信息到控制台
        
        返回：
        - info_dict：结构为 {sample_id: {"overall_info": {...}, "verbose_info": [...]}}
        """
        info_dict = defaultdict(dict)
        
        if print_info:
            print(f"Total samples: {len(self.dicts)}")
        
        for sample_id, clusters in self.dicts.items():
            # 计算每个 sample 的平均成员数（衡量步骤重复率）
            avg_member_len = np.mean([len(cinfo["members_texts"]) for cinfo in clusters.values()])
            
            if print_info:
                print(f"Sample ID: {sample_id}, Clusters: {len(clusters)}, Avg Members: {avg_member_len:.2f}")
            
            # 摘要信息：簇数量 + 平均成员数
            info_dict[sample_id]["overall_info"] = {
                "clusters_cnt": len(clusters),
                "avg_member_len": avg_member_len
            }
            
            if verbose_info:
                # 详细信息：每个簇的代表文本、成员数、正确率
                info_dict[sample_id]["verbose_info"] = []
                for cid, cinfo in clusters.items():
                    if print_info:
                        print(
                            f"  Cluster ID: {cid}, Rep text: {cinfo['rep_text'][:80]}, "
                            f"Members: {len(cinfo['members_texts'])}, "
                            f"Acc: {cinfo['correct_cnt'] / len(cinfo['members_texts']) if cinfo['members_texts'] else 0}"
                        )
                    info_dict[sample_id]["verbose_info"].append({
                        "cluster_id": cid,
                        "rep_text": cinfo["rep_text"][:80],              # 只取前 80 字符防止输出过长
                        "members_count": len(cinfo["members_texts"]),
                        "sampled_member_texts": cinfo["members_texts"],
                        "lead_to_correct": cinfo["correct_cnt"],          # 能导向正确答案的成员数
                        "accuracy": cinfo["correct_cnt"] / len(cinfo["members_texts"]) if cinfo["members_texts"] else 0,
                    })
        
        return info_dict
    
    def save_info(self, filepath: str, overwrite: bool = True) -> None:
        """
        把当前字典序列化保存到磁盘（用于断点续训或分析）。
        
        会在 filepath 目录下生成两个文件：
        - step_hash_dict.pkl：主字典（簇信息）
        - resp_len_stats.pkl：响应长度统计
        
        参数：
        - filepath：保存目录路径（必须已存在）
        - overwrite：是否覆盖已有文件（False 时如果文件存在会抛异常）
        """
        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(
                f"{filepath} already exists. Set overwrite=True to overwrite."
            )
        
        # 把 defaultdict 转为普通 dict 再保存（更通用，加载时不依赖原来的 lambda）
        dicts_to_dump = dict(self.dicts)
        resp_len_stats_to_dump = dict(self.resp_len_stats)
        
        # 保存主字典
        with open(os.path.join(filepath, 'step_hash_dict.pkl'), "wb") as f:
            pickle.dump(dicts_to_dump, f)
        
        # 保存响应长度统计
        with open(os.path.join(filepath, 'resp_len_stats.pkl'), "wb") as f:
            pickle.dump(resp_len_stats_to_dump, f)
        
        print(f"[StepHashDict] and [RespLenStats] saved to folder {filepath}")
    
    def load_info(self, filepath: str) -> None:
        """
        从磁盘加载字典，覆盖当前内存中的字典（用于恢复训练状态）。
        
        参数：
        - filepath：之前 save_info 保存的目录路径
        
        异常：
        - FileNotFoundError：如果目录不存在
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        
        # 加载主字典
        with open(os.path.join(filepath, 'step_hash_dict.pkl'), "rb") as f:
            dicts_loaded = pickle.load(f)
        
        # 加载响应长度统计
        with open(os.path.join(filepath, 'resp_len_stats.pkl'), "rb") as f:
            resp_len_stats_loaded = pickle.load(f)
        
        # 包回 defaultdict，保持"访问不存在的 key 时返回默认值"的行为
        self.dicts = defaultdict(dict, dicts_loaded)
        self.resp_len_stats = defaultdict(
            lambda: {"min_len": float("inf"), "mean_len": 0.0, "cnt": 0},
            resp_len_stats_loaded
        )
        
        print(f"[StepHashDict] and [RespLenStats] loaded dicts from folder {filepath}")


# ─────────────────────── 代码示例（已注释，仅供参考） ───────────────────────
# 使用示例：
'''
d = StepHashDict(similarity_threshold=0.85, rep_mode="medoid")
d.update_sample_step_hash_dict(
    sample_id=1,
    embeddings=np.array([[0.1, 0.2], [0.2, 0.3], [0.9, 0.8]]),
    texts=["", "a", "b"],
    lead_correct_list=[True, False, True]
)

d.look_up_step_correctness(
    sample_id=1,
    texts=["", "b", "c"]
)
'''


class SampleHashDict:
    """
    样本（题目）级状态字典。
    
    数据结构概览：
    self.dicts = {
        sample_id: {
            "corret_answered": False,    # 这道题是否曾被正确回答（注意：字段名拼写有误，保持兼容）
            "min_len": inf               # 见过的最短回答长度
        },
        ...
    }
    
    self.resp_len_stats = {
        sample_id: {"min_len": ..., "mean_len": ..., "cnt": ...},
        ...
    }
    
    特点：
    - "corret_answered" 是单调递增的：一旦设为 True，永远保持 True（幂等）
    - 用于 advantage 规范化：如果某道题从未被答对，跳过规范化（避免 NaN）
    """
    
    def __init__(self):
        """初始化样本级状态字典"""
        # 主字典：sample_id → {correct_answered, min_len}
        # 默认值：从未回答过（False），最短长度为无穷大
        self.dicts: Dict[int, dict] = defaultdict(
            lambda: {"corret_answered": False, "min_len": float("inf")}
        )
        
        # 响应长度统计（与 StepHashDict 接口一致，便于上层统一调用）
        self.resp_len_stats: Dict[int, dict] = defaultdict(
            lambda: {"min_len": float("inf"), "mean_len": 0.0, "cnt": 0}
        )
    
    def set_correct_answered(self, sample_id: int, value: bool) -> None:
        """
        标记某道题是否被正确回答。
        
        幂等逻辑：
        - 如果 value=True，无论历史值是什么，设为 True
        - 如果 value=False，但历史值已经是 True（曾经答对过），保持 True
        - 确保"正确性"只增不减
        
        参数：
        - sample_id：题目 ID
        - value：本次是否正确回答了
        """
        info = self.dicts[sample_id]
        # or 运算保证幂等：历史为 True → 结果为 True（无论 value）
        info["corret_answered"] = bool(info.get("corret_answered", False) or value)
    
    def get_info(self, sample_id: int) -> dict:
        """
        获取某道题的完整状态信息（浅拷贝，防止调用方意外修改内部状态）。
        
        参数：
        - sample_id：题目 ID
        
        返回：
        - 状态字典的浅拷贝：{"corret_answered": bool, "min_len": float}
        """
        info = self.dicts[sample_id]
        return dict(info)  # 返回浅拷贝
    
    def update_min_mean_correct_resp_len(self, sample_id: int, resp_len: int):
        """
        更新某道题的正确回答长度统计。
        
        参数：
        - sample_id：题目 ID
        - resp_len：本次正确回答的 token 长度
        """
        stats = self.resp_len_stats[sample_id]
        
        # 更新最小长度
        stats["min_len"] = min(stats["min_len"], resp_len)
        
        # 在线均值更新：new_mean = (old_mean × old_cnt + new_val) / new_cnt
        stats["mean_len"] = (stats["mean_len"] * stats["cnt"] + resp_len) / (stats["cnt"] + 1)
        
        # 增加计数器
        stats["cnt"] += 1
        
        # 同步更新主字典里的 min_len 字段
        info = self.dicts[sample_id]
        info["min_len"] = min(info.get("min_len", float("inf")), resp_len)
        
        return None
    
    def look_up_min_mean_correct_resp_len(self, sample_id: int) -> int:
        """
        查询某道题的最短/均值正确回答长度。
        
        参数：
        - sample_id：题目 ID
        
        返回：
        - (min_len, mean_len)：如果从未记录过，返回 (inf, 0.0)
        """
        stats = self.resp_len_stats.get(sample_id, {"min_len": float("inf"), "mean_len": 0.0})
        return stats["min_len"], stats["mean_len"]
    
    def save_info(self, filepath: str, overwrite: bool = True) -> None:
        """
        把样本状态字典保存到磁盘。
        
        会在 filepath 目录下生成两个文件：
        - sample_hash_dict.pkl：主字典（正确性 + 最短长度）
        - sample_resp_len_stats.pkl：响应长度统计
        
        参数：
        - filepath：保存目录
        - overwrite：是否覆盖已有文件
        """
        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(
                f"{filepath} already exists. Set overwrite=True to overwrite."
            )
        
        # 转为普通 dict 以提升兼容性
        dicts_to_dump = dict(self.dicts)
        resp_len_stats_to_dump = dict(self.resp_len_stats)
        
        with open(os.path.join(filepath, 'sample_hash_dict.pkl'), 'wb') as f:
            pickle.dump(dicts_to_dump, f)
        
        with open(os.path.join(filepath, 'sample_resp_len_stats.pkl'), 'wb') as f:
            pickle.dump(resp_len_stats_to_dump, f)
        
        print(f"[SampleHashDict] and [SampleRespLenStats] saved to folder {filepath}")
    
    def load_info(self, filepath: str) -> None:
        """
        从磁盘加载样本状态字典（恢复训练状态）。
        
        参数：
        - filepath：之前 save_info 保存的目录路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        
        with open(os.path.join(filepath, 'sample_hash_dict.pkl'), 'rb') as f:
            dicts_loaded = pickle.load(f)
        
        with open(os.path.join(filepath, 'sample_resp_len_stats.pkl'), 'rb') as f:
            resp_len_stats_loaded = pickle.load(f)
        
        # 包回 defaultdict，保持默认值行为
        self.dicts = defaultdict(
            lambda: {"corret_answered": False, "min_len": float("inf")},
            dicts_loaded
        )
        self.resp_len_stats = defaultdict(
            lambda: {"min_len": float("inf"), "mean_len": 0.0, "cnt": 0},
            resp_len_stats_loaded
        )
        
        print(f"[SampleHashDict] and [SampleRespLenStats] loaded dicts from folder {filepath}")
