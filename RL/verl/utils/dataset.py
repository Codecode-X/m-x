"""
本文件的作用：定义 RL 训练用的数据集类（Dataset）和图像处理工具。

RL 训练与 SFT 训练的数据集的区别：
- SFT：数据集直接提供 (input, output) 对，模型做监督学习
- RL：数据集只提供 (input, ground_truth)，模型先生成回答，再用奖励函数评分
  因此 RL 数据集只需要准备好"题目 + 答案"，不需要参考输出

主要类和函数：

1. RLHFDataset：RL 训练的主数据集（PyTorch Dataset 子类）
   - 从 parquet/HuggingFace Hub 加载数据
   - 对每道题目进行分词、图像预处理、position_ids 计算
   - 支持过长/无效 prompt 的过滤（缓存结果避免重复计算）
   - 支持多模态输入（图片 + 文字）

2. CorrectAnswerDataset：从"历史正确回答池"中采样的数据集
   - 用于 VLPO 的离线难度采样（只对模型已经做对过的题目再训练）

3. ImageProcessMixin：图像尺寸归一化 mixin 类
   - 确保所有图片在 [min_pixels, max_pixels] 范围内

数据流（RLHFDataset.__getitem__ 的完整流程）：
    原始样本 (dict)
    → _build_example()：从数据集特定格式构建统一的 {images, problem, answer}
    → _build_messages()：构建 HuggingFace 聊天消息格式 [{role, content}]
    → processor / tokenizer：tokenize + 图像处理 → input_ids, attention_mask
    → get_rope_index()：计算 Qwen2.5-VL 的 MRoPE position_ids
    → postprocess_data()：左 padding 到 max_prompt_length
    → 最终 example（包含 input_ids、attention_mask、position_ids、raw_prompt_ids、ground_truth 等）
"""

# 数学工具（用于图像尺寸缩放比例计算）
import math
# 文件路径工具
import os
# 用于收集 features 时按 key 分组
from collections import defaultdict
# 字节流（图像 bytes → PIL Image）
from io import BytesIO
# 调试工具
import pdb
# 类型注解
from typing import Any, Dict, List, Optional, Union

# 数值计算
import numpy as np
# PyTorch
import torch
# HuggingFace datasets 库：加载 parquet 和 Hub 数据集
from datasets import load_dataset
# Jinja2 模板引擎（用于 format_prompt 功能，把题目内容嵌入到提示词模板中）
from jinja2 import Template
# PIL 图像库
from PIL import Image
from PIL.Image import Image as ImageObject
# PyTorch Dataset 基类
from torch.utils.data import Dataset
# HuggingFace tokenizer/processor 基类
from transformers import PreTrainedTokenizer, ProcessorMixin

# Qwen2.5-VL 的 MRoPE 位置编码计算（多模态旋转位置编码）
from ..models.transformers.qwen2_vl import get_rope_index
# 张量工具函数（postprocess_data 等）
from . import torch_functional as VF
# 随机采样工具
import random
# Base64 解码、字节流、正则表达式（用于图像 Base64 字符串解析）
import base64, io, re
# glob 模式匹配（用于枚举目录下的所有 parquet 文件）
import glob


def dataset_name_from_path(data_path: str) -> str:
    """
    根据数据路径推断数据集名称。
    
    用于生成"有效样本 ID 缓存"的目录路径（不同数据集的缓存单独存放）。
    
    参数：data_path - 数据集路径字符串
    返回：数据集名称字符串
    
    异常：NotImplementedError - 如果路径不匹配任何已知数据集
    """
    if "math3k" in data_path or "geometry3k" in data_path:
        return "Geometry3K"
    elif "math" in data_path:
        return "Math3K"
    elif "Thyme-RL" in data_path and 'val' not in data_path:
        return "Thyme-train"
    elif "Thyme-RL" in data_path and 'val' in data_path:
        return "Thyme-val"
    else:
        raise NotImplementedError(f"Dataset {data_path} not supported yet.")


def b64_to_pil(s: str) -> Image.Image:
    """
    把 Base64 编码（可选 data URL 前缀）的图像字符串解码为 PIL Image。
    
    支持格式：
    - 纯 Base64 字符串
    - data URL（如 "data:image/png;base64,iVBORw0KGgo..."）
    
    注意：这里不做 RGB 转换，留给后续 process_image 统一处理，
    避免提前完整解码（尤其是 JPEG 图片的 progressive decode 优化）。
    
    参数：s - Base64 图像字符串
    返回：PIL Image 对象
    """
    # 去掉 data URL 前缀（如果有）
    s = re.sub(r'^\s*data:image/[^;]+;base64,', '', s.strip(), flags=re.I)
    # 补全 Base64 padding（Base64 字符串长度必须是 4 的倍数）
    s += '=' * (-len(s) % 4)
    # 解码 Base64 → 字节流 → PIL Image（懒加载，不立即完整解码）
    return Image.open(io.BytesIO(base64.b64decode(s)))


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    DataLoader 的 collate 函数：把一个 batch 的样本列表合并为一个 dict。
    
    合并规则：
    - 张量类型（torch.Tensor）：沿 dim=0 堆叠（stack）
    - 非张量类型（str、list、PIL Image 等）：用 numpy object array 打包
    
    为什么不全用 torch.stack？
    - 图像（PIL Image）、字符串、变长列表无法直接 stack
    - numpy object array 可以存放任意 Python 对象
    
    参数：features - 一个 batch 的样本列表（每个样本是 __getitem__ 返回的 dict）
    返回：合并后的 batch dict（键名相同，值变为 batched 格式）
    """
    tensors = defaultdict(list)    # 张量类字段
    non_tensors = defaultdict(list)  # 非张量类字段
    
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)
    
    # 张量字段：stack 成 (batch_size, ...) 的批量张量
    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)
    
    # 非张量字段：打包为 numpy object array
    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)
    
    return {**tensors, **non_tensors}


class ImageProcessMixin:
    """
    图像预处理 Mixin 类（不能单独使用，需要与 Dataset 组合）。
    
    职责：
    - 把图像缩放到 [min_pixels, max_pixels] 范围内（控制 token 数量）
    - 转换为 RGB 格式（确保颜色通道一致性）
    
    Qwen2.5-VL 的图像 token 数量与图像像素数成正比，
    max_pixels 限制了每张图的最大 token 数，防止一道题里的图片占用过多 context。
    """
    
    max_pixels: int  # 图像最大像素数（超过则缩小）
    min_pixels: int  # 图像最小像素数（低于则放大）
    
    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        """
        对图像做尺寸归一化。
        
        参数：
        - image：PIL Image 对象，或包含 "bytes" 字段的 dict（HuggingFace 格式）
        
        返回：处理后的 RGB PIL Image（尺寸在 [min_pixels, max_pixels] 范围内）
        """
        # 支持 dict 格式（HuggingFace datasets 中图片可能以字节形式存储）
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        
        # 如果图片太大（像素数超过 max_pixels），按比例缩小
        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width = int(image.width * resize_factor)
            height = int(image.height * resize_factor)
            
            # JPEG/WEBP 格式支持在解码时直接下采样（节省内存）
            fmt = (getattr(image, "format", None) or "").upper()
            if fmt in {"JPEG", "JPG", "WEBP"}:
                try:
                    image.draft("RGB", (width, height))  # 告诉解码器目标分辨率
                except Exception:
                    pass
            image = image.resize((width, height))
        
        # 如果图片太小（像素数低于 min_pixels），按比例放大
        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width = int(image.width * resize_factor)
            height = int(image.height * resize_factor)
            image = image.resize((width, height))
        
        # 统一转为 RGB（排除 RGBA/L/P 等格式的影响）
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image


class RLHFDataset(Dataset, ImageProcessMixin):
    """
    RL 训练用的数据集类（RLHF = Reinforcement Learning from Human Feedback）。
    
    在 Monet 中，数据集包含视觉数学题（图片 + 题目文字 + 答案）。
    
    数据集结构（以 Thyme-RL 为例）：
    - images：图片列表（Base64 编码的字符串）
    - question：题目文字
    - solution：标准答案
    
    数据集结构（以 Geometry3K 为例）：
    - image：图片文件路径
    - problem：题目文字
    - answer：标准答案
    
    每个样本经过 __getitem__ 处理后包含：
    - input_ids：左 padding 后的 prompt token IDs（长度固定为 max_prompt_length）
    - attention_mask：有效 token 的掩码
    - position_ids：MRoPE 位置编码（3D 张量，形状 (3, seq_len)）
    - raw_prompt_ids：未 padding 的原始 prompt token IDs（供 vLLM 使用）
    - ground_truth：标准答案
    - global_index：在整个数据集中的索引（用于哈希服务器查询）
    - multi_modal_data：图片列表（PIL Image 对象，供 vLLM 使用）
    """
    
    def __init__(
        self,
        data_path: str,                           # 数据集路径（本地 parquet 目录/文件 或 Hub 路径）
        tokenizer: PreTrainedTokenizer,           # 文本 tokenizer
        processor: Optional[ProcessorMixin],     # 多模态 processor（None 表示纯文本模型）
        prompt_key: str = "prompt",               # 数据集中题目字段的键名
        answer_key: str = "answer",               # 数据集中答案字段的键名
        image_key: str = "images",                # 数据集中图片字段的键名
        max_prompt_length: int = 1024,            # 最大 prompt 长度（超过则截断或报错）
        truncation: str = "error",                # 截断策略："left"/"right"/"error"
        format_prompt: Optional[str] = None,     # 提示词模板文件路径（Jinja2 格式）
        max_pixels: Optional[int] = None,         # 图片最大像素数
        min_pixels: Optional[int] = None,         # 图片最小像素数
        filter_overlong_and_invalid_prompts: bool = True,  # 是否过滤过长/无效的 prompt
    ):
        """
        初始化数据集。
        
        支持三种数据源：
        1. 目录（包含多个 parquet 文件）
        2. 单个 parquet 文件
        3. HuggingFace Hub 数据集路径（如 "username/dataset_name"）
        
        过滤逻辑：
        - 第一次运行时，遍历所有样本检查 prompt 是否超长
        - 把有效样本的 ID 缓存到文件（./examples/dataset_valid_ids/{dataset_name}/valid_ids.txt）
        - 之后的运行直接读取缓存，跳过重复过滤（节省时间）
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.filter_overlong_and_invalid_prompts = filter_overlong_and_invalid_prompts
        
        # ── 解析数据集路径（支持 "path@split" 格式指定数据切分）──
        if "@" in data_path:
            data_path, data_split = data_path.split("@")  # 如 "data/train.parquet@train"
        else:
            data_split = "train"  # 默认使用训练切分
        
        # ── 加载数据集 ──
        if os.path.isdir(data_path):
            # 目录：枚举所有根目录下的 parquet 文件（不递归进子目录）
            root_parquets = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
            if root_parquets:
                # 把所有 parquet 文件合并为一个数据集
                self.dataset = load_dataset("parquet", data_files=root_parquets, split="train")
            else:
                # 没有找到 parquet 文件，让 datasets 库自己推断格式
                self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            # 单个文件
            self.dataset = load_dataset("parquet", data_files=data_path, split=data_split)
        else:
            # HuggingFace Hub 路径
            self.dataset = load_dataset(data_path, split=data_split)
        
        # ── 加载 format_prompt 模板（如果提供了）──
        self.format_prompt_path = format_prompt
        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()  # 读入 Jinja2 模板字符串
        
        # ── 过滤过长/无效的 prompt ──
        
        self._valid_ids_cache_path = None
        
        if self.filter_overlong_and_invalid_prompts:
            # 构建缓存文件路径（按数据集名称分目录存储）
            dataset_name = dataset_name_from_path(self.data_path)
            cache_dir = os.path.join("./examples/dataset_valid_ids", dataset_name)
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "valid_ids.txt")
            self._valid_ids_cache_path = cache_file
            
            if os.path.exists(cache_file):
                # ── 快速路径：直接读取缓存的有效 ID 列表 ──
                with open(cache_file, "r") as f:
                    valid_ids = [int(line.strip()) for line in f if line.strip()]
                
                # 安全检查：确保缓存的 ID 没有超出当前数据集的范围
                max_id = max(valid_ids) if valid_ids else -1
                if max_id >= len(self.dataset):
                    raise ValueError(
                        f"Cached ids out of range: max_id={max_id}, dataset_len={len(self.dataset)}. "
                        "Make sure the dataset order/size matches the cache."
                    )
                
                # 直接用 HuggingFace datasets 的 select 方法选取有效样本
                self.dataset = self.dataset.select(valid_ids)
            
            else:
                # ── 慢速路径：第一次运行，遍历过滤并缓存结果 ──
                
                orig_idx_col = "__orig_idx__"
                
                # 先清理可能残留的辅助列
                if orig_idx_col in self.dataset.column_names:
                    self.dataset = self.dataset.remove_columns([orig_idx_col])
                
                # 添加原始索引列（过滤后需要知道被保留的样本的原始 ID）
                self.dataset = self.dataset.add_column(orig_idx_col, list(range(len(self.dataset))))
                
                # 执行过滤（遍历所有样本，去掉 prompt 超长的）
                filtered = self.dataset.filter(
                    self._filter_overlong_and_invalid_prompts,
                    desc="Filtering overlong prompts"
                )
                
                # 提取被保留的原始 ID 列表
                valid_ids = filtered[orig_idx_col]
                
                # 原子写入缓存文件（先写临时文件，再 rename，防止写入中途崩溃）
                tmp_file = cache_file + ".tmp"
                with open(tmp_file, "w") as f:
                    for idx in valid_ids:
                        f.write(f"{int(idx)}\n")
                os.replace(tmp_file, cache_file)  # 原子替换
                
                # 删除辅助列，保存最终过滤后的数据集
                self.dataset = filtered.remove_columns([orig_idx_col])
            
            print(f"Dataset size for training: {len(self.dataset)}")
    
    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        把样本转换为 HuggingFace 聊天消息格式。
        
        消息格式：
        - 纯文字：[{"role": "user", "content": "题目文字"}]
        - 多模态（含图片）：
          [{"role": "user", "content": [
              {"type": "image"},    ← 每个 <image> 占位符对应一张图片
              {"type": "text", "text": "题目文字的剩余部分"}
          ]}]
        
        参数：example - 经过 _build_example 处理后的样本 dict
        返回：消息列表（符合 HuggingFace chat_template 格式）
        """
        # 获取 prompt 字符串
        prompt_str: str = example[self.prompt_key]
        
        # 如果提供了 Jinja2 模板，把 prompt_str 嵌入模板中
        # 模板例如："Think carefully. {{content}}\nPut your answer in \\boxed{}."
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)
        
        if self.image_key in example:
            # ── 多模态消息：按 <image> 切分文字，每个切分点插入一个 {"type": "image"} ──
            content_list = []
            
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    # 不是第一段，在前面插入图片占位符
                    content_list.append({"type": "image"})
                
                if content:  # 非空文字段才插入
                    content_list.append({"type": "text", "text": content})
            
            return [{"role": "user", "content": content_list}]
        else:
            # 纯文字消息
            return [{"role": "user", "content": prompt_str}]
    
    def _filter_overlong_and_invalid_prompts(self, example: Dict[str, Any]) -> bool:
        """
        判断一个样本是否有效且 prompt 不超长。
        
        过滤逻辑：
        1. 先用 _build_example 转换成统一格式（如果失败则返回 False）
        2. 构建消息格式
        3. 用 processor/tokenizer 计算 token 数量
        4. 如果 token 数超过 max_prompt_length，返回 False（过滤掉）
        
        参数：example - 原始数据集样本
        返回：True（保留）或 False（过滤）
        """
        if 'geometry3k' not in self.data_path:
            if 'Thyme-RL' in self.data_path:
                example = self._build_example(example, dataset_name="Thyme")
            else:
                raise NotImplementedError(f"Dataset {self.data_path} not supported yet.")
            
            if not example:
                return False  # _build_example 返回空 dict，说明样本无效
        
        messages = self._build_messages(example)
        processing_class = self.processor if self.processor is not None else self.tokenizer
        
        # 计算 prompt 的 token 长度（不包括生成内容）
        return len(processing_class.apply_chat_template(messages, add_generation_prompt=True)) <= self.max_prompt_length
    
    def __len__(self):
        """返回数据集的有效样本数。"""
        return len(self.dataset)
    
    def _build_example(self, example, dataset_name):
        """
        把数据集原始格式的样本转换为统一格式 {images, problem, answer}。
        
        不同数据集的字段名和图片格式不同，这个函数做适配转换：
        - Thyme-RL：images 是 Base64 编码的字符串列表，question/solution 字段
        
        参数：
        - example：原始数据集样本（dict）
        - dataset_name：数据集名称（用于选择转换逻辑）
        
        返回：统一格式的样本 dict（如果样本无效，返回空 dict {}）
        """
        data = {}
        
        if dataset_name == "Thyme":
            # 过滤掉没有图片或超过 1 张图片的样本（这类样本可能有问题）
            if not example["images"] or len(example["images"]) > 1:
                return {}  # 空 dict 表示无效样本
            
            # 把 Base64 图片字符串解码为 PIL Image
            img = b64_to_pil(example["images"][0])
            data["images"] = [img]
            
            # 构建 prompt：在题目文字前面加 <image> 占位符
            data["problem"] = "<image>" + example["question"]
            data["answer"] = example["solution"]
        
        return data
    
    def __getitem__(self, index):
        """
        获取第 index 个样本，返回经过完整预处理的 dict。
        
        处理步骤：
        1. 从数据集读取原始样本
        2. _build_example：转换为统一格式
        3. _build_messages：构建聊天消息格式
        4. processor / tokenizer：tokenize 和图像处理
        5. get_rope_index：计算 MRoPE position_ids（Qwen2.5-VL 专用）
        6. postprocess_data：左 padding 到 max_prompt_length
        
        返回的 dict 包含：
        - input_ids：(max_prompt_length,) 的 token ID 张量（左 padding）
        - attention_mask：(max_prompt_length,) 的 attention mask 张量
        - position_ids：(3, max_prompt_length) 的 MRoPE 位置编码张量
        - raw_prompt_ids：未 padding 的 token ID 列表（供 vLLM 直接使用）
        - ground_truth：标准答案字符串
        - global_index：样本在原始数据集中的全局索引（供哈希服务器使用）
        - multi_modal_data：{"image": [PIL Image]} （如果有图片）
        - multi_modal_inputs：其他 processor 输出字段（如 image_grid_thw）
        """
        # 读取原始样本
        example: dict = self.dataset[index]
        
        # ── 数据格式转换 ──
        if 'geometry3k' not in self.data_path:
            if 'Thyme-RL' in self.data_path:
                example = self._build_example(example, dataset_name="Thyme")
            else:
                raise NotImplementedError(f"Dataset {self.data_path} not supported yet.")
        
        # ── 构建聊天消息格式 ──
        messages = self._build_messages(example)
        
        if self.image_key in example:
            # ── 多模态输入处理 ──
            
            # 生成文字提示词（不 tokenize，因为 processor 会一起处理图片和文字）
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            # 对所有图片做尺寸归一化（限制在 [min_pixels, max_pixels] 范围内）
            images = [self.process_image(image) for image in example.pop(self.image_key)]
            
            # 用 processor 同时处理图片和文字
            # 返回：input_ids, attention_mask, image_grid_thw（图片 patch 网格大小）等
            model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]         # 去掉 batch 维度
            attention_mask = model_inputs.pop("attention_mask")[0]
            
            # 保存图片和其他 processor 输出（供 vLLM 使用）
            example["multi_modal_data"] = {"image": images}       # PIL Image 列表
            example["multi_modal_inputs"] = dict(model_inputs)    # 其他 processor 输出（image_grid_thw 等）
        else:
            # ── 纯文字输入处理 ──
            
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
        
        # ── 计算 position_ids ──
        
        if (
            self.processor is not None
            and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor"
        ):
            # Qwen2.5-VL 使用 MRoPE（多模态旋转位置编码）
            # 视觉 token 和文字 token 有不同的位置编码策略
            # 返回形状：(3, seq_len)，3 对应 MRoPE 的时间/高度/宽度三个维度
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),  # 图片 patch 网格信息
                attention_mask=attention_mask,
            )
        else:
            # 普通模型：position_ids 就是从 0 开始的累积注意力掩码
            # cumsum - 1 把 [0,0,1,1,1] 转换为 [0,0,0,1,2]（left padding 下的正确位置）
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)
        
        # ── 左 Padding 到 max_prompt_length ──
        
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,          # 左 padding（因为生成时需要对齐右边界）
            truncation=self.truncation,  # 截断策略
        )
        
        # ── 准备 raw_prompt_ids（不 padding，供 vLLM 使用）──
        
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        # 如果 raw_prompt_ids 超长，根据截断策略处理
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]  # 保留末尾
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]   # 保留开头
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
        
        # ── 整理最终样本 dict ──
        
        example["input_ids"] = input_ids                          # Padded token IDs
        example["attention_mask"] = attention_mask                # Attention mask
        example["position_ids"] = position_ids                   # MRoPE position IDs
        example["raw_prompt_ids"] = raw_prompt_ids               # 原始 token IDs（给 vLLM）
        example["ground_truth"] = example.pop(self.answer_key)   # 标准答案
        example["prompt_key"] = self.prompt_key                  # 字段名（调试用）
        example["answer_key"] = self.answer_key                  # 字段名（调试用）
        example["image_key"] = self.image_key                    # 字段名（调试用）
        example["prompt_before_processor"] = prompt               # 原始提示词文字（调试用）
        example["global_index"] = index                          # 全局索引（供哈希服务器查询）
        
        return example


class CorrectAnswerDataset(Dataset):
    """
    从"历史正确回答池"中构建的数据集（VLPO 离线难度采样专用）。
    
    用途：
    VLPO 的一种训练策略是"只对模型曾经做对过的题目继续训练"：
    - 对从未做对的题（太难）：跳过，模型可能还没有足够能力
    - 对总是做对的题（太简单）：也跳过，已经学会了
    - 对偶尔做对的题（难度适中）：重点训练
    
    correct_pool 的格式：
    {题目全局 ID（int）: [正确回答1（str）, 正确回答2（str）, ...]}
    
    每次 __getitem__ 随机从该题的正确回答池中选一个，
    作为 prev_correct_answer 传给奖励函数（作为长度惩罚的参考长度来源）。
    """
    
    def __init__(self, base_dataset, correct_pool):
        """
        参数：
        - base_dataset：原始的 RLHFDataset（提供完整的样本信息）
        - correct_pool：正确回答池 {题目ID: [正确回答列表]}
        """
        self.base_dataset = base_dataset
        self.correct_pool = correct_pool
        # 只对 correct_pool 中有记录的题目训练
        self.question_ids = list(correct_pool.keys())
    
    def __len__(self):
        """返回有历史正确回答的题目数量。"""
        return len(self.question_ids)
    
    def __getitem__(self, idx):
        """
        获取第 idx 个样本，从正确回答池中随机选一个正确回答附在样本上。
        
        参数：idx - 在 question_ids 列表中的索引（不是全局索引）
        返回：base_dataset 的完整样本 + "prev_correct_answer" 字段
        """
        # 获取题目的全局 ID
        qid = self.question_ids[idx]
        
        # 从 base_dataset 获取完整样本（含 input_ids、ground_truth 等）
        sample = self.base_dataset[qid]
        
        # 从正确回答池中随机选一个历史正确回答（作为参考长度基线）
        answer = random.choice(self.correct_pool[qid])
        sample["prev_correct_answer"] = answer
        
        return sample
