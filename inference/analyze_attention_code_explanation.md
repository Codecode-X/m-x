# `analyze_attention.py` 代码逐行详细讲解

> **面向初学者**：本文档将从零开始，逐步拆解每一个函数、每一行关键代码，不仅解释"做了什么"，更深入解释"为什么这样做"以及背后的工程考量。

---

## 目录

1. [第一部分：背景知识 —— 你需要先了解什么](#第一部分背景知识)
2. [第二部分：整体架构与四步策略](#第二部分整体架构与四步策略)
3. [第三部分：逐函数详细讲解](#第三部分逐函数详细讲解)
   - 3.1 [文件头部：导入与常量定义](#31-文件头部导入与常量定义)
   - 3.2 [`replace_abs_vis_token_content` 函数](#32-replace_abs_vis_token_content-函数)
   - 3.3 [`load_model` 函数 —— 模型加载](#33-load_model-函数)
   - 3.4 [`prepare_input_for_sample` 函数 —— 输入准备](#34-prepare_input_for_sample-函数)
   - 3.5 [`classify_token_positions` 函数 —— Token 分类](#35-classify_token_positions-函数)
   - 3.6 [`compute_attention_allocation` 函数 —— 注意力分配计算](#36-compute_attention_allocation-函数)
   - 3.7 [`create_heatmaps` 函数 —— 热力图生成](#37-create_heatmaps-函数)
   - 3.8 [`main` 函数 —— 主流程](#38-main-函数主流程)
4. [第四部分：核心工程细节深入](#第四部分核心工程细节深入)
5. [第五部分：术语速查表](#第五部分术语速查表)

---

## 第一部分：背景知识

### 1.1 什么是 Monet 模型？

Monet 是一个基于 Qwen2.5-VL（视觉语言模型）改进的模型，它引入了**Latent Chain-of-Thought（隐式思维链）**机制。

**通俗类比**：
- 普通的 LLM 回答问题时，会把"思考过程"用文字写出来（比如 CoT: "首先..., 然后..., 所以..."），这些文字会占用 token。
- Monet 模型不同：它在内部用**不可见的 latent tokens**（隐式 token）进行"思考"，这些 token 不会出现在最终输出中，但模型的回答质量会因为这些"隐式思考"而更好。

**关键概念**：
- **Latent tokens**：模型内部的"思考单元"，类似于人类大脑中的思维活动，不会转化为可见文字，但影响最终输出。
- **`<abs_vis_token>` 和 `</abs_vis_token>`**：latent 区域的边界标记，就像一对括号，中间包裹着 latent tokens。
- **`<abs_vis_token_pad>`**：latent 区域内部的占位 token，训练时会被真正的 latent embedding 替换。

### 1.2 什么是注意力（Attention）？

在 Transformer 模型中，每个 token 在生成时都会"看向"序列中的其他 token，这种"看向"的程度就是**注意力权重**。

- 比如，当模型生成"因为"这个词时，它可能会把 80% 的注意力分配给前面的"下雨"，只有 5% 分给 latent tokens。
- 注意力分析的核心问题是：**模型在生成每个文字 token 时，把多少注意力分配给了那些隐式思考的 latent tokens？**

如果 latent tokens 获得了较高的注意力占比，说明模型确实在"参考"自己的隐式思考过程来生成文字回答 —— 这就验证了 Latent CoT 机制的有效性。

### 1.3 为什么需要两步前向传播？

这是一个关键的工程问题。理解它需要先知道模型内部的工作方式：

**`latent_mode=True` 的行为**：
- 模型会逐个处理 latent token
- 每个 latent token 的 embedding = 前一个 token 的隐藏状态（也就是模型把"前一步的思考结果"作为"下一步思考的输入"）
- 这就是"隐式思考链"的实现方式
- 但这个模式下，模型的注意力计算方式特殊（逐 token 步进），**不支持 `output_attentions=True`**

**`latent_mode=False` + `ce_patch_pos/ce_patch_vec` 的行为**：
- 模型一次性处理整个序列
- 之前 latent tokens 的位置会被替换为真正的 latent embeddings（由 Phase 1 生成）
- 这个模式**支持 `output_attentions=True`**，可以获取标准的注意力权重矩阵

所以策略是：
1. 先用 `latent_mode=True` 获取真正的 latent embeddings（Phase 1）
2. 再用 `latent_mode=False` + 替换 latent embeddings + `output_attentions=True` 获取注意力权重（Phase 2）

这就像：你先让一个人认真思考（Phase 1），记录下他的思考内容，然后让另一个人参考这些思考内容回答问题（Phase 2），同时记录他参考思考内容的程度。

### 1.4 关键 Token ID 对照

在 Qwen2.5-VL 的词汇表中，有一些特殊 token 有固定的 ID：

| Token 名称 | ID | 含义 |
|---|---|---|
| `<|image_pad|>` | 151655 | 图像占位 token |
| `<|vision_start|>` | 151652 | 图像区域开始标记 |
| `<|vision_end|>` | 151653 | 图像区域结束标记 |
| `<|vision_token|>` | 151654 | 视觉 token |
| `<abs_vis_token_pad>` | 151665 | Latent 占位 token |
| `<abs_vis_token>` | 151666 | Latent 区域开始标记 |
| `</abs_vis_token>` | 151667 | Latent 区域结束标记 |

> **为什么这些 ID 是固定的？** 因为它们是训练时预先定义的特殊 token，在 tokenizer 的词汇表中占有固定的位置。代码中硬编码这些 ID 是为了后续分析时能快速识别 token 类型，而不需要每次都通过 tokenizer 查询。

---

## 第二部分：整体架构与四步策略

### 2.1 整体流程图

```
┌─────────────────────────────────────────────────┐
│                   main() 函数                     │
│                                                   │
│  Step 1: 加载模型                                 │
│    └─ load_model() → model, processor             │
│                                                   │
│  Step 2: 加载数据集                               │
│    └─ 读取 JSON → 取前 2 个样本                    │
│                                                   │
│  Step 3: 对每个样本做注意力分析                    │
│    ┌─ 构建对话 (提取 user message)                │
│    ├─ prepare_input_for_sample()                  │
│    │   └ 在 assistant 后插入 latent pad tokens     │
│    ├─ Phase 1: latent_mode=True 前向传播           │
│    │   └ 获取 ce_patch_pos, ce_patch_vec           │
│    ├─ Phase 2a: latent_mode=False + ce_patch       │
│    │   └ 获取 KV cache                            │
│    ├─ Phase 2b: 逐步生成 answer tokens             │
│    │   └ 手动 greedy decode                       │
│    ├─ Phase 2c: 对完整序列 output_attentions=True  │
│    │   └ 提取注意力权重                            │
│    ├─ classify_token_positions()                  │
│    │   └ 将每个 token 分类                         │
│    └─ compute_attention_allocation()              │
│        └ 计算各类 token 的注意力占比                │
│                                                   │
│  Step 4: 生成热力图                               │
│    └ create_heatmaps()                            │
│    ├─ 单样本热力图                                │
│    ├─ 平均热力图                                  │
│    ├─ 注意力演化曲线                              │
│    └─ 统计 JSON                                  │
└─────────────────────────────────────────────────┘
```

### 2.2 四步策略详解（人话版）

| 步骤 | 人话描述 | 技术描述 |
|---|---|---|
| **Step 1** | 让模型准备好工作 | 加载 Monet 模型，设置 eager attention（因为要提取注意力权重），冻结视觉模块 |
| **Step 2** | 准备要分析的数据 | 从数据集加载样本，构建包含图像和文本的对话 |
| **Step 3** | 分析注意力（核心） | 见下方四阶段详解 |
| **Step 4** | 把分析结果画出来 | 生成热力图和统计文件 |

**Step 3 的四阶段详解**：

| 阶段 | 目的 | 关键参数 |
|---|---|---|
| **Phase 1** | 让模型做 latent 推理，获取真正的"思考内容" | `latent_mode=True` |
| **Phase 2a** | 把"思考内容"塞回去，获取 KV cache | `latent_mode=False` + `ce_patch_pos/ce_patch_vec` |
| **Phase 2b** | 让模型生成文字回答 | 手动 greedy decode，用 KV cache 加速 |
| **Phase 2c** | 对完整序列提取注意力 | `output_attentions=True`，构建 prompt+latent+answer 的完整序列 |

---

## 第三部分：逐函数详细讲解

### 3.1 文件头部：导入与常量定义

#### 代码（行 1-57）

```python
"""
Monet-SFT-7B 注意力分配分析脚本
...
"""
import os, sys, json, re, gc
import numpy as np
import torch
import PIL.Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from monet_qwen_model import apply_qwen2_5_monet
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.utils import add_latent_pad_after_auxiliary_img, replace_latent_placeholder_with_img_pad

LATENT_START_ID = 151666
LATENT_END_ID = 151667
LATENT_TOKEN_ID = 151665
IMAGE_TOKEN_ID = 151655
VISION_START_ID = 151652
VISION_END_ID = 151653
VISION_TOKEN_ID = 151654
```

#### 逐行解析

**文档字符串（行 1-29）**：

这是一个非常长的文档字符串，包含了：
- **脚本目的**：分析模型在 latent 推理后生成的文本 token 的注意力分配情况
- **策略说明**：四步策略的技术描述
- **人话解释**：用通俗易懂的语言解释了每一步的含义
- **运行方式**：两条可用的命令行

> **工程细节**：写详细文档字符串是一个好习惯。这个脚本特别值得注意的是，它同时提供了"技术描述"和"人话解释"。对于复杂的研究型脚本，这种双层次文档非常有价值 —— 技术人员看策略描述，非技术人员看人话解释。

**`matplotlib.use('Agg')`（行 40）**：

```python
matplotlib.use('Agg')
```

> **为什么这样写？** `Agg` 是 matplotlib 的非交互式后端（Anti-Grain Geometry），它不需要 GUI 界面，可以直接将图表保存为图片文件。在服务器环境中（没有显示器），如果使用默认的交互式后端，matplotlib 会报错。所以必须在 `import matplotlib.pyplot as plt` **之前** 设置后端。
>
> **常见错误**：如果先 `import plt` 再 `use('Agg')`，后端设置不会生效，因为 pyplot 已经初始化了默认后端。

**猴子补丁导入（行 45）**：

```python
from monet_qwen_model import apply_qwen2_5_monet
```

> **什么是猴子补丁（Monkey Patch）？** 这是一种在运行时动态替换模块的技术。`apply_qwen2_5_monet` 的作用是把 `transformers` 库中官方的 `Qwen2.5-VL` 模型代码替换为我们自己修改过的 Monet 版本（包含 latent_mode 等新功能）。
>
> **为什么不直接修改 transformers 源码？** 因为 transformers 是第三方库，直接修改的话：
> 1. 其他项目如果也依赖 transformers 就会受影响
> 2. transformers 更新后你的修改会被覆盖
> 3. 不方便版本管理
>
> 猴子补丁的方式可以在不修改原始代码的情况下，临时替换掉需要改变的部分。

**常量定义（行 51-57）**：

```python
LATENT_START_ID = 151666  # <abs_vis_token> 的 token ID
LATENT_END_ID = 151667   # </abs_vis_token> 的 token ID
LATENT_TOKEN_ID = 151665  # <abs_vis_token_pad> 的 token ID
IMAGE_TOKEN_ID = 151655   # <|image_pad|> 的 token ID
VISION_START_ID = 151652  # <|vision_start|> 的 token ID
...
```

> **为什么硬编码 ID 而不用 tokenizer 查询？**
> 1. **性能**：在 `classify_token_positions` 中需要逐个 token 比较，如果每次都调用 tokenizer 会非常慢
> 2. **一致性**：这些特殊 token 的 ID 在模型训练时就固定了，不会变化
> 3. **可读性**：用命名常量比数字更清晰，`LATENT_START_ID` 比 `151666` 更容易理解
>
> 但这种方式也有风险：如果换了不同版本的 tokenizer，这些 ID 可能变化。所以代码在 `load_model` 中也会通过 tokenizer 动态获取并设置到 model.config 中。

---

### 3.2 `replace_abs_vis_token_content` 函数

#### 代码（行 59-61）

```python
def replace_abs_vis_token_content(s):
    pattern = re.compile(r'(<abs_vis_token>)(.*?)(</abs_vis_token>)', flags=re.DOTALL)
    return pattern.sub(r'\1<latent>\3', s)
```

#### 解析

**功能**：将字符串中的 `<abs_vis_token>...内容...</abs_vis_token>` 替换为 `<abs_vis_token><latent></abs_vis_token>`。

**逐参数解析**：

- `r'(<abs_vis_token>)(.*?)(</abs_vis_token>)'`：正则表达式，三个捕获组
  - 第1组：`<abs_vis_token>`（开始标记）
  - 第2组：`.*?`（中间任意内容，`?` 表示非贪婪匹配）
  - 第3组：`</abs_vis_token>`（结束标记）
- `flags=re.DOTALL`：让 `.` 也匹配换行符（默认 `.` 不匹配 `\n`）
- `r'\1<latent>\3'`：替换字符串，`\1` 和 `\3` 分别引用第1组和第3组

**使用场景**：在生成 answer 后，解码的文本可能包含 `<abs_vis_token>一些latent内容</abs_vis_token>`，这个函数把它替换成 `<abs_vis_token><latent></abs_vis_token>`，让输出更简洁。

> **为什么用非贪婪匹配 `.*?` 而不是 `.*` ？** 如果用贪婪匹配，当字符串中有多个 `<abs_vis_token>...</abs_vis_token>` 对时，`.*` 会从第一个开始标记一直匹配到最后一个结束标记，把中间所有内容（包括其他正常的对）都吃掉。非贪婪匹配确保每次只匹配一对标记。

---

### 3.3 `load_model` 函数 —— 模型加载

#### 代码（行 64-110）

```python
def load_model(model_path, device="cuda:0", latent_size=10):
    print("=" * 60)
    print("Step 1: 加载 Monet 模型...")
    print("=" * 60)
    
    config = Qwen2_5_VLConfig.from_pretrained(model_path)
    config.text_config._attn_implementation = "eager"  # 只有 LLM 用 eager
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, config=config, torch_dtype=torch.bfloat16,
    )
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.add_tokens("<abs_vis_token_pad>", special_tokens=True)
    processor.tokenizer.add_tokens("<abs_vis_token>", special_tokens=True)
    processor.tokenizer.add_tokens("</abs_vis_token>", special_tokens=True)
    processor.tokenizer.add_tokens("<observation>", special_tokens=True)
    processor.tokenizer.add_tokens("</observation>", special_tokens=True)
    
    # Resize embeddings
    new_vocab_size = len(processor.tokenizer)
    model.resize_token_embeddings(new_vocab_size)
    model.config.vocab_size = new_vocab_size
    
    # 设置 latent token IDs
    latent_start_idx = processor.tokenizer("<abs_vis_token>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("</abs_vis_token>", return_tensors="pt")["input_ids"][0]
    latent_pad_idx = processor.tokenizer("<abs_vis_token_pad>", return_tensors="pt")["input_ids"][0]
    answer_start_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
    
    model.config.latent_token_id = int(latent_pad_idx)
    model.config.latent_start_id = int(latent_start_idx)
    model.config.latent_end_id = int(latent_end_idx)
    model.config.answer_start_pattern = answer_start_pattern.tolist()
    
    # Freeze visual
    for p in model.visual.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    
    return model, processor
```

#### 逐段解析

**为什么设置 `_attn_implementation = "eager"`？**

```python
config.text_config._attn_implementation = "eager"
```

Transformer 模型有多种注意力实现方式：
- **`eager`**：最基础的实现，显式计算 Q×K^T 再 softmax，**会返回完整的注意力权重矩阵**（shape: `[batch, heads, seq_len, seq_len]`）
- **`flash_attention`**：使用 Flash Attention 算法，速度快、省内存，但**不返回注意力权重矩阵**（因为它是一种近似算法，中间结果不保存）
- **`sdpa`**：PyTorch 的 scaled dot product attention，介于两者之间

> **工程决策**：我们需要分析注意力权重，所以必须用 `eager` 实现。但 `eager` 的缺点是计算速度慢、内存占用大（因为要存储完整的注意力矩阵）。所以这里只让 LLM（语言模型部分）用 `eager`，视觉部分仍然可以用 flash attention。

**为什么用 `torch.bfloat16`？**

```python
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, config=config, torch_dtype=torch.bfloat16,
)
```

- `bfloat16`（Brain Float 16）是一种 16 位浮点格式，相比 `float16`：
  - 动态范围更大（8位指数 vs 5位指数），不容易溢出
  - 精度稍低（7位尾数 vs 10位尾数）
  - 在现代 GPU（A100、H100 等）上计算效率高
> 这几乎是现代大模型推理的标配选择。

**为什么要添加新 token 并 resize embeddings？**

```python
processor.tokenizer.add_tokens("<abs_vis_token_pad>", special_tokens=True)
...
new_vocab_size = len(processor.tokenizer)
model.resize_token_embeddings(new_vocab_size)
```

Monet 模型在 Qwen2.5-VL 的基础上新增了 5 个特殊 token。添加新 token 的步骤：
1. `add_tokens`：在 tokenizer 中注册新 token，词汇表扩大
2. `resize_token_embeddings`：模型的 embedding 层也要扩大，新增的 embedding 向量会被随机初始化

> **为什么 `add_tokens` 要设 `special_tokens=True`？** 特殊 token 在分词时不会被拆分。比如如果 `<abs_vis_token>` 不是特殊 token，tokenizer 可能把它拆成 `<`, `abs`, `_`, `vis`, `_`, `token`, `>` 等多个子 token，那就不是我们想要的效果了。

**为什么获取 `answer_start_pattern`？**

```python
answer_start_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
```

`<|im_start|>assistant` 在 tokenizer 中会被编码成多个 token（不是一个！），因为 `<|im_start|>` 是一个特殊 token，`assistant` 是普通文本。这个 pattern 用于在 `latent_mode=True` 的模型 forward 中定位"回答开始的位置"，模型需要知道从哪里开始做 latent 推理。

**为什么冻结视觉模块？**

```python
for p in model.visual.parameters():
    p.requires_grad = False
model.eval()
```

- `requires_grad = False`：告诉 PyTorch 不要为视觉模块计算梯度，节省内存
- `model.eval()`：切换到评估模式，关闭 dropout 等训练时才用的功能

> 我们只是做分析，不需要训练模型，所以冻结视觉模块是合理的内存优化。

---

### 3.4 `prepare_input_for_sample` 函数 —— 输入准备

#### 代码（行 113-177）

```python
def prepare_input_for_sample(conversation, processor, latent_size, device):
    prompt_text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
    )
    
    prompt_text = replace_latent_placeholder_with_img_pad(prompt_text)
    
    sep_token = "<|im_start|>assistant"
    latent_pad_str = "<abs_vis_token_pad>"
    latent_pad_strs = latent_pad_str * latent_size
    
    prompt_text_with_latent = prompt_text.replace(
        sep_token,
        f"{sep_token}<abs_vis_token>{latent_pad_strs}</abs_vis_token>"
    )
    
    image_inputs, _ = process_vision_info(conversation, return_video_kwargs=False)
    
    inputs = processor(
        text=[prompt_text_with_latent],
        images=image_inputs,
        return_tensors="pt",
        padding=True,
        min_pixels=256 * 28 * 28,
        max_pixels=8192 * 28 * 28,
    )
    
    model_inputs = {
        'input_ids': inputs.input_ids.to(device),
        'attention_mask': inputs.attention_mask.to(device),
        'pixel_values': inputs.pixel_values.to(device) if inputs.pixel_values is not None else None,
        'image_grid_thw': inputs.image_grid_thw.to(device) if inputs.image_grid_thw is not None else None,
    }
    
    prompt_len = inputs.input_ids.shape[1]
    
    return model_inputs, prompt_len, image_inputs
```

#### 逐段解析

**`apply_chat_template` 的作用**：

```python
prompt_text = processor.apply_chat_template(
    conversation, tokenize=False, add_generation_prompt=True,
)
```

这会把对话格式化为模型期望的输入格式。例如：

```
输入 conversation:
[{"role": "user", "content": [{"type": "image", "image": ...}, {"type": "text", "text": "描述这个图片"}]}]

输出 prompt_text:
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>描述这个图片<|im_end|>
<|im_start|>assistant
```

> **`add_generation_prompt=True`**：确保末尾有 `<|im_start|>assistant`，这样模型就知道接下来该生成回答了。

**为什么调用 `replace_latent_placeholder_with_img_pad`？**

```python
prompt_text = replace_latent_placeholder_with_img_pad(prompt_text)
```

查看 `src/utils.py` 中的实现：

```python
def replace_latent_placeholder_with_img_pad(text, ...):
    text = text.split(sep_token)  # 按 <|im_start|>assistant 分割
    res_text = process_multiple_question_img(text[0])  # 处理用户区域
    assistant_texts = text[1:]
    for text in assistant_texts:
        if latent_placeholder in text:
            text = text.replace(image_pad, "")
            text = text.replace(latent_placeholder, image_pad)
        res_text += sep_token + text
    return res_text
```

这个函数的逻辑是：如果 assistant 区域有 `<abs_vis_token></abs_vis_token>`（训练时才会出现），就把它替换为 `<|vision_start|><|image_pad|><|vision_end|>`。但在推理时，assistant 区域没有这个标记，所以这个调用实际上**不会改变任何内容**。

> **为什么还要调用它？** 这是一个防御性编程策略。即使当前场景不需要替换，调用这个函数可以确保代码的通用性 —— 如果将来输入格式发生变化（比如 assistant 区域确实有 latent placeholder），代码仍然能正确处理。

**在 assistant 后插入 latent tokens**：

```python
sep_token = "<|im_start|>assistant"
latent_pad_str = "<abs_vis_token_pad>"
latent_pad_strs = latent_pad_str * latent_size

prompt_text_with_latent = prompt_text.replace(
    sep_token,
    f"{sep_token}<abs_vis_token>{latent_pad_strs}</abs_vis_token>"
)
```

这是整个输入准备中最关键的一步！

**效果示例**：
```
原始: <|im_start|>assistant
替换后: <|im_start|>assistant<abs_vis_token><abs_vis_token_pad><abs_vis_token_pad>...(10个)<abs_vis_token_pad></abs_vis_token>
```

> **为什么这样插入？**
> - `<abs_vis_token>` 和 `</abs_vis_token>` 是 latent 区域的边界标记，告诉模型"这是一个隐式思考区域"
> - `<abs_vis_token_pad>` 是占位 token，在 `latent_mode=True` 时会被模型内部替换为真正的 latent embedding
> - `latent_size=10` 表示模型有 10 步隐式思考
> - 插入位置在 `<|im_start|>assistant` 之后，因为 latent 推理发生在模型开始回答之前

**`latent_pad_str * latent_size` 的字符串乘法**：

```python
latent_pad_strs = latent_pad_str * latent_size
# 结果: "<abs_vis_token_pad><abs_vis_token_pad><abs_vis_token_pad>...(10次)"
```

Python 的字符串乘法 `str * n` 会重复字符串 n 次。这是生成重复占位 token 的简洁方式。

**`process_vision_info` 的作用**：

```python
image_inputs, _ = process_vision_info(conversation, return_video_kwargs=False)
```

这个函数从对话中提取 PIL Image 对象。Qwen2.5-VL 的 processor 需要原始的 PIL Image 来处理图像输入。

**processor 处理**：

```python
inputs = processor(
    text=[prompt_text_with_latent],
    images=image_inputs,
    return_tensors="pt",
    padding=True,
    min_pixels=256 * 28 * 28,
    max_pixels=8192 * 28 * 28,
)
```

- `text`：包含 latent tokens 的完整 prompt
- `images`：PIL Image 对象列表
- `return_tensors="pt"`：返回 PyTorch tensor
- `min_pixels / max_pixels`：控制图像分辨率范围
  - Qwen2.5-VL 使用 ViT 处理图像，每个 patch 是 28×28 像素
  - `256 * 28 * 28 = 200704` 像素 = 约 448×448 的图像
  - `8192 * 28 * 28 = 6422528` 像素 = 约 2528×2528 的图像

> **为什么用 `min_pixels` 和 `max_pixels`？** 太小的图像会让 ViT 提取不到足够的细节，太大的图像会占用过多 token（每 28×28 patch 对应一个 token）。设置范围是速度和质量的平衡。

**构建 model_inputs 字典**：

```python
model_inputs = {
    'input_ids': inputs.input_ids.to(device),
    'attention_mask': inputs.attention_mask.to(device),
    'pixel_values': inputs.pixel_values.to(device) if inputs.pixel_values is not None else None,
    'image_grid_thw': inputs.image_grid_thw.to(device) if inputs.pixel_values is not None else None,
}
```

> **为什么要手动构建字典而不是直接用 `inputs`？**
> 1. 确保所有 tensor 都在正确的设备（GPU）上
> 2. 处理 None 值（如果输入没有图像，`pixel_values` 和 `image_grid_thw` 就是 None）
> 3. 只提取模型 forward 需要的字段，避免多余数据占用内存

> **`image_grid_thw` 是什么？** 它是每个图像在 LLM 中的 (temporal, height, width) 信息，用于 3D 旋转位置编码（RoPE）。比如一张 448×448 的图像，经过 ViT 处理后变成 `grid_thw = [1, 16, 16]`（1帧，16×16 个 patch）。

---

### 3.5 `classify_token_positions` 函数 —— Token 分类

#### 代码（行 180-234）

```python
def classify_token_positions(token_ids):
    categories = {
        'latent_tokens': [],
        'latent_boundary': [],
        'image_tokens': [],
        'system_tokens': [],
        'question_tokens': [],
        'answer_tokens': [],
    }
    
    latent_ranges = []
    current_start = None
    
    for i, tid in enumerate(token_ids):
        if tid == LATENT_START_ID:
            current_start = i
            categories['latent_boundary'].append(i)
        elif tid == LATENT_END_ID:
            if current_start is not None:
                latent_ranges.append((current_start, i))
                current_start = None
            categories['latent_boundary'].append(i)
        elif tid in [IMAGE_TOKEN_ID, VISION_START_ID, VISION_END_ID, VISION_TOKEN_ID]:
            categories['image_tokens'].append(i)
    
    for i, tid in enumerate(token_ids):
        if tid == LATENT_TOKEN_ID:
            for start, end in latent_ranges:
                if start < i < end:
                    categories['latent_tokens'].append(i)
                    break
    
    last_latent_end = max(categories['latent_boundary']) if categories['latent_boundary'] else None
    if last_latent_end is not None:
        for i in range(last_latent_end + 1, len(token_ids)):
            if i not in categories['latent_boundary'] and \
               i not in categories['latent_tokens'] and \
               i not in categories['image_tokens']:
                categories['answer_tokens'].append(i)
    
    first_latent_start = min(categories['latent_boundary']) if categories['latent_boundary'] else len(token_ids)
    first_image_pos = min(categories['image_tokens']) if categories['image_tokens'] else first_latent_start
    
    for i in range(0, first_latent_start):
        if i not in categories['latent_boundary'] and \
           i not in categories['latent_tokens'] and \
           i not in categories['image_tokens']:
            if i < first_image_pos:
                categories['system_tokens'].append(i)
            else:
                categories['question_tokens'].append(i)
    
    return categories
```

#### 解析

**功能**：将 token 序列中的每个位置分类到不同的语义类别。

**为什么需要分类？** 后续计算注意力分配时，我们需要知道"每个 answer token 把注意力分配给了哪类 token"。如果不分类，我们只能看到"token 5 对 token 3 的注意力是 0.02"，但不能说"token 5 对 latent tokens 的总注意力是多少"。

**分类逻辑的逐步拆解**：

**第1步：扫描 latent 区域边界和图像 token**

```python
for i, tid in enumerate(token_ids):
    if tid == LATENT_START_ID:    # 遇到 <abs_vis_token>
        current_start = i
        categories['latent_boundary'].append(i)
    elif tid == LATENT_END_ID:    # 遇到 </abs_vis_token>
        if current_start is not None:
            latent_ranges.append((current_start, i))  # 记录一个完整的 latent 区域
            current_start = None
        categories['latent_boundary'].append(i)
    elif tid in [IMAGE_TOKEN_ID, ...]:  # 遇到图像相关 token
        categories['image_tokens'].append(i)
```

这会记录所有 latent 区域的范围（如 `(5, 16)` 表示位置 5 到 16 是一个 latent 区域），以及图像 token 的位置。

**第2步：识别 latent 内部的占位 token**

```python
for i, tid in enumerate(token_ids):
    if tid == LATENT_TOKEN_ID:  # <abs_vis_token_pad>
        for start, end in latent_ranges:
            if start < i < end:  # 严格在 latent 区域内部
                categories['latent_tokens'].append(i)
                break
```

> **为什么用 `start < i < end` 而不是 `start <= i <= end`？** 因为 `start` 和 `end` 位置是边界标记 `<abs_vis_token>` 和 `</abs_vis_token>`，它们本身不属于 latent 内容 token，只是标记。所以用严格小于。

**第3步：标记 answer tokens（latent 区域之后的所有非特殊 token）**

```python
last_latent_end = max(categories['latent_boundary'])
for i in range(last_latent_end + 1, len(token_ids)):
    if i not in categories['latent_boundary'] and \
       i not in categories['latent_tokens'] and \
       i not in categories['image_tokens']:
        categories['answer_tokens'].append(i)
```

> **为什么只从 `last_latent_end + 1` 开始？** 因为 answer tokens 是 `</abs_vis_token>` 之后的所有文本 token。在 Monet 的设计中，模型先做隐式思考（latent 区域），然后生成文字回答（answer 区域）。

**第4步：标记 system 和 question tokens（latent 区域之前的 token）**

```python
first_latent_start = min(categories['latent_boundary'])
first_image_pos = min(categories['image_tokens'])

for i in range(0, first_latent_start):
    if i not in ...:
        if i < first_image_pos:
            categories['system_tokens'].append(i)   # 系统提示词
        else:
            categories['question_tokens'].append(i)   # 用户问题（含图像）
```

> **分类策略**：第一个图像 token 之前的是系统提示词（如 `<|im_start|>system You are...`），之后的是用户问题（包含图像和文本）。这个分界点是基于 Qwen2.5-VL 的对话格式设计的。

**最终分类示意**：

```
位置: 0  1  2  3  4  5  6  7  ...  16  17  18  19  20 ...
Token: sys sys img img img img <abs> pad pad ... </abs> ans ans ans ...
分类: system system image image image image boundary latent latent ... boundary answer answer answer ...
```

---

### 3.6 `compute_attention_allocation` 函数 —— 注意力分配计算

#### 代码（行 237-272）

```python
def compute_attention_allocation(all_attentions, categories, token_ids):
    answer_positions = categories['answer_tokens']
    if len(answer_positions) == 0:
        return None
    
    focus_categories = [
        ('latent_tokens', 'Latent CoT'),
        ('latent_boundary', 'Latent Boundary'),
        ('image_tokens', 'Image'),
        ('system_tokens', 'System Prompt'),
        ('question_tokens', 'Question Text'),
        ('answer_tokens', 'Self (Answer)'),
    ]
    
    # 取最后一层的注意力权重
    last_layer_attn = all_attentions[-1]
    avg_attn = last_layer_attn[0].mean(dim=0).cpu().float().numpy()
    
    seq_len = len(token_ids)
    attention_matrix = np.zeros((len(answer_positions), len(focus_categories)))
    
    for i, ans_pos in enumerate(answer_positions):
        if ans_pos >= seq_len:
            continue
        attn_weights = avg_attn[ans_pos, :seq_len]
        
        for j, (cat_key, cat_name) in enumerate(focus_categories):
            cat_positions = [p for p in categories[cat_key] if p < seq_len]
            if len(cat_positions) > 0:
                cat_attn = attn_weights[cat_positions].sum()
                total_attn = attn_weights.sum()
                attention_matrix[i, j] = (cat_attn / total_attn * 100) if total_attn > 0 else 0
    
    return attention_matrix, focus_categories, answer_positions
```

#### 逐段解析

**为什么只取最后一层的注意力？**

```python
last_layer_attn = all_attentions[-1]
```

Transformer 有多层（比如 28 层），每层都有自己的注意力。最后一层的注意力最接近模型的最终决策，因为经过多层处理后，信息已经被充分整合。

> **工程考量**：如果取所有层的平均，低层的注意力可能更多是"语法级"的关注（比如关注相邻 token），而高层的注意力才是"语义级"的关注。我们关心的是模型在语义层面是否参考了 latent tokens，所以取最后一层。

**为什么对多头注意力取平均？**

```python
avg_attn = last_layer_attn[0].mean(dim=0).cpu().float().numpy()
```

`last_layer_attn[0]` 的 shape 是 `[num_heads, seq_len, seq_len]`，`mean(dim=0)` 在 head 维度上取平均，得到 `[seq_len, seq_len]`。

> **为什么不单独看每个 head？** 每个注意力头可能关注不同的模式（有的关注语法，有的关注语义），但我们要看的是整体趋势。取平均可以给出一个宏观的注意力分配视图。当然，如果想深入分析，可以分别看每个 head —— 但对于初版分析脚本，取平均是更合理的简化。

**`.cpu().float().numpy()` 的转换链**：

> **为什么这样转换？**
> - `.cpu()`：从 GPU 移到 CPU，因为 numpy 不支持 GPU tensor
> - `.float()`：从 bfloat16 转为 float32，numpy 不支持 bfloat16
> - `.numpy()`：转为 numpy 数组，方便后续用 matplotlib 绘图

**注意力分配比例的计算**：

```python
for j, (cat_key, cat_name) in enumerate(focus_categories):
    cat_positions = [p for p in categories[cat_key] if p < seq_len]
    if len(cat_positions) > 0:
        cat_attn = attn_weights[cat_positions].sum()  # 该类别所有 token 的注意力之和
        total_attn = attn_weights.sum()  # 总注意力
        attention_matrix[i, j] = (cat_attn / total_attn * 100)  # 百分比
```

> **数学含义**：对于 answer token 在位置 `ans_pos`，它的注意力权重向量是 `attn_weights[ans_pos, :]`（长度 = seq_len）。这个向量经过 softmax，总和 = 1。我们计算：
> 
> \[
> \text{Latent CoT 占比} = \frac{\sum_{p \in \text{latent\_tokens}} \text{attn}[p]}{\sum_{p=0}^{\text{seq\_len}-1} \text{attn}[p]} \times 100\%
> \]
> 
> 这个比值告诉我们：模型在生成这个 answer token 时，有多少"思考精力"花在了 latent tokens 上。

> **为什么用 `attn_weights[cat_positions]` 而不是逐个累加？** NumPy 的索引可以用列表直接选取多个位置，`attn_weights[cat_positions]` 会一次性取出所有 latent token 位置的注意力值，然后 `.sum()` 求和。这比写循环逐个加要高效得多。

---

### 3.7 `create_heatmaps` 函数 —— 热力图生成

#### 代码（行 275-382）

这个函数是整个脚本中代码量最大的部分（约 107 行），负责生成三种可视化：
1. 单样本热力图
2. 平均热力图
3. 注意力演化曲线 + 统计 JSON

**单样本热力图（行 279-304）**：

```python
for i, m in enumerate(all_matrices):
    if m is None:
        continue
    max_show = min(m.shape[0], 80)  #最多显示 80 个 token
    matrix_show = m[:max_show]
    
    fig, ax = plt.subplots(figsize=(max_show * 0.15 + 3, len(focus_categories) * 0.6 + 2))
    im = ax.imshow(matrix_show, cmap=plt.cm.YlOrRd, aspect='auto', vmin=0, vmax=100)
    ...
```

> **为什么限制最多 80 个 token？**
> 1. 如果 answer 有 200+ 个 token，热力图会太宽，显示效果差
> 2. `imshow` 的图像大小受 figsize 控制，太宽会导致标签重叠
> 3. 通常前 80 个 token 就能看到注意力分配的趋势

> **为什么热力图的横轴是 answer token 位置、纵轴是 token 类别？** 因为我们要看的是"每个 answer token 对各类 token 的注意力分配"。这就像一个表格：行是类别，列是 answer token 位置，单元格的值是注意力占比。用热力图可以直观地看到哪些类别在哪些位置获得了高注意力。

> **`YlOrRd` colormap**：黄-橙-红渐变色。0% 是浅黄色，100% 是深红色。这是一种常见的"热力"配色方案，视觉上很容易区分高低值。

**平均热力图（行 306-341）**：

```python
trimmed = []
for m in all_matrices:
    if m is not None and m.shape[0] > 0:
        t = m[:max_show]
        if t.shape[0] < max_show:
            p = np.full((max_show, t.shape[1]), np.nan)  # 用 NaN 填充
            p[:t.shape[0]] = t
            trimmed.append(p)
        else:
            trimmed.append(t)

avg = np.nanmean(np.stack(trimmed), axis=0)
```

> **为什么要用 NaN 填充？** 不同样本的 answer token 数量不同（比如样本1有50个，样本2有80个）。为了计算平均值，需要把它们对齐到相同的长度。用 NaN 填充短样本的缺失位置，然后 `np.nanmean` 在计算平均时会忽略 NaN 值 —— 这比用 0 填充更准确（0 会拉低平均值）。

> **为什么用 `np.nanmean` 而不是 `np.mean`？** `np.mean` 会把 NaN 也算进去，结果会是 NaN。`np.nanmean` 会跳过 NaN 值，只对有效数据计算平均。

**注意力演化曲线（行 343-369）**：

```python
latent_curves = [m[:max_show, 0] for m in all_matrices if m is not None and m.shape[0] > 0]
max_len = max(len(c) for c in latent_curves)
stacked = np.full((len(latent_curves), max_len), np.nan)
for j, c in enumerate(latent_curves):
    stacked[j, :len(c)] = c

avg_curve = np.nanmean(stacked, axis=0)
std_curve = np.nanstd(stacked, axis=0)

ax.plot(pos, avg_curve, 'b-', linewidth=2, label='Mean Latent Attention %')
ax.fill_between(pos, np.nan_to_num(avg_curve - std_curve, nan=0),
                np.nan_to_num(avg_curve + std_curve, nan=100),
                alpha=0.3, color='blue', label='±1 Std Dev')
ax.axhline(y=avg_curve.mean(), color='g', linestyle='-', alpha=0.7,
           label=f'Overall mean = {avg_curve.mean():.1f}%')
```

> **这条曲线展示什么？** 横轴是 answer token 的位置（从第1个到第80个），纵轴是该位置对 latent tokens 的注意力占比。这让我们能看到：随着生成进行，模型对 latent tokens 的关注度是如何变化的。
>
> 比如，如果曲线逐渐下降，说明模型一开始很依赖 latent 思考，但随着生成进行，越来越依赖自己已经生成的文字。如果曲线保持平稳，说明模型始终在参考 latent 思考。

> **`fill_between` 是什么？** 它画出平均值上下一个标准差的区域（蓝色阴影），表示数据的离散程度。如果阴影很窄，说明不同样本的行为一致；如果很宽，说明样本间差异大。

> **为什么 `np.nan_to_num(avg_curve - std_curve, nan=0)`？** 下界可能因为 NaN 而变成 NaN，`nan_to_num` 把 NaN 替换为 0（注意力占比不能低于0%）。同理，上界的 NaN 替换为 100（注意力占比不能超过100%）。

**统计 JSON（行 371-382）**：

```python
all_vals = np.concatenate([m for m in all_matrices if m is not None], axis=0)
stats = {}
for j, (ck, cn) in enumerate(focus_categories):
    stats[cn] = {'mean': float(all_vals[:, j].mean()), 'std': float(all_vals[:, j].std())}
stats['latent_detail'] = {
    'mean_pct': float(all_vals[:, 0].mean()),
    'pct_above_10': float(np.mean(all_vals[:, 0] > 10) * 100),
    'pct_above_30': float(np.mean(all_vals[:, 0] > 30) * 100),
}
```

> **为什么统计 `pct_above_10` 和 `pct_above_30`？** 这两个指标回答了关键问题：
> - 有多少比例的 answer token 把超过 10% 的注意力分配给了 latent tokens？（说明至少有一定的参考）
> - 有多少比例的 answer token 把超过 30% 的注意力分配给了 latent tokens？（说明有显著的参考）
>
> 如果 `pct_above_30` 很低（比如只有 5%），那说明 latent CoT 的作用可能不大。如果很高（比如 60%），那说明模型确实在大量参考隐式思考。

---

### 3.8 `main` 函数 —— 主流程

#### 代码（行 385-708）

这是整个脚本的核心，约 320 行代码。按功能分阶段讲解。

**配置与初始化（行 385-404）**：

```python
model_path = os.environ.get("MONET_MODEL_PATH", "/home/xiaojunhao/m-x/data/Monet-SFT-7B/stage3")
dataset_path = "/home/xiaojunhao/m-x/data/Monet-SFT-125K/Zebra_CoT_geometry/train.json"
base_image_dir = "/home/xiaojunhao/m-x/data/Monet-SFT-125K"
output_dir = "/home/xiaojunhao/m-x/inference/attention_analysis_results"
latent_size = int(os.environ.get("LATENT_SIZE", "10"))

num_samples = 2
device = "cuda:0" if torch.cuda.is_available() else "cpu"
```

> **为什么用环境变量？** `os.environ.get("MONET_MODEL_PATH", ...)` 允许通过环境变量覆盖默认值。这样：
> 1. 不需要修改代码就能切换模型路径
> 2. 在不同机器上运行时，只需设置环境变量
> 3. Shell 脚本中可以灵活配置
>
> `latent_size` 也用环境变量，因为不同训练阶段可能使用不同的 latent token 数量。

> **为什么只取 2 个样本？** 因为注意力分析非常消耗资源（`output_attentions=True` 会存储完整的注意力矩阵，28层 × 每层 `[1, heads, seq_len, seq_len]`）。2 个样本足以验证代码正确性，如果要做大规模分析，可以增加 `num_samples`。

**构建对话（行 429-439）**：

```python
user_msg = next(msg for msg in item["data"] if msg["role"] == "user")
conv_content = []
for block in user_msg["content"]:
    if block["type"] == "image":
        img_path = os.path.join(base_image_dir, block["image"])
        conv_content.append({"type": "image", "image": PIL.Image.open(img_path).convert("RGB")})
    elif block["type"] == "text":
        conv_content.append(block)

conversation = [{"role": "user", "content": conv_content}]
```

> **为什么只取 user message？** 注意力分析需要看模型生成 answer 时对 latent tokens 的注意力。如果包含 assistant 的回答，模型就会"看到"答案，这不是我们想要的。我们只需要用户的输入（图像 + 问题），让模型自己生成回答。

> **为什么用 `.convert("RGB")`？** 有些图像可能是 RGBA（含透明通道）或灰度图。Qwen2.5-VL 的 ViT 只接受 RGB 格式，所以统一转换。

**Phase 1：latent_mode=True 前向传播（行 446-490）**：

```python
with torch.inference_mode():
    latent_outputs = model(
        **model_inputs,
        latent_mode=True,
        output_latent_embeds=False,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )

ce_patch_pos = latent_outputs.ce_patch_pos
ce_patch_vec = latent_outputs.ce_patch_vec
```

> **`torch.inference_mode()` vs `torch.no_grad()`？** 两者都是禁用梯度计算，但 `inference_mode` 更彻底：
> - `no_grad()` 只禁用梯度计算，但仍然追踪 tensor 的版本信息（用于 autograd）
> - `inference_mode()` 禁用所有 autograd 相关的功能，包括版本追踪
> - `inference_mode()` 更快、更省内存
>
> 在纯推理场景下，`inference_mode()` 是更好的选择。

> **`ce_patch_pos` 和 `ce_patch_vec` 是什么？** 这是 Phase 1 最重要的输出。
> - `ce_patch_pos`：latent tokens 在序列中的位置列表，如 `[[6, 7, 8, ..., 15]]`（batch_size=1）
> - `ce_patch_vec`：每个 latent token 位置对应的真正 embedding 向量，shape 如 `(10, 3584)`（10个latent token，每个3584维）
>
> 在模型的 latent_mode forward 中，每个 latent token 的 embedding 是前一个 token 的隐藏状态：
> ```python
> # 模型内部代码（modeling_qwen2_5_vl_monet.py 行 914-918）
> prev_hidden = batch_last_hidden_state[b, pos - 1, :]
> latent_embed = prev_hidden.clone() if self.training else prev_hidden.detach()
> ce_patch_pos[b].append(pos)
> ce_patch_vec[b].append(latent_embed[0, 0])
> ```
> 这就是"隐式思维链"的实现：每一步思考的输入是上一步的输出。

> **为什么 `output_latent_embeds=False`？** 这个参数控制是否返回 latent embeddings。在这个脚本中我们不需要完整的 latent embeddings（只需要 ce_patch_pos 和 ce_patch_vec），所以设为 False 节省内存。

**Phase 2a：获取 KV cache（行 493-512）**：

```python
with torch.inference_mode():
    phase2a_outputs = model(
        **model_inputs,
        latent_mode=False,
        ce_patch_pos=ce_patch_pos,
        ce_patch_vec=ce_patch_vec,
        output_attentions=False,
        use_cache=True,
        return_dict=True,
    )

past_kv = phase2a_outputs.past_key_values
```

> **为什么需要 KV cache？** Phase 2b 需要逐步生成 token，每次只传一个新 token。KV cache 存储了之前所有 token 的 Key 和 Value，这样每一步只需要计算新 token 的 Query，然后和缓存的 Key/Value 做注意力计算 —— 大大节省计算量。

> **`ce_patch_pos` 和 `ce_patch_vec` 在 latent_mode=False 时如何工作？** 查看模型内部代码（行 2006-2013）：
> ```python
> if ce_patch_pos is not None and ce_patch_vec is not None:
>     for b in range(len(ce_patch_pos)):
>         pos_list = ce_patch_pos[b]
>         vecs = ce_patch_vec[b].to(inputs_embeds.device, inputs_embeds.dtype)
>         inputs_embeds[b, torch.tensor(pos_list), :] = vecs
> ```
> 它把 Phase 1 得到的真正 latent embeddings 直接替换到 inputs_embeds 中对应的位置！这就是"把思考内容塞回去"的实现。

> **为什么不直接在 Phase 2a 就开启 `output_attentions=True`？** 因为 Phase 2a 只处理 prompt 部分（包含 latent tokens），还没有 answer tokens。注意力分析需要看 answer tokens 对 latent tokens 的注意力，所以必须先生成 answer，再对完整序列做注意力提取。

**Phase 2b：逐步生成 answer tokens（行 514-580）**：

```python
max_new_tokens = 256
generated_tokens = []

next_input_ids = model_inputs['input_ids'][:, -1:]  # 最后一个 token
full_attn_mask = model_inputs['attention_mask'].clone()
past_kv_for_gen = past_kv

with torch.inference_mode():
    for step in range(max_new_tokens):
        model_out = model.model(  # 注意：用的是 model.model，不是 model
            input_ids=next_input_ids,
            attention_mask=full_attn_mask,
            past_key_values=past_kv_for_gen,
            pixel_values=None,
            image_grid_thw=None,
            latent_mode=False,
            use_cache=True,
            ce_patch_pos=None,
            ce_patch_vec=None,
            return_dict=True,
        )
        
        hidden_states = model_out.last_hidden_state
        logits = model.lm_head(hidden_states)
        next_token_logits = logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1)
        
        if next_token.item() == processor.tokenizer.eos_token_id:
            break
        
        generated_tokens.append(next_token.item())
        
        next_input_ids = next_token.unsqueeze(0)
        full_attn_mask = torch.cat([
            full_attn_mask,
            torch.ones(1, 1, dtype=torch.long, device=device)
        ], dim=1)
        past_kv_for_gen = model_out.past_key_values
        
        if step % 50 == 0 and step > 0:
            decoded = processor.tokenizer.decode(generated_tokens[-10:], skip_special_tokens=True)
            print(f"    Step {step}: {len(generated_tokens)} tokens, recent: {decoded}")
```

> **为什么用 `model.model()` 而不是 `model()`？** 这是本脚本中一个关键的工程细节。
>
> `model()` 是 `Qwen2_5_VLForConditionalGeneration.forward()`，它内部会：
> 1. 调用 `self.model()`（即 `Qwen2_5_VLModel.forward()`）获取 hidden states
> 2. 调用 `self.lm_head(hidden_states)` 计算 logits
> 3. 如果有 labels，计算 loss
>
> 但在 `latent_mode=False` 且 `loss_type=[]`（空列表）的情况下，模型的 forward 函数会**不计算 logits**（因为不需要 loss），导致 `logits=None`。
>
> 所以这里拆成两步：
> 1. `model.model()` 获取 hidden states（绕过外层 forward 的 logits 计算逻辑）
> 2. `model.lm_head(hidden_states)` 手动计算 logits
>
> 这样就避免了 `logits=None` 的问题。

> **为什么 `pixel_values=None` 和 `image_grid_thw=None`？** 在 KV cache 模式下，图像的 KV 已经在 Phase 2a 中计算好了，缓存在 `past_key_values` 中。后续每步只需要传新 token 的 input_ids，不需要再传图像数据。

> **为什么 `ce_patch_pos=None` 和 `ce_patch_vec=None`？** 同理，latent embeddings 的替换已经在 Phase 2a 中完成了，它们的 KV 也已经缓存。

> **greedy decode 是什么？**
> ```python
> next_token = next_token_logits.argmax(dim=-1)
> ```
> `argmax` 选择概率最高的 token，这就是 greedy decoding（贪心解码）。它不考虑概率分布的多样性，总是选最确定的 token。
>
> 在注意力分析中，我们不需要多样化的输出，只需要一个确定性的回答来分析其注意力模式。所以 greedy decode 是合适的。

> **attention_mask 为什么要逐步扩展？**
> ```python
> full_attn_mask = torch.cat([
>     full_attn_mask,
>     torch.ones(1, 1, dtype=torch.long, device=device)
> ], dim=1)
> ```
> 每生成一个新 token，attention_mask 需要加一个 `1`，表示这个新位置是有效的（不是 padding）。如果不更新 attention_mask，模型在计算注意力时会把新生成的 token 当作 padding 而忽略。

**Phase 2c：提取注意力权重（行 582-690）**：

这是最复杂也最关键的阶段。

```python
# 构建完整序列 = prompt + latent + answer
full_input_ids = torch.cat([
    model_inputs['input_ids'],
    torch.tensor([generated_tokens], dtype=torch.long, device=device)
], dim=1)

full_attention_mask = torch.ones(1, full_input_ids.shape[1], dtype=torch.long, device=device)
```

> **为什么要构建完整序列？** `output_attentions=True` 需要一次性处理完整序列。我们需要看 answer tokens 对 latent tokens 的注意力，所以序列必须包含 prompt（含 latent）+ answer。

> **为什么 `full_attention_mask` 全是 1？** 因为完整序列中没有 padding，所有位置都是有效的。

**OOM 处理（行 600-652）**：

```python
try:
    with torch.no_grad():
        outputs = model(
            input_ids=full_input_ids,
            ...
            output_attentions=True,
            ...
        )
except torch.cuda.OutOfMemoryError:
    print("  ❌ OOM! 截断序列...")
    torch.cuda.empty_cache()
    
    # 找 latent_end 位置
    latent_end_pos = None
    for j, tid in enumerate(full_ids_list):
        if tid == LATENT_END_ID:
            latent_end_pos = j
    
    if latent_end_pos is not None:
        start_pos = max(0, latent_end_pos - 1500)
        trunc_ids = full_input_ids[:, start_pos:].to(device)
        ...
        adj_pos = [[p - start_pos for p in pos] for pos in ce_patch_pos]
        
        with torch.no_grad():
            outputs = model(
                input_ids=trunc_ids,
                ...
                pixel_values=None,  # 截断时不传图像
                ce_patch_pos=adj_pos,  # 调整 latent 位置
                ...
            )
```

> **为什么会有 OOM？** 注意力矩阵的大小是 `O(seq_len^2)`。如果序列有 2000 个 token，每层每个 head 的注意力矩阵就是 `2000×2000 = 4M` 个 float16 值 ≈ 8MB。28层 × 28 head = 总共约 `28 × 28 × 8MB ≈ 6.3GB`！再加上模型参数和其他中间结果，很容易 OOM。

> **截断策略**：
> 1. 找到 `</abs_vis_token>` 的位置（latent 区域结束）
> 2. 从该位置往前取 1500 个 token，加上之后的 answer tokens
> 3. 这样保留了 latent tokens 和 answer tokens，只丢弃了部分 system/question tokens
> 4. `ce_patch_pos` 需要相应调整：`adj_pos = [[p - start_pos for p in pos] for pos in ce_patch_pos]`
>
> **为什么从 latent_end 往前取 1500？** 因为 system + question + latent 的总长度大约是 1500-2000，取 1500 可以保留大部分上下文，同时把序列长度控制在可处理范围内。

> **为什么截断时 `pixel_values=None`？** 截断后，图像相关的 token 可能已经被丢弃了（在 system/question 区域），所以不需要再传图像数据。

> **为什么用 `torch.no_grad()` 而不是 `torch.inference_mode()`？** 在 try-except 块中，`inference_mode` 可能和异常处理有兼容性问题。`no_grad()` 更安全。

> **为什么 `attn_size_gb` 的计算公式是 `28 * 28 * total_len * total_len * 2 / (1024**3)`？**
> - 28 层 × 28 个注意力头
> - `total_len × total_len`：每个 head 的注意力矩阵大小
> - `2`：每个 float16 值占 2 bytes
> - `1024^3`：转换为 GB

---

## 第四部分：核心工程细节深入

### 4.1 为什么需要"三阶段"而不是"两阶段"？

原始的四步策略文档说：
1. Phase 1: latent_mode=True 获取 latent embeddings
2. Phase 2: latent_mode=False + ce_patch 获取注意力

但实际代码中，Phase 2 被拆成了 2a、2b、2c 三个子阶段。为什么？

**原因**：`output_attentions=True` 的限制。

- Phase 2a 用 `output_attentions=False` + `use_cache=True`：快速获取 KV cache，**不存储注意力矩阵**
- Phase 2b 用 KV cache 逐步生成 answer tokens：**也不存储注意力**
- Phase 2c 对完整序列用 `output_attentions=True`：**这才存储注意力**

如果直接在 Phase 2a 就开启 `output_attentions=True`，会有两个问题：
1. 只处理 prompt 部分，没有 answer tokens，分析不了 answer 对 latent 的注意力
2. prompt 部分可能很长（含图像），注意力矩阵巨大，容易 OOM

所以三阶段策略是：
> **先高效地生成完整序列（Phase 2a+2b），再对完整序列做一次注意力提取（Phase 2c）**

### 4.2 Eager Attention vs Flash Attention 的选择

```python
config.text_config._attn_implementation = "eager"
```

| 特性 | Eager Attention | Flash Attention |
|---|---|---|
| 返回注意力权重 | ✅ 是 | ❌ 否 |
| 计算速度 | 慢 | 快（约2-4倍） |
| 内存占用 | 高（O(n²)） | 低（O(n)） |
| 适用场景 | 分析、调试 | 训练、推理 |

> **关键决策**：Flash Attention 不保存中间的注意力权重矩阵，所以无法用于注意力分析。但 Flash Attention 的优势在训练和推理中非常明显（速度+内存），所以只在分析脚本中切换到 eager。

### 4.3 内存管理策略

脚本中大量使用了以下内存管理技巧：

```python
del latent_outputs
torch.cuda.empty_cache()

del phase2a_outputs
torch.cuda.empty_cache()

del past_kv_for_gen
torch.cuda.empty_cache()

del outputs, all_attentions
torch.cuda.empty_cache()
```

> **为什么需要手动管理内存？** Python 的垃圾回收（GC）不会立即释放 GPU 内存。即使变量不再使用，PyTorch 的 CUDA 内存分配器也会保留内存池以备后用。`torch.cuda.empty_cache()` 会把未使用的内存归还给 CUDA，让其他操作可以使用。

> **为什么在每个 Phase 结束后都释放？** 因为每个 Phase 产生的中间结果（如 KV cache、注意力矩阵）可能很大。如果不及时释放，后续 Phase 会叠加内存使用，容易 OOM。

> **`gc.collect()` 的作用**：在最后（行 702），还调用了 Python 的垃圾回收器。这是因为有些 Python 对象可能持有对 CUDA tensor 的引用（如闭包、弱引用等），`gc.collect()` 确保这些引用被清理。

### 4.4 为什么手动实现 greedy decode 而不用 `model.generate()`？

Transformers 提供了 `model.generate()` 方法，可以自动生成文本。但这个脚本手动实现了 greedy decode，原因是：

1. **`model.generate()` 不支持 `latent_mode`**：generate 方法是 transformers 库的标准实现，不了解 Monet 的 latent 推理机制
2. **需要使用 Phase 2a 的 KV cache**：generate 方法有自己的 KV cache 管理，无法直接传入我们预先计算的 KV cache
3. **需要绕过 `logits=None` 的问题**：在 `loss_type=[]` 时，model.forward() 不计算 logits

所以手动实现更灵活、更可控。

### 4.5 注意力矩阵的存储格式

`all_attentions` 是一个 tuple，每个元素对应一层 transformer：

```
all_attentions[0]  → 第0层的注意力
all_attentions[1]  → 第1层的注意力
...
all_attentions[27] → 第27层的注意力（最后一层）

每个元素的 shape: (batch_size, num_heads, seq_len, seq_len)
```

对于 Monet-SFT-7B（基于 Qwen2.5-VL-7B）：
- batch_size = 1
- num_heads = 28
- seq_len = 完整序列长度（prompt + latent + answer）

> **为什么 shape 是 `(1, 28, seq_len, seq_len)` 而不是 `(1, seq_len, seq_len)`？** 因为每个注意力头有独立的注意力分布。28 个头可能关注不同的模式。`mean(dim=0)` 在 head 维度上取平均，得到 `(seq_len, seq_len)` 的综合注意力矩阵。

---

## 第五部分：术语速查表

| 术语 | 英文 | 含义 |
|---|---|---|
| Latent CoT | Latent Chain-of-Thought | 隐式思维链，模型内部不可见的思考过程 |
| Latent token | Latent token | 隐式思考单元，不会转化为可见文字 |
| Latent embedding | Latent embedding | latent token 的向量表示（即模型的"思考内容"） |
| ce_patch_pos | Cross-entropy patch positions | latent tokens 在序列中的位置列表 |
| ce_patch_vec | Cross-entropy patch vectors | latent tokens 的真正 embedding 向量 |
| KV cache | Key-Value cache | 存储已计算过的 Key 和 Value，加速逐步生成 |
| Eager attention | Eager attention | 基础注意力实现，返回完整注意力矩阵 |
| Flash Attention | Flash Attention | 高效注意力算法，不返回注意力矩阵 |
| Greedy decode | Greedy decoding | 每步选概率最高的 token，不考虑多样性 |
| OOM | Out of Memory | GPU 内存不足 |
| Monkey patch | Monkey patch | 运行时动态替换模块代码的技术 |
| RoPE | Rotary Position Embedding | 旋转位置编码，Qwen2.5-VL 使用3D RoPE |
| Attention allocation | Attention allocation | 每个 token 对各类 token 的注意力占比 |
| mrope | Multimodal RoPE | 多模态旋转位置编码，同时处理文本和图像的位置 |

---

## 附录：代码中的关键"为什么"总结

| 问题 | 答案 |
|---|---|
| 为什么需要两步前向传播？ | latent_mode=True 不支持 output_attentions |
| 为什么用 eager attention？ | 需要获取注意力权重矩阵，Flash Attention 不提供 |
| 为什么手动 greedy decode？ | model.generate() 不支持 latent_mode 和自定义 KV cache |
| 为什么用 model.model() 而不是 model()？ | loss_type=[] 时 model() 不计算 logits |
| 为什么只取最后一层注意力？ | 最后一层最接近模型最终决策，语义级关注 |
| 为什么对多头取平均？ | 简化分析，看整体趋势 |
| 为什么限制80个token显示？ | 防止热力图太宽，影响可视化效果 |
| 为什么用NaN填充？ | 不同样本长度不同，NaN 填充比0填充更准确 |
| 为什么 Phase 2c 要 OOM 处理？ | 注意力矩阵 O(n²) 内存，长序列容易溢出 |
| 为什么冻结视觉模块？ | 只做分析，不需要训练，节省内存 |
| 为什么用 bfloat16？ | 现代大模型推理标配，动态范围大，计算效率高 |
| 为什么 add_tokens 设 special_tokens=True？ | 防止 tokenizer 拆分特殊 token |
| 为什么 attention_mask 要逐步扩展？ | 新生成的 token 需要标记为有效 |
| 为什么 ce_patch_pos 截断时要调整？ | 位置偏移了 start_pos，需要减去偏移量 |

---

> **最后建议**：如果你是初学者，建议按以下顺序学习这个脚本：
> 1. 先读懂文件头部的文档字符串（行 1-29）—— 理解整体目的
> 2. 再读 `classify_token_positions` —— 理解 token 分类
> 3. 再读 `compute_attention_allocation` —— 理解核心计算
> 4. 最后读 `main` 函数 —— 理解完整流程
> 5. 需要深入理解 latent_mode 时，再看 `modeling_qwen2_5_vl_monet.py` 中 `Qwen2_5_VLModel.forward` 的 latent_mode 部分