"""
Monet-SFT-7B 注意力分配分析脚本

目的：分析模型在 latent 推理后生成的文本 token 的注意力分配情况，
特别关注有多少注意力分配到 latent tokens，用热力图展示。

策略：
1. 用 processor 处理对话，在 assistant 回答区域插入 <abs_vis_token> 和 latent pad tokens
2. 用 latent_mode=True + output_latent_embeds=True 前向传播，获取 latent embeddings
3. 用 latent_mode=False + ce_patch_pos/ce_patch_vec + output_attentions=True 获取注意力权重
4. 分析每个文本 token 对 latent tokens 的注意力占比

运行方式：
cd /home/xiaojunhao/m-x && \
CUDA_VISIBLE_DEVICES=4 python -m inference.analyze_attention

source /home/xiaojunhao/miniconda3/etc/profile.d/conda.sh && conda activate monet && export LATENT_SIZE=10 && export MONET_MODEL_PATH=/home/xiaojunhao/m-x/data/Monet-SFT-7B/stage3 && export CUDA_VISIBLE_DEVICES=4 && cd /home/xiaojunhao/m-x && timeout 600 python -m inference.analyze_attention 2>&1 | grep -v "🚨"

"""

import os
import sys
import json
import re
import gc
import numpy as np
import torch
import PIL.Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

# ─── 打补丁 ───
from monet_qwen_model import apply_qwen2_5_monet
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.utils import add_latent_pad_after_auxiliary_img, replace_latent_placeholder_with_img_pad

# 特殊 token ID
LATENT_START_ID = 151666
LATENT_END_ID = 151667
LATENT_TOKEN_ID = 151665
IMAGE_TOKEN_ID = 151655
VISION_START_ID = 151652
VISION_END_ID = 151653
VISION_TOKEN_ID = 151654

def replace_abs_vis_token_content(s):
    pattern = re.compile(r'(<abs_vis_token>)(.*?)(</abs_vis_token>)', flags=re.DOTALL)
    return pattern.sub(r'\1<latent>\3', s)


def load_model(model_path, device="cuda:0", latent_size=10):
    """加载 Monet 模型（LLM 用 eager attention，视觉用 Flash Attention）"""
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
    
    print(f"✅ 模型加载完成, device={device}")
    print(f"  latent_start_id={model.config.latent_start_id}, latent_end_id={model.config.latent_end_id}")
    print(f"  latent_token_id={model.config.latent_token_id}")
    
    return model, processor


def prepare_input_for_sample(conversation, processor, latent_size, device):
    """
    处理输入，在 assistant 回答区域插入 latent tokens。
    参考训练代码中的做法：
    1. apply_chat_template 生成 prompt
    2. replace_latent_placeholder_with_img_pad 处理 latent placeholder
    3. add_latent_pad_after_auxiliary_img 在 image 后面插入 latent pad tokens
    """
    # 生成 chat template
    prompt_text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
    )
    
    # Step 1: 替换 latent placeholder 为 image pad（如果有的话）
    # 这里 prompt_text 是 <|im_start|>user ... <|im_end|><|im_start|>assistant
    # 没有 <abs_vis_token></abs_vis_token>（这是推理，不是训练）
    # 所以 replace_latent_placeholder_with_img_pad 不会改变什么
    prompt_text = replace_latent_placeholder_with_img_pad(prompt_text)
    
    # Step 2: 在 <|im_start|>assistant 后面的 image 区域后面插入 latent pad tokens
    # 这会生成：<|im_start|>assistant<|vision_start|><|image_pad|><|vision_end|><abs_vis_token><abs_vis_token_pad>...<abs_vis_token_pad></abs_vis_token>
    # 但实际上推理时 assistant 区域没有 image，只有文本
    # 我们需要在 assistant 后面直接插入 <abs_vis_token>{latent_pad_str*latent_size}</abs_vis_token>
    
    # 实际上 add_latent_pad_after_auxiliary_img 是在 <|vision_start|><|image_pad|><|vision_end|> 后面插入的
    # 如果 assistant 区域没有图像，这个函数不会做任何事情
    # 我们需要手动在 <|im_start|>assistant 后面插入 latent tokens
    
    sep_token = "<|im_start|>assistant"
    latent_pad_str = "<abs_vis_token_pad>"
    latent_pad_strs = latent_pad_str * latent_size
    
    # 在 <|im_start|>assistant 后面插入 <abs_vis_token>{latent_pads}</abs_vis_token>
    # 这让模型在 latent_mode 下知道需要在哪里做 latent 推理
    prompt_text_with_latent = prompt_text.replace(
        sep_token,
        f"{sep_token}<abs_vis_token>{latent_pad_strs}</abs_vis_token>"
    )
    
    # 提取图像
    image_inputs, _ = process_vision_info(conversation, return_video_kwargs=False)
    
    # 用 processor 处理
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
    
    print(f"  Prompt with latent: {prompt_len} tokens")
    print(f"  Has latent tokens: {LATENT_TOKEN_ID in inputs.input_ids[0].tolist()}")
    
    return model_inputs, prompt_len, image_inputs


def classify_token_positions(token_ids):
    """将 token 序列中的每个位置分类为不同的语义类别"""
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
    
    # answer_tokens = </abs_vis_token> 之后的所有非特殊 tokens
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


def compute_attention_allocation(all_attentions, categories, token_ids):
    """计算每个 answer token 对各类 token 的注意力分配比例"""
    answer_positions = categories['answer_tokens']
    if len(answer_positions) == 0:
        print("  ❌ 没有 answer tokens")
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


def create_heatmaps(all_matrices, focus_categories, results, output_dir, per_sample_dir):
    """生成所有热力图和统计"""
    
    # 单样本热力图
    for i, m in enumerate(all_matrices):
        if m is None:
            continue
        max_show = min(m.shape[0], 80)
        matrix_show = m[:max_show]
        
        fig, ax = plt.subplots(figsize=(max_show * 0.15 + 3, len(focus_categories) * 0.6 + 2))
        im = ax.imshow(matrix_show, cmap=plt.cm.YlOrRd, aspect='auto', vmin=0, vmax=100)
        
        cat_names = [name for _, name in focus_categories]
        ax.set_yticks(range(len(cat_names)))
        ax.set_yticklabels(cat_names, fontsize=10)
        
        step = max(1, max_show // 20)
        ax.set_xticks(range(0, max_show, step))
        ax.set_xticklabels([str(j+1) for j in range(0, max_show, step)], fontsize=8)
        ax.set_xlabel('Answer Token Position', fontsize=11)
        ax.set_ylabel('Token Category', fontsize=11)
        
        latent_pct = m[:, 0].mean()
        ax.set_title(f'Sample {i}: Latent CoT Attention = {latent_pct:.1f}%\n'
                     f'(after </abs_vis_token>)', fontsize=12)
        plt.colorbar(im, ax=ax, label='Attention %')
        plt.tight_layout()
        plt.savefig(os.path.join(per_sample_dir, f'sample_{i}_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 平均热力图
    if all_matrices:
        max_show = 80
        trimmed = []
        for m in all_matrices:
            if m is not None and m.shape[0] > 0:
                t = m[:max_show]
                if t.shape[0] < max_show:
                    p = np.full((max_show, t.shape[1]), np.nan)
                    p[:t.shape[0]] = t
                    trimmed.append(p)
                else:
                    trimmed.append(t)
        
        if trimmed:
            avg = np.nanmean(np.stack(trimmed), axis=0)
            
            fig, ax = plt.subplots(figsize=(max_show * 0.15 + 3, len(focus_categories) * 0.8 + 3))
            im = ax.imshow(avg, cmap=plt.cm.YlOrRd, aspect='auto', vmin=0, vmax=100)
            
            cat_names = [name for _, name in focus_categories]
            ax.set_yticks(range(len(cat_names)))
            ax.set_yticklabels(cat_names, fontsize=11)
            step = max(1, max_show // 20)
            ax.set_xticks(range(0, max_show, step))
            ax.set_xticklabels([str(j+1) for j in range(0, max_show, step)], fontsize=9)
            ax.set_xlabel('Answer Token Position (after </abs_vis_token>)', fontsize=12)
            ax.set_ylabel('Token Category', fontsize=12)
            
            latent_avg = avg[:, 0].mean()
            ax.set_title(f'Average Attention Allocation ({len(trimmed)} samples)\n'
                         f'Latent CoT = {latent_avg:.1f}% avg', fontsize=13, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Attention %')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'averaged_attention_heatmap.png'), dpi=200, bbox_inches='tight')
            plt.close()
            
            # Evolution curve
            latent_curves = [m[:max_show, 0] for m in all_matrices if m is not None and m.shape[0] > 0]
            max_len = max(len(c) for c in latent_curves)
            stacked = np.full((len(latent_curves), max_len), np.nan)
            for j, c in enumerate(latent_curves):
                stacked[j, :len(c)] = c
            
            avg_curve = np.nanmean(stacked, axis=0)
            std_curve = np.nanstd(stacked, axis=0)
            
            fig, ax = plt.subplots(figsize=(12, 5))
            pos = range(1, len(avg_curve) + 1)
            ax.plot(pos, avg_curve, 'b-', linewidth=2, label='Mean Latent Attention %')
            ax.fill_between(pos, np.nan_to_num(avg_curve - std_curve, nan=0),
                            np.nan_to_num(avg_curve + std_curve, nan=100),
                            alpha=0.3, color='blue', label='±1 Std Dev')
            ax.axhline(y=avg_curve.mean(), color='g', linestyle='-', alpha=0.7,
                       label=f'Overall mean = {avg_curve.mean():.1f}%')
            ax.set_xlabel('Answer Token Position', fontsize=12)
            ax.set_ylabel('Attention to Latent CoT (%)', fontsize=12)
            ax.set_title(f'Evolution of Latent Attention ({len(trimmed)} samples)', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'latent_attention_evolution.png'), dpi=200, bbox_inches='tight')
            plt.close()
            
            # Stats JSON
            all_vals = np.concatenate([m for m in all_matrices if m is not None], axis=0)
            stats = {}
            for j, (ck, cn) in enumerate(focus_categories):
                stats[cn] = {'mean': float(all_vals[:, j].mean()), 'std': float(all_vals[:, j].std())}
            stats['latent_detail'] = {
                'mean_pct': float(all_vals[:, 0].mean()),
                'pct_above_10': float(np.mean(all_vals[:, 0] > 10) * 100),
                'pct_above_30': float(np.mean(all_vals[:, 0] > 30) * 100),
            }
            with open(os.path.join(output_dir, 'attention_stats.json'), 'w') as f:
                json.dump(stats, f, indent=2)


def main():
    model_path = os.environ.get("MONET_MODEL_PATH", "/home/xiaojunhao/m-x/data/Monet-SFT-7B/stage3")
    dataset_path = "/home/xiaojunhao/m-x/data/Monet-SFT-125K/Zebra_CoT_geometry/train.json"
    base_image_dir = "/home/xiaojunhao/m-x/data/Monet-SFT-125K"
    output_dir = "/home/xiaojunhao/m-x/inference/attention_analysis_results"
    latent_size = int(os.environ.get("LATENT_SIZE", "10"))
    
    os.makedirs(output_dir, exist_ok=True)
    per_sample_dir = os.path.join(output_dir, "per_sample")
    os.makedirs(per_sample_dir, exist_ok=True)
    
    num_samples = 2
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Latent size: {latent_size}")
    print(f"Samples: {num_samples}")
    print(f"Device: {device}")
    
    # Step 1: 加载模型
    model, processor = load_model(model_path, device, latent_size)
    
    # Step 2: 加载数据集
    print("=" * 60)
    print("Step 2: 加载数据集...")
    print("=" * 60)
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    data = data[:num_samples]
    
    # Step 3: 对每个样本做注意力分析
    print("=" * 60)
    print("Step 3: 注意力分析...")
    print("=" * 60)
    
    all_matrices = []
    focus_categories = None
    all_results = []
    
    for i, item in enumerate(data):
        print(f"\n=== Sample {i} ===")
        
        # 构建对话格式（只取 user 部分）
        user_msg = next(msg for msg in item["data"] if msg["role"] == "user")
        conv_content = []
        for block in user_msg["content"]:
            if block["type"] == "image":
                img_path = os.path.join(base_image_dir, block["image"])
                conv_content.append({"type": "image", "image": PIL.Image.open(img_path).convert("RGB")})
            elif block["type"] == "text":
                conv_content.append(block)
        
        conversation = [{"role": "user", "content": conv_content}]
        
        # 准备输入（包含 latent pad tokens）
        model_inputs, prompt_len, image_inputs = prepare_input_for_sample(
            conversation, processor, latent_size, device
        )
        
        # ─── Phase 1: latent_mode=True 前向传播 ───
        print("  Phase 1: latent_mode=True...")
        
        with torch.inference_mode():
            latent_outputs = model(
                **model_inputs,
                latent_mode=True,
                output_latent_embeds=False,  # 不需要 latent_embeds（需要 alignment_poss，我们没有）
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
        
        # 提取 ce_patch_pos 和 ce_patch_vec（用于 Phase 2 替换 latent embeddings）
        ce_patch_pos = latent_outputs.ce_patch_pos
        ce_patch_vec = latent_outputs.ce_patch_vec
        
        if ce_patch_pos is None or ce_patch_vec is None:
            print("  ❌ 无 ce_patch_pos/ce_patch_vec，跳过")
            all_results.append(None)
            del latent_outputs
            torch.cuda.empty_cache()
            continue
        
        # 计算 latent token 数量
        num_latents = len(ce_patch_pos[0]) if ce_patch_pos[0] else 0
        print(f"  latent embeddings: {num_latents} tokens")
        print(f"  ce_patch_pos: {ce_patch_pos}")
        
        # 解码 input_ids 来看有没有 latent tokens
        input_ids = model_inputs['input_ids'][0].tolist()
        has_latent_start = LATENT_START_ID in input_ids
        has_latent_end = LATENT_END_ID in input_ids
        print(f"  <abs_vis_token> in input_ids: {has_latent_start}")
        print(f"  </abs_vis_token> in input_ids: {has_latent_end}")
        
        if not has_latent_start or not has_latent_end:
            print("  ❌ input_ids 中没有 latent 边界标记，跳过")
            all_results.append(None)
            del latent_outputs
            torch.cuda.empty_cache()
            continue
        
        del latent_outputs
        torch.cuda.empty_cache()
        
        # ─── Phase 2: 生成 answer + 提取注意力 ───
        # Step 2a: 用 latent_mode=False + ce_patch_vec 前向传播获取 KV cache
        print("  Phase 2a: latent_mode=False 前向传播获取 KV cache...")
        
        with torch.inference_mode():
            phase2a_outputs = model(
                **model_inputs,
                latent_mode=False,
                ce_patch_pos=ce_patch_pos,
                ce_patch_vec=ce_patch_vec,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True,
            )
        
        past_kv = phase2a_outputs.past_key_values
        print(f"  KV cache 获取成功")
        
        del phase2a_outputs
        torch.cuda.empty_cache()
        
        # Step 2b: 手动逐步生成 answer tokens
        # 由于 model.forward() 在 loss_type=[] 时 logits=None，
        # 我们需要用 model.model() 获取 hidden_states，然后用 model.lm_head() 计算 logits
        print("  Phase 2b: 生成 answer tokens...")
        
        max_new_tokens = 256
        generated_tokens = []
        
        # 用 Phase 2a 的 KV cache 加速
        # 逐步生成：每次传一个 token + past KV cache，获取 hidden_states，然后 lm_head 计算 logits
        next_input_ids = model_inputs['input_ids'][:, -1:]  # 最后一个 token
        full_attn_mask = model_inputs['attention_mask'].clone()
        past_kv_for_gen = past_kv
        
        with torch.inference_mode():
            for step in range(max_new_tokens):
                # 使用 past KV cache + 新 token
                # 注意：使用 KV cache 时不需要传入 pixel_values
                # 图像的 KV cache 已经在 Phase 2a 的 past_key_values 中
                # ce_patch_pos/ce_patch_vec 也已经在 Phase 2a 中被替换过了
                model_out = model.model(
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
                
                # 用 lm_head 计算 logits
                hidden_states = model_out.last_hidden_state
                logits = model.lm_head(hidden_states)
                next_token_logits = logits[:, -1, :]
                
                # greedy decode
                next_token = next_token_logits.argmax(dim=-1)
                
                if next_token.item() == processor.tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(next_token.item())
                
                # 更新
                next_input_ids = next_token.unsqueeze(0)
                full_attn_mask = torch.cat([
                    full_attn_mask,
                    torch.ones(1, 1, dtype=torch.long, device=device)
                ], dim=1)
                past_kv_for_gen = model_out.past_key_values
                
                if step % 50 == 0 and step > 0:
                    decoded = processor.tokenizer.decode(generated_tokens[-10:], skip_special_tokens=True)
                    print(f"    Step {step}: {len(generated_tokens)} tokens, recent: {decoded}")
        
        print(f"  生成完成: {len(generated_tokens)} answer tokens")
        
        # 解码输出文本
        answer_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=False)
        cleaned_answer = replace_abs_vis_token_content(answer_text)
        print(f"  Answer preview: {cleaned_answer[:100]}...")
        
        del past_kv_for_gen
        torch.cuda.empty_cache()
        
        # Step 2c: 对完整序列做 output_attentions=True 前向传播
        print("  Phase 2c: 对完整序列提取注意力权重...")
        
        # 构建完整序列 = prompt + latent + answer
        full_input_ids = torch.cat([
            model_inputs['input_ids'],
            torch.tensor([generated_tokens], dtype=torch.long, device=device)
        ], dim=1)
        
        full_attention_mask = torch.ones(1, full_input_ids.shape[1], dtype=torch.long, device=device)
        
        total_len = full_input_ids.shape[1]
        attn_size_gb = 28 * 28 * total_len * total_len * 2 / (1024**3)
        print(f"  Total tokens: {total_len}, Attn matrix: {attn_size_gb:.2f} GB")
        
        # 构建完整的 token ID 列表（用于分类和分析）
        full_ids_list = full_input_ids[0].tolist()
        
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    pixel_values=model_inputs['pixel_values'],
                    image_grid_thw=model_inputs['image_grid_thw'],
                    latent_mode=False,
                    ce_patch_pos=ce_patch_pos,
                    ce_patch_vec=ce_patch_vec,
                    output_attentions=True,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
        except torch.cuda.OutOfMemoryError:
            print("  ❌ OOM! 截断序列...")
            torch.cuda.empty_cache()
            
            # 找 latent_end 位置
            full_ids_list = full_input_ids[0].tolist()
            latent_end_pos = None
            for j, tid in enumerate(full_ids_list):
                if tid == LATENT_END_ID:
                    latent_end_pos = j
            
            if latent_end_pos is not None:
                start_pos = max(0, latent_end_pos - 1500)
                trunc_ids = full_input_ids[:, start_pos:].to(device)
                trunc_mask = torch.ones(1, trunc_ids.shape[1], dtype=torch.long, device=device)
                adj_pos = [[p - start_pos for p in pos] for pos in ce_patch_pos]
                
                print(f"  截断: {start_pos}~{total_len}")
                
                with torch.no_grad():
                    outputs = model(
                        input_ids=trunc_ids,
                        attention_mask=trunc_mask,
                        pixel_values=None,
                        image_grid_thw=None,
                        latent_mode=False,
                        ce_patch_pos=adj_pos,
                        ce_patch_vec=ce_patch_vec,
                        output_attentions=True,
                        output_hidden_states=False,
                        use_cache=False,
                        return_dict=True,
                    )
                full_ids_list = full_ids_list[start_pos:]
            else:
                print("  ❌ Cannot truncate")
                all_results.append(None)
                continue
        
        all_attentions = outputs.attentions
        if all_attentions is None:
            print("  ❌ attentions is None")
            all_results.append(None)
            del outputs
            torch.cuda.empty_cache()
            continue
        
        print(f"  ✅ Got {len(all_attentions)} layers, shape: {all_attentions[0].shape}")
        
        # 分类 token 位置
        categories = classify_token_positions(full_ids_list)
        for cn, ps in categories.items():
            if ps:
                print(f"    {cn}: {len(ps)} tokens")
        
        # 计算注意力分配
        alloc = compute_attention_allocation(all_attentions, categories, full_ids_list)
        if alloc is None:
            all_results.append(None)
            del outputs
            torch.cuda.empty_cache()
            continue
        
        attn_matrix, focus_cats, answer_pos = alloc
        focus_categories = focus_cats
        
        latent_pct = attn_matrix[:, 0].mean()
        print(f"    Latent CoT avg attention: {latent_pct:.2f}%")
        print(f"    First 5 answer tokens latent attn: {attn_matrix[:5, 0]}")
        print(f"    Last 5 answer tokens latent attn: {attn_matrix[-5:, 0]}")
        
        all_matrices.append(attn_matrix)
        all_results.append({'num_latents': num_latents})
        
        del outputs, all_attentions
        torch.cuda.empty_cache()
    
    # Step 4: 生成热力图
    print("\n" + "=" * 60)
    print("Step 4: 生成热力图...")
    print("=" * 60)
    
    if focus_categories is not None:
        create_heatmaps(all_matrices, focus_categories, all_results, output_dir, per_sample_dir)
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\n✅ 完成！结果保存在:", output_dir)


if __name__ == '__main__':
    main()