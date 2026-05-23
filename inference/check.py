"""
Monet-SFT-7B 两种 latent mode 一致性检测脚本

目的：验证 latent_mode=True 和 latent_mode=False 两种推理模式的行为是否一致。
这是确保注意力分析有效性的关键前提。

检测方法：
1. 加载同一个样本
2. Phase 1: latent_mode=True 前向传播 → 获取 hidden_states at latent positions
3. Phase 2: latent_mode=False + ce_patch_vec 前向传播 → 获取 hidden_states at same positions
4. 比较：hidden states 是否一致？生成的 answer tokens 是否一致？

运行方式：
cd /home/xiaojunhao/m-x && \
CUDA_VISIBLE_DEVICES=4 python -m inference.check

source /home/xiaojunhao/miniconda3/etc/profile.d/conda.sh && conda activate monet && \
export LATENT_SIZE=10 && \
export MONET_MODEL_PATH=/home/xiaojunhao/m-x/data/Monet-SFT-7B/stage3 && \
export CUDA_VISIBLE_DEVICES=4 && \
cd /home/xiaojunhao/m-x && \
timeout 300 python -m inference.check 2>&1 | grep -v "🚨"
"""

import os
import sys
import json
import gc
import numpy as np
import torch
import PIL.Image
from typing import List, Dict, Optional, Tuple

# ─── 打补丁 ───
from monet_qwen_model import apply_qwen2_5_monet
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.utils import replace_latent_placeholder_with_img_pad

# 特殊 token ID
LATENT_START_ID = 151666
LATENT_END_ID = 151667
LATENT_TOKEN_ID = 151665
IMAGE_TOKEN_ID = 151655


def load_model(model_path, device="cuda:0"):
    """加载 Monet 模型"""
    print("=" * 60)
    print("加载 Monet 模型...")
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
    
    model.eval()
    model.to(device)
    
    print(f"✅ 模型加载完成, device={device}")
    print(f"  latent_start_id={model.config.latent_start_id}, latent_end_id={model.config.latent_end_id}")
    
    return model, processor


def prepare_input(conversation, processor, latent_size, device):
    """准备输入（在 assistant 区域插入 latent tokens）"""
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
    
    return model_inputs, inputs.input_ids[0].tolist()


def check_consistency(model, processor, model_inputs, input_ids_list, device, latent_size):
    """
    检测两种 latent mode 的一致性
    
    核心思路：
    1. latent_mode=True: 模型自动替换 latent token embeddings 为 previous hidden state
    2. latent_mode=False + ce_patch_vec: 模型用传入的 ce_patch_vec 替换 latent token embeddings
    3. 如果两者一致，那么最终产生的 hidden states 应该在 latent positions 上相同，
       生成的 answer tokens 也应该相同
    """
    results = {
        'latent_pos_hidden_states_match': False,
        'hidden_states_cosine_sim': 0.0,
        'final_answer_tokens_match': False,
        'answer_tokens_cosine_sim': 0.0,
        'details': {},
    }
    
    # ─── Mode 1: latent_mode=True ───
    print("\n  [Mode 1] latent_mode=True 前向传播...")
    
    # 使用 model.model() 直接获取 last_hidden_state (batch, seq_len, hidden_dim)
    # 而非 model() 的 hidden_states 输出（latent_mode=True 时 hidden_states 格式为
    # (num_layers, num_latents, hidden_dim)，不是 (seq_len, hidden_dim)，会导致
    # 用 ce_patch_pos 的绝对位置索引时越界）
    with torch.inference_mode():
        mode1_outputs = model.model(
            **model_inputs,
            latent_mode=True,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
    
    mode1_last_hidden = mode1_outputs.last_hidden_state[0]  # [seq_len, hidden_dim]
    mode1_ce_patch_pos = mode1_outputs.ce_patch_pos
    mode1_ce_patch_vec = mode1_outputs.ce_patch_vec
    
    print(f"    隐藏状态形状: {mode1_last_hidden.shape}")
    print(f"    ce_patch_pos: {mode1_ce_patch_pos}")
    
    del mode1_outputs
    torch.cuda.empty_cache()
    
    # ─── Mode 2: latent_mode=False + ce_patch_vec ───
    print("\n  [Mode 2] latent_mode=False + ce_patch_vec 前向传播...")
    
    # 同样使用 model.model() 直接获取 last_hidden_state
    with torch.inference_mode():
        mode2_outputs = model.model(
            **model_inputs,
            latent_mode=False,
            ce_patch_pos=mode1_ce_patch_pos,
            ce_patch_vec=mode1_ce_patch_vec,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
    
    mode2_last_hidden = mode2_outputs.last_hidden_state[0]  # [seq_len, hidden_dim]
    print(f"    隐藏状态形状: {mode2_last_hidden.shape}")
    
    del mode2_outputs
    torch.cuda.empty_cache()
    
    # ─── 比较 1: latent positions 上的 hidden states ───
    print("\n  比较 1: Latent positions 上的 hidden states...")
    
    if mode1_ce_patch_pos and mode1_ce_patch_pos[0]:
        latent_pos = mode1_ce_patch_pos[0]  # e.g., [8089, 8090, ..., 8098]
        
        # 检查 latent_pos 是否在有效范围内
        max_pos = mode1_last_hidden.shape[0] - 1
        if any(p > max_pos for p in latent_pos):
            print(f"    ❌ latent_pos 超出范围! max valid: {max_pos}, latent_pos: {latent_pos}")
            print(f"    hidden_states shape: {mode1_last_hidden.shape}")
        else:
            # Mode 1 latent position hidden states
            mode1_latent_hs = mode1_last_hidden[latent_pos].cpu().float()  # [num_latent, hidden_dim]
            
            # Mode 2 latent position hidden states
            mode2_latent_hs = mode2_last_hidden[latent_pos].cpu().float()  # [num_latent, hidden_dim]
            
            print(f"    Latent positions: {latent_pos}")
            print(f"    Mode 1 latent hidden shape: {mode1_latent_hs.shape}")
            print(f"    Mode 2 latent hidden shape: {mode2_latent_hs.shape}")
            
            # 计算 cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                mode1_latent_hs.flatten(),
                mode2_latent_hs.flatten(),
                dim=0
            ).item()
            
            # 计算 L2 距离
            l2_dist = torch.norm(mode1_latent_hs - mode2_latent_hs).item() / np.sqrt(mode1_latent_hs.numel())
            
            print(f"    Cosine similarity: {cos_sim:.6f}")
            print(f"    Normalized L2 distance: {l2_dist:.6f}")
            
            results['latent_pos_hidden_states_match'] = cos_sim > 0.99
            results['hidden_states_cosine_sim'] = cos_sim
            results['details']['latent_cosine_sim'] = cos_sim
            results['details']['latent_l2_dist'] = l2_dist
            results['details']['num_latent_positions'] = len(latent_pos)
    else:
        print("    ❌ 无 latent positions")
    
    # ─── 比较 2: latent_end 之后的 hidden states ───
    print("\n  比较 2: Latent end 之后的 hidden states...")
    
    # 找 latent_end 位置
    latent_end_pos = None
    for i, tid in enumerate(input_ids_list):
        if tid == LATENT_END_ID:
            latent_end_pos = i
            break
    
    seq_len = mode1_last_hidden.shape[0]  # 实际 hidden states 的序列长度
    if latent_end_pos is not None:
        # 取 latent_end + 1, +2, +3 位置的 hidden states
        positions_to_check = [latent_end_pos + 1, latent_end_pos + 2, latent_end_pos + 3]
        positions_to_check = [p for p in positions_to_check if p < seq_len]
        
        if positions_to_check:
            mode1_after = mode1_last_hidden[positions_to_check].cpu().float()
            mode2_after = mode2_last_hidden[positions_to_check].cpu().float()
            
            cos_sim_after = torch.nn.functional.cosine_similarity(
                mode1_after.flatten(),
                mode2_after.flatten(),
                dim=0
            ).item()
            
            print(f"    Positions: {positions_to_check}")
            print(f"    Cosine similarity (after latent): {cos_sim_after:.6f}")
            
            results['details']['after_latent_cosine_sim'] = cos_sim_after
    else:
        print("    ❌ 未找到 latent_end 位置")
    
    # ─── 比较 3: 生成 answer tokens ───
    print("\n  比较 3: 生成 answer tokens...")
    
    # 由于 model.generate() 不能用于 Monet，我们需要手动生成
    # 这里我们简化：只比较第一层 decode 的 logits
    
    # 对于 Mode 1，我们需要获取 latent_end 后第一个 token 的 logits
    # 但 Mode 1 没有 logits（loss_type=[] 时返回 None）
    
    # 简化：用 hidden states 预测下一个 token，看是否一致
    # 提取 latent_end + 1 位置的 hidden states
    if latent_end_pos is not None:
        next_pos = latent_end_pos + 1
        if next_pos < seq_len:
            mode1_next_hs = mode1_last_hidden[next_pos]
            mode2_next_hs = mode2_last_hidden[next_pos]
            
            # 用 lm_head 计算 logits
            # lm_head 在 GPU 上，需要 clone inference tensor 并移到 GPU 上调用 lm_head，
            # 然后再移到 CPU 做 cosine similarity 比较
            # clone() 是因为 inference_mode 产生的 tensor 不能被 autograd 操作使用
            mode1_next_logits = model.lm_head(mode1_next_hs.clone().unsqueeze(0).to(device)).squeeze(0).cpu().float()
            mode2_next_logits = model.lm_head(mode2_next_hs.clone().unsqueeze(0).to(device)).squeeze(0).cpu().float()
            
            # 比较 logits
            logit_cos_sim = torch.nn.functional.cosine_similarity(
                mode1_next_logits,
                mode2_next_logits,
                dim=0
            ).item()
            
            # 比较 top-1 token
            mode1_top1 = mode1_next_logits.argmax().item()
            mode2_top1 = mode2_next_logits.argmax().item()
            mode1_top1_token = processor.tokenizer.decode([mode1_top1])
            mode2_top1_token = processor.tokenizer.decode([mode2_top1])
            
            print(f"    Next token position: {next_pos}")
            print(f"    Mode 1 top-1 token: {mode1_top1} ({mode1_top1_token!r})")
            print(f"    Mode 2 top-1 token: {mode2_top1} ({mode2_top1_token!r})")
            print(f"    Logits cosine similarity: {logit_cos_sim:.6f}")
            
            results['final_answer_tokens_match'] = (mode1_top1 == mode2_top1)
            results['answer_tokens_cosine_sim'] = logit_cos_sim
            results['details']['mode1_top1_token'] = mode1_top1_token
            results['details']['mode2_top1_token'] = mode2_top1_token
            results['details']['logits_cosine_sim'] = logit_cos_sim
    
    return results


def run_check(model, processor, data, device, latent_size):
    """对每个样本运行一致性检测"""
    all_results = []
    
    for i, item in enumerate(data):
        print(f"\n{'='*60}")
        print(f"Sample {i}")
        print('='*60)
        
        # 构建对话格式
        user_msg = next(msg for msg in item["data"] if msg["role"] == "user")
        conv_content = []
        for block in user_msg["content"]:
            if block["type"] == "image":
                img_path = os.path.join("/home/xiaojunhao/m-x/data/Monet-SFT-125K", block["image"])
                conv_content.append({"type": "image", "image": PIL.Image.open(img_path).convert("RGB")})
            elif block["type"] == "text":
                conv_content.append(block)
        
        conversation = [{"role": "user", "content": conv_content}]
        
        # 准备输入
        model_inputs, input_ids_list = prepare_input(
            conversation, processor, latent_size, device
        )
        
        print(f"  Prompt tokens: {len(input_ids_list)}")
        
        # 检查是否有 latent tokens
        has_latent_start = LATENT_START_ID in input_ids_list
        has_latent_end = LATENT_END_ID in input_ids_list
        print(f"  Has <abs_vis_token>: {has_latent_start}")
        print(f"  Has </abs_vis_token>: {has_latent_end}")
        
        if not has_latent_start or not has_latent_end:
            print("  ❌ 没有 latent tokens，跳过")
            all_results.append({'skipped': True})
            continue
        
        # 运行一致性检测
        results = check_consistency(
            model, processor, model_inputs, input_ids_list, device, latent_size
        )
        
        all_results.append(results)
        
        # 打印结果摘要
        print(f"\n  📊 一致性检测结果:")
        print(f"    Latent positions hidden states match: {results['latent_pos_hidden_states_match']}")
        print(f"    Cosine similarity: {results['hidden_states_cosine_sim']:.6f}")
        print(f"    Answer tokens match: {results['final_answer_tokens_match']}")
        print(f"    Logits cosine sim: {results['answer_tokens_cosine_sim']:.6f}")
        
        # 清理
        torch.cuda.empty_cache()
        gc.collect()
    
    return all_results


def main():
    model_path = os.environ.get("MONET_MODEL_PATH", "/home/xiaojunhao/m-x/data/Monet-SFT-7B/stage3")
    dataset_path = "/home/xiaojunhao/m-x/data/Monet-SFT-125K/Zebra_CoT_geometry/train.json"
    latent_size = int(os.environ.get("LATENT_SIZE", "10"))
    num_samples = 2
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Latent size: {latent_size}")
    print(f"Samples: {num_samples}")
    print(f"Device: {device}")
    
    # 加载模型
    model, processor = load_model(model_path, device)
    
    # 加载数据集
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    data = data[:num_samples]
    
    # 运行检测
    results = run_check(model, processor, data, device, latent_size)
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for i, r in enumerate(results):
        if r.get('skipped'):
            skipped += 1
            print(f"  Sample {i}: SKIPPED")
            continue
        
        latent_match = r.get('latent_pos_hidden_states_match', False)
        logits_match = r.get('final_answer_tokens_match', False)
        
        if latent_match and logits_match:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"
        
        print(f"  Sample {i}: {status}")
        print(f"    - Latent HS cosine sim: {r.get('hidden_states_cosine_sim', 0):.4f}")
        print(f"    - Logits cosine sim: {r.get('answer_tokens_cosine_sim', 0):.4f}")
    
    print(f"\n通过: {passed}/{num_samples - skipped}, 失败: {failed}, 跳过: {skipped}")
    
    if failed == 0 and skipped == 0:
        print("\n✅ 两种 latent mode 推理行为完全一致！")
        return 0
    else:
        print("\n⚠️ 两种 latent mode 推理行为存在差异，需要检查！")
        return 1


if __name__ == '__main__':
    sys.exit(main())