import os
import json
import csv
import re
import PIL.Image

"""
export http_proxy=http://oversea-squid2.ko.txyun:11080 https_proxy=http://oversea-squid2.ko.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
# gpu指定 4,5,6,7
source /home/xiaojunhao/miniconda3/etc/profile.d/conda.sh && conda activate monet && export LATENT_SIZE=10 && CUDA_VISIBLE_DEVICES=4,5,6,7 python -m inference.evaluate_zebra_geo
"""

# 导入 Monet 补丁和工具
import inference.apply_vllm_monet
from inference.load_and_gen_vllm import *

def replace_abs_vis_token_content(s: str) -> str:
    pattern = re.compile(r'(<abs_vis_token>)(.*?)(</abs_vis_token>)', flags=re.DOTALL)
    # 将 latent 内容替换为 <latent> 占位符
    return pattern.sub(r'\1<latent>\3', s)

def extract_boxed_value(text: str) -> str:
    """提取 \boxed{} 里面的内容，如果没有则返回空字符串。
       通过正则处理匹配 \boxed{...} 结构，支持嵌套一层 {} 的情况（如果有更深嵌套可以考虑栈匹配，但此处正则够用大部分）。
    """
    pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return ""

def evaluate_dataset(dataset_path, output_csv_path, mllm, sampling_params, processor, base_image_dir):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # print("!!! 小样本测试模式：仅测试前 5 个样本 !!!")
    # data = data[:5]
    
    conversations = []
    questions = []
    image_paths = []
    ground_truths = []
    ground_truths_cleaned = []
    
    for item in data:
        # 提取 user
        user_msg = next(msg for msg in item["data"] if msg["role"] == "user")
        conv_content = []
        question_text = ""
        img_path = ""
        for block in user_msg["content"]:
            if block["type"] == "image":
                img_path = os.path.join(base_image_dir, block["image"])
                conv_content.append({
                    "type": "image",
                    "image": PIL.Image.open(img_path).convert("RGB")
                })
            elif block["type"] == "text":
                question_text = block["text"]
                conv_content.append(block)
        
        conversations.append([{
            "role": "user",
            "content": conv_content
        }])
        questions.append(question_text)
        image_paths.append(img_path)

        # 尝试提取 Ground Truth
        # 在数据中找到 role: assistant 的内容
        gt_value = ""
        try:
            assistant_msg = next(msg for msg in item["data"] if msg["role"] == "assistant")
            text_blocks = [b["text"] for b in assistant_msg["content"] if b.get("type") == "text"]
            if text_blocks:
                last_text = text_blocks[-1].strip()
                # 取最后一个空格后的文本，并去掉末尾可能自带的句号
                gt_value = last_text.split()[-1].rstrip('.')
        except Exception as e:
            pass
            
        ground_truths.append(gt_value)
        ground_truths_cleaned.append(last_text)  # 保存原始文本以便后续分析（如是否包含 \boxed{} 结构等）
        
    print(f"============================================================")
    print(f"正在处理 {len(conversations)} 个样本: {dataset_path}...")
    print(f"============================================================")
    
    # 初次创建 CSV 并写入表头
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Image Path", "Triggered latent CoT", "Latent CoT Count", "Model Output (cleaned)", "Model Prediction (boxed)", "Ground Truth (raw)", "Ground Truth (cleaned)", "Correct Prediction (0/1)"])
        
    batch_size = 100

    for i in range(0, len(conversations), batch_size):
        batch_convs = conversations[i:i+batch_size]
        batch_questions = questions[i:i+batch_size]
        batch_images = image_paths[i:i+batch_size]
        batch_gts = ground_truths[i:i+batch_size]
        batch_gts_cleaned = ground_truths_cleaned[i:i+batch_size]

        print(f"Processing Batch {i}/{len(conversations)}...")
        inputs = vllm_mllm_process_batch_from_messages(batch_convs, processor)
        outputs = vllm_generate(inputs, sampling_params, mllm)

        batch_raw_outputs = [out.outputs[0].text for out in outputs]
        
        # 每次 batch 结束追加保存为 CSV
        with open(output_csv_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            
            for q, p, out_text, gt, gt_cleaned in zip(batch_questions, batch_images, batch_raw_outputs, batch_gts, batch_gts_cleaned):
                cleaned = replace_abs_vis_token_content(out_text)
                count = cleaned.count("<latent>")
                triggered = count > 0
                
                pred_val = extract_boxed_value(cleaned)
                
                is_correct = 0
                if pred_val and gt:
                    if pred_val.lower() == gt.lower() or pred_val.lower() in gt.lower() or gt.lower() in pred_val.lower():
                        is_correct = 1
                        
                writer.writerow([q, p, triggered, count, cleaned, pred_val, gt_cleaned, gt, is_correct])
            
    print(f"✅ 结果已保存至 {output_csv_path}\n")

def main():
    # 设置模型和环境变量
    model_path = os.environ.get("MONET_MODEL_PATH", "NOVAglow646/Monet-7B")
    os.environ["LATENT_SIZE"] = "10"
    
    print("正在初始化 vLLM...")

    # mllm, sampling_params = vllm_mllm_init(model_path, tp=1, gpu_memory_utilization=0.9, max_model_len=16384)
    # Visual Transformer (Qwen2.5) 需要 num_heads (16) 能够被 tp 整除。因此 tp 只能选能被 16 整除的并发数 (如 1, 2, 4)。
    # mllm, sampling_params = vllm_mllm_init(model_path, tp=4, gpu_memory_utilization=0.9, max_model_len=16384)
    # DONE：在测试过程中发现这个模型在 vLLM 中只能单卡运行，否则会有各种奇怪的错误（如显存占用异常、输出异常等）。因此这里改为单卡并适当降低显存利用率以保证稳定。
    #  inference/patch_vllm.sh 给 vLLM 打补丁（只需执行一次，补丁会备份原文件）。补丁会让 vLLM 支持 Monet 的模型和推理方式。
    # FIXED：运行前请先执行 inference/patch_vllm.sh 给 vLLM 打补丁（只需执行一次，补丁会备份原文件）。补丁会让 vLLM 支持 Monet 的模型和推理方式。
    mllm, sampling_params = vllm_mllm_init(model_path, tp=4, gpu_memory_utilization=0.8, max_model_len=16384)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 基础目录
    base_dir = "/home/xiaojunhao/m-x/data/Monet-SFT-125K"
    
    # 1. 评估 Zebra_CoT_geometry 数据集 (小样本测试)
    evaluate_dataset(
        os.path.join(base_dir, "Zebra_CoT_geometry/train.json"),
        "/home/xiaojunhao/m-x/inference/results/Zebra_CoT_geometry_results.csv",
        mllm, sampling_params, processor, base_dir
    )

if __name__ == '__main__':
    main()
