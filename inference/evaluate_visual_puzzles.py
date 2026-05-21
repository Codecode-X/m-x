import os
import csv
import re
from datasets import load_dataset
import PIL.Image

# export http_proxy=http://oversea-squid2.ko.txyun:11080 https_proxy=http://oversea-squid2.ko.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
# source /home/xiaojunhao/miniconda3/etc/profile.d/conda.sh && conda activate monet && export LATENT_SIZE=10 && CUDA_VISIBLE_DEVICES=4,5,6,7 python -m inference.evaluate_visual_puzzles

import inference.apply_vllm_monet
from inference.load_and_gen_vllm import *

def replace_abs_vis_token_content(s: str) -> str:
    pattern = re.compile(r'(<abs_vis_token>)(.*?)(</abs_vis_token>)', flags=re.DOTALL)
    return pattern.sub(r'\1<latent>\3', s)

def extract_boxed_value(text: str) -> str:
    pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return ""

def evaluate_dataset(output_csv_path, mllm, sampling_params, processor):
    print("Loading neulab/VisualPuzzles dataset...")
    # Load the dataset
    ds = load_dataset("neulab/VisualPuzzles", split="train")

    # # 小样本测试模式：仅测试前 5 个样本
    # print("!!! 小样本测试模式：仅测试前 5 个样本 !!!")
    # ds = ds.select(range(5))

    conversations = []
    questions = []
    image_paths = []
    ground_truths = []

    tmp_image_dir = "tmp_visual_puzzles_images"
    os.makedirs(tmp_image_dir, exist_ok=True)
    
    for idx, item in enumerate(ds):
        image = item['image']
        image_path = os.path.join(tmp_image_dir, f"img_{idx}.png")
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(image_path)

        question = item['question']
        options = item.get('options')
        if options:
            options_str = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(options)])
            full_question = f"{question}\nOptions:\n{options_str}\nPlease output the final answer within \\boxed{{}}."
        else:
            full_question = f"{question}\nPlease output the final answer within \\boxed{{}}."

        conversations.append([
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful multimodal assistant. You are required to answer the question based on the image provided. Put your final answer in \\boxed{}."}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": PIL.Image.open(image_path).convert("RGB")
                    },
                    {
                        "type": "text",
                        "text": full_question
                    }
                ]
            }
        ])

        # print(f"Processed sample {idx+1}/{len(ds)}: \n{full_question}\n with image saved at \n{image_path}\n")
        # exit(0)

        questions.append(full_question)
        image_paths.append(image_path)
        ground_truths.append(str(item.get('answer', '')))

    print(f"============================================================")
    print(f"正在处理 {len(conversations)} 个样本: VisualPuzzles...")
    print(f"============================================================")
    
    start_idx = 0
    if os.path.exists(output_csv_path):
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            try:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) > 1:
                    start_idx = len(rows) - 1
                    print(f"发现已存在的结果文件，已处理 {start_idx} 个样本，将从这之后继续评估...")
            except Exception as e:
                print(f"读取已有 CSV 失败: {e}，将重新开始。")

    if start_idx == 0:
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "Image Path", "Triggered latent CoT", "Latent CoT Count", "Model Output (raw)", "Model Output (cleaned)", "Model Prediction (boxed)", "Ground Truth (raw)", "Correct Prediction (0/1)"])
        
    batch_size = 100

    for i in range(start_idx, len(conversations), batch_size):
        batch_convs = conversations[i:i+batch_size]
        batch_questions = questions[i:i+batch_size]
        batch_images = image_paths[i:i+batch_size]
        batch_gts = ground_truths[i:i+batch_size]

        print(f"Processing Batch {i}/{len(conversations)}...")
        inputs = vllm_mllm_process_batch_from_messages(batch_convs, processor)
        outputs = vllm_generate(inputs, sampling_params, mllm)

        batch_raw_outputs = [out.outputs[0].text for out in outputs]
        
        with open(output_csv_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            
            for q, p, out_text, gt in zip(batch_questions, batch_images, batch_raw_outputs, batch_gts):
                cleaned = replace_abs_vis_token_content(out_text)
                count = cleaned.count("<latent>")
                triggered = "Yes" if count > 0 else "No"
                
                pred_val = extract_boxed_value(cleaned)
                
                is_correct = 0
                if pred_val and gt:
                    if pred_val.lower() == gt.lower() or pred_val.lower() in gt.lower() or gt.lower() in pred_val.lower():
                        is_correct = 1
                        
                writer.writerow([q, p, triggered, count, out_text, cleaned, pred_val, gt, is_correct])
            
    print(f"✅ 结果已保存至 {output_csv_path}\n")

def main():
    model_path = os.environ.get("MONET_MODEL_PATH", "NOVAglow646/Monet-7B")
    os.environ["LATENT_SIZE"] = "10"
    
    print("正在初始化 vLLM...")
    mllm, sampling_params = vllm_mllm_init(model_path, tp=4, gpu_memory_utilization=0.8, max_model_len=16384)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    output_csv = "/home/xiaojunhao/m-x/inference/results/visual_puzzles_results.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    evaluate_dataset(output_csv, mllm, sampling_params, processor)

if __name__ == '__main__':
    main()
