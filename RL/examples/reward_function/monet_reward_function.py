"""
本文件的作用：定义 RL 训练中使用的奖励函数（Reward Function）。

在 GRPO/VLPO 强化学习训练中，每当模型生成一个回答（rollout），
就需要计算这个回答的"得分"作为奖励信号，指导模型往更好的方向更新参数。

本文件定义了三类奖励信号和两个对外暴露的评分函数：

━━━ 三类奖励信号 ━━━
1. 格式奖励（format_reward）：回答是否包含 \boxed{} 格式（合规性）
2. 准确性奖励（accuracy_reward）：回答是否正确（最重要）
3. Latent 使用奖励（use_latent_reward）：回答是否使用了 latent 推理（激励模型用视觉思维）

━━━ 两个对外评分函数 ━━━
1. compute_score：规则判断答案正确性，用于 RL 训练的主奖励函数
2. compute_score_w_prev_correctness：使用预先判断好的正确性结果，结合格式和长度惩罚

━━━ 判断流程 ━━━
- 首先用规则（正则表达式 + mathruler 库）判断答案是否正确
- 规则判断不了的（模糊答案、需要语义理解的），调用外部 API（Gemini/DeepSeek）判断
- rule_then_api_batch_judge 函数实现了这个"先规则、再 API"的两阶段判断逻辑
"""

# 导入正则表达式库（用于提取答案、清理格式）
import re
# 导入类型注解工具
from typing import Dict, List, Union, Optional
# 导入数值计算库
import numpy as np
# 从 mathruler 库导入两个工具函数：
# - extract_boxed_content：从 \boxed{...} 中提取答案文本
# - grade_answer：判断预测答案和标准答案是否等价
from mathruler.grader import extract_boxed_content, grade_answer
# 导入答案格式转换函数（把各种格式的答案统一化，如分数/百分比/英文数字等）
from examples.reward_function.answer_transformation import answer_transformation_fn
# 导入从非 boxed 格式输出中提取答案的工具函数
from verl.workers.rollout.utils.util import extract_no_boxed_answer
# 重复导入 re（已在上面导入，这里是遗留代码）
import re
# 导入 PyTorch（用于张量计算，主要用于批量计算长度惩罚）
import torch
# 导入批量 API 评判函数（调用 Gemini/DeepSeek 大模型来判断答案是否正确）
from tools.api_judge import api_batch_judge
# 调试工具（实际训练时注释掉）
import pdb

####################################################################
# 规则判断函数（不需要调 API，速度快）
####################################################################

# 预编译正则表达式：匹配 \boxed{...} 格式
# re.DOTALL 让 . 也能匹配换行符（答案内容可能有换行）
BOXED_RE = re.compile(r"\\boxed\{.*?\}", re.DOTALL)


def format_reward(predict: str):
    """
    格式奖励：检查模型回答是否包含 \boxed{} 格式。
    
    RL 训练要求模型把最终答案写在 \boxed{} 里，否则给 0 分惩罚。
    
    参数：predict - 模型生成的回答文本
    返回：1.0（有 boxed 格式）或 0.0（没有）
    """
    # 如果没有找到 \boxed{...}，返回 0 分
    if not BOXED_RE.search(predict):
        return 0.0
    # 有 boxed 格式，返回 1 分
    return 1.0


def use_latent_reward(predict: str):
    """
    Latent 使用奖励：检查模型回答是否使用了 latent 推理（即有 <abs_vis_token> 标记）。
    
    这个奖励信号激励模型在推理时主动使用视觉 latent 思维，
    而不是仅靠文字推理得出答案。
    
    参数：predict - 模型生成的回答文本
    返回：1.0（使用了 latent 推理）或 0.0（没有使用）
    """
    # 如果回答中包含 <abs_vis_token> 标记，说明模型使用了 latent 推理
    if "<abs_vis_token>" in predict:
        return 1.0
    return 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    """
    准确性奖励：检查预测答案是否与标准答案等价。
    
    参数：
    - predict：模型生成的回答
    - ground_truth：标准答案
    
    返回：1.0（正确）或 0.0（错误）
    """
    # 调用 extract_and_check 做规则判断
    return 1.0 if extract_and_check(predict, ground_truth) else 0.0


def extract_and_check(predict: str, ground_truth: str) -> float:
    """
    从预测输出中提取答案，并与标准答案做等价性比较。
    
    提取策略（依次尝试）：
    1. 优先从 \boxed{...} 中提取答案
    2. 如果没有 boxed，用 extract_no_boxed_answer 从普通文本中提取
    然后用 mathruler 的 grade_answer 做语义等价判断（支持数学表达式等价）
    
    参数：
    - predict：模型回答
    - ground_truth：标准答案
    
    返回：True（正确）或 False（错误）
    """
    # 从 \boxed{} 中提取答案
    answer = extract_boxed_content(predict)
    
    # 如果没有 boxed（返回字符串 'None'），尝试从普通文本中提取
    if answer == 'None':
        answer = extract_no_boxed_answer(predict)
    
    # 用 mathruler 判断提取的答案是否与标准答案等价
    # grade_answer 支持：数字、分数、百分比、选项字母等多种格式的等价判断
    return grade_answer(answer, ground_truth)


def compute_score(predicts: List[str], ground_truths: List[str],
                  format_weight: float = 0.1,
                  length_penalty_weight=0.001,
                  resp_lengths=None,
                  ref_resp_lengths=None) -> List[Dict[str, float]]:
    """
    批量计算 RL 训练的奖励分数（规则判断版，不调 API）。
    
    总分公式：
    overall = (1 - format_weight) × accuracy_score
            + format_weight × format_score
            - length_penalty_weight × max(0, resp_length - ref_length)
    
    参数：
    - predicts：模型生成的回答列表（batch）
    - ground_truths：标准答案列表（batch）
    - format_weight：格式奖励的权重（默认 0.1，即格式占 10%）
    - length_penalty_weight：长度惩罚系数（默认 0.001）
    - resp_lengths：每个生成回答的 token 长度（tensor）
    - ref_resp_lengths：参考长度（通常是 teacher 或历史生成的长度）
    
    返回：
    - scores：每个样本的分数字典列表，格式为 [{"overall": x, "format": y, "accuracy": z}, ...]
    """
    scores = []
    
    # 把参考长度转为 tensor 以便向量化计算
    ref_resp_lengths = torch.tensor(ref_resp_lengths)
    
    # 计算长度惩罚：只惩罚比参考长度更长的回答
    # 条件：resp_length > ref_length 且 ref_length != 0（参考长度为 0 说明没有参考，不惩罚）
    if resp_lengths is not None and ref_resp_lengths is not None:
        length_penalty = torch.where(
            torch.logical_and(resp_lengths > ref_resp_lengths, ref_resp_lengths != 0),
            resp_lengths - ref_resp_lengths,   # 惩罚值 = 超出的 token 数
            torch.zeros_like(resp_lengths)      # 未超出则惩罚为 0
        )
    else:
        # 没有长度信息时，不做长度惩罚
        length_penalty = torch.zeros(len(predicts))
    
    # 遍历 batch 里每个样本
    for i, (predict, ground_truth) in enumerate(zip(predicts, ground_truths)):
        # 清理格式：统一处理 < > / 两侧的空格（Qwen2.5-VL-32B 有时会多加空格）
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)
        
        # 计算格式奖励（有 \boxed{} 得 1 分，没有得 0 分）
        format_score = format_reward(predict)
        
        # 计算准确性奖励（规则判断）
        accuracy_score = accuracy_reward(predict, ground_truth)
        
        # 计算综合总分
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score
                         + format_weight * format_score
                         - length_penalty_weight * length_penalty[i],
                "format": format_score,      # 格式分（用于监控）
                "accuracy": accuracy_score,  # 准确率（用于监控）
            }
        )
    
    return scores


####################################################################
# API 评判函数（调用外部大模型，判断规则难以处理的模糊答案）
####################################################################


def build_prompt_mcq(question, options, prediction):
    """
    为多选题（MCQ）构建一个 API 评判的 prompt。
    
    当模型回答是模糊文字（如"a cute teddy bear"），
    需要让 API 判断它对应哪个选项（A/B/C/D）。
    
    参数：
    - question：题目文字
    - options：选项文字（如 "A. teddy bear B. rabbit C. cat D. dog"）
    - prediction：模型的回答
    
    返回：格式化好的评判 prompt 字符串
    """
    # 定义评判 prompt 模板（含3个示例供 API 参考）
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 3: \n'
        'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: '
    )
    # 用实际值填充模板第3个示例
    return tmpl.format(question, options, prediction)


# API 评判的示例 prompt（few-shot examples），用于引导 API 正确提取和判断答案
# 包含两个含 <abs_vis_token> latent 推理的示例，让 API 知道如何处理 latent 内容
demo_prompt_extract_and_judge = """
The [Standard Answer] is the correct answer to the question, and the [Model Response] is the answer generated by a model for that question. [Question] is the original question.
Thoroughly read both the [Question], [Standard Answer] and the [Model Response]. You need to:

1. Extract the answer from the [Model Response], output '[Extracted answer]: XXX'.
2. Assess the consistency of the extracted answer with the [Standard Answer] according to the [Question]. If the [Model Answer] is consistent with the [Standard Answer], please output '1'. If not, or the answer for the [Question] cannot be extrated, output '0'.

Below are some examples:
[Question]: A wedding photo of a newlywed couple in front of a castle-like building. What color are the earrings on the bride's ears?
[Standard Answer]: silver
[Model Response]: To answer the question, I need to locate the bride in the image and identify her earrings. The image is quite dark, so I will focus on the bride's face to discern any details on her ears.To get a clearer view of the bride's ears and any accessories, I will generate a zoomed-in image of that specific area.
<abs_vis_token><latent></abs_vis_token>
The zoomed-in view clearly shows the bride's face. Upon close inspection,  her left ear, which is visible in the image, is adorned with  a distinct  red earring.
[Extracted answer]: red
[Judgment]: 0

[Question]: Under the warm yellow candlelight, the two sat opposite each other. The table was piled high with books and scrolls. How many candles were there in total on the table?
[Standard Answer]: Two
[Model Response]: To answer the question, I need to carefully examine the image to locate all the candles present on the table. I will focus on the area around the table where candles might be visible.To accurately count the candles, I will generate a zoomed-in view of the area around the table where candles are typically placed to ensure clear visibility and precise counting.
<abs_vis_token><latent></abs_vis_token>
The zoomed-in image clearly shows 2 distinct candles: one on the left side of the table, one on the right side. Each candle is clearly visible and identifiable.The visual evidence from the detailed view confirms the presence of 2 candles on the table.
[Extracted answer]: 2
[Judgment]: 1

"""


def get_evaluation_chat_response(sys_prompt, user_prompt, client, temperature=0.7):
    """
    调用 DeepSeek API 获取评判结果。
    
    参数：
    - sys_prompt：系统提示（告诉 API 它的角色）
    - user_prompt：用户消息（包含问题、标准答案、模型回答）
    - client：已初始化的 API 客户端对象
    - temperature：生成温度（0.7 表示适度多样性）
    
    返回：API 的回答文本
    """
    response = client.chat.completions.create(
        model="deepseek-chat",       # 使用 DeepSeek 模型
        messages=[
            {"role": "system", "content": sys_prompt},   # 系统提示
            {"role": "user", "content": user_prompt},    # 用户输入
        ],
        max_tokens=1024,             # 最多生成 1024 个 token
        temperature=0.7,             # 生成温度
        stream=False                 # 不使用流式输出（等待完整响应）
    )
    # 返回第一个候选回答的文本内容
    return response.choices[0].message.content


def process_judgment(judgment):
    """
    解析 API 返回的判断结果，确保格式正确。
    
    API 应该返回 '0' 或 '1'（可能有前缀如 "[Judgment]: 1"）。
    如果格式不对（如返回了其他内容），视为判断失败。
    
    参数：judgment - API 返回的文本
    返回：True（API 判断正确 = 1）或 False（API 判断错误 = 0，或格式异常）
    """
    # 如果 API 没有返回任何内容，视为失败
    if judgment is None:
        return False
    
    # 清理格式：转小写，去掉可能的 "[judgment]:" 前缀和空格
    judgment = judgment.lower().replace("[judgment]:", "").strip()
    
    # 只接受 '0' 或 '1'，其他格式视为无效
    if judgment not in ['0', '1']:
        return False
    
    # 返回布尔值（'1' → True，'0' → False）
    return True


def create_test_prompt(demo_prompt, question, answer, extraction):
    """
    构建完整的 API 评判 prompt（few-shot 示例 + 当前测试样本）。
    
    参数：
    - demo_prompt：few-shot 示例部分（demo_prompt_extract_and_judge）
    - question：当前题目
    - answer：标准答案
    - extraction：模型的回答
    
    返回：完整的评判 prompt 字符串
    """
    # 清理示例部分（去掉首尾空白）
    demo_prompt = demo_prompt.strip()
    
    # 构造当前测试样本的提问部分
    test_prompt = f"[Question]: {question}\n[Standard Answer]: {answer}\n[Model Response]: {extraction}\n[Extracted answer]: "
    
    # 拼接示例和测试样本，之间空两行
    full_prompt = f"{demo_prompt}\n\n{test_prompt}"
    
    return full_prompt


def extract_and_check_api(question: str, predict: str, ground_truth: str, client, verbose=False) -> float:
    """
    用 API（DeepSeek）判断单个样本的答案是否正确。
    
    有重试机制（最多尝试 3 次），如果 API 调用全部失败，
    则回退到规则判断（extract_and_check）。
    
    参数：
    - question：题目
    - predict：模型回答
    - ground_truth：标准答案
    - client：API 客户端
    - verbose：是否打印详细错误信息
    
    返回：True（正确）或 False（错误）
    """
    # 系统提示：告诉 API 它是一个辅助评判助手
    sys_prompt = "You are a helper judge assistant."
    
    # 最多重试 3 次
    retries = 3
    for _ in range(retries):
        try:
            # 构建完整的评判 prompt
            test_prompt = create_test_prompt(demo_prompt_extract_and_judge, question, ground_truth, predict)
            # 调用 API 获取判断结果
            judgment = get_evaluation_chat_response(sys_prompt, test_prompt, client)
            # 解析并返回判断结果
            return process_judgment(judgment)
        except Exception as e:
            # 打印错误信息（调试用）
            print(e, verbose)
            print(f"Error in matching answer:\n[Standard Answer] {ground_truth}\n[Model Answer] {predict}")
    
    # 所有重试都失败了，回退到规则判断
    print("All retries failed in extract_and_check_api, fall back to rule-based judge.")
    return extract_and_check(predict, ground_truth)


def rule_then_api_batch_judge(
    questions: List[Optional[str]],
    preds: List[Optional[str]],
    gts: List[Optional[str]],
    *,
    api_name: Optional[str] = 'gemini-2.5-pro',   # 使用的 API 模型名称
    api_max_workers: int = 32,                      # API 并发调用数
    api_kwargs: Optional[Dict] = None,              # 额外的 API 调用参数
    client=None,                                    # API 客户端（如果已有）
    dataset_name: str = "",                         # 数据集名称（用于日志）
    repetition_penalty: bool = False                # 是否对重复输出做惩罚
):
    """
    两阶段批量评判：先规则，规则不确定的再调 API。
    
    这是 RL 训练中 rule_based_judge 调用的核心函数。
    
    两阶段策略的优点：
    - 大部分能规则判断的题（数字、选项字母等），不需要调 API，节省费用
    - 只有规则判断不了的（模糊文字答案），才调用 Gemini/DeepSeek
    
    参数：
    - questions：题目文本列表
    - preds：模型回答列表
    - gts：标准答案列表
    - api_name：使用的 API 模型（默认 gemini-2.5-pro）
    - api_max_workers：API 并发线程数
    - api_kwargs：额外参数（如 temperature）
    - client：预先创建的 API 客户端（None 时内部创建）
    - dataset_name：数据集名称（便于日志追踪）
    - repetition_penalty：是否开启重复惩罚（True 时重复输出给 -1 分）
    
    返回：
    - correctness_list：每个样本的正确性列表（True/False 或 -1.0 表示重复惩罚）
    """
    
    # ── 第一阶段：规则判断 ──
    correctness_list = []
    for pred, gt in zip(preds, gts):
        # 对每个样本用规则方法判断
        correctness_list.append(extract_and_check(pred, gt))
    
    # ── 第二阶段：对规则判断为"错误"的样本，再用 API 复核 ──
    # （规则判断"正确"的不需要复核，节省 API 调用）
    
    # 收集需要 API 复核的样本
    questions_api = []
    preds_api = []
    gts_api = []
    for i, correct in enumerate(correctness_list):
        if not correct:
            # 规则判断为错，加入 API 复核队列
            questions_api.append(questions[i])
            preds_api.append(preds[i])
            gts_api.append(gts[i])
    
    # 如果有需要 API 复核的样本，批量调用 API
    if len(preds_api) > 0:
        api_correctness_list = api_batch_judge(
            questions_api,
            preds_api,
            gts_api,
            api_name=api_name,
            api_max_workers=api_max_workers,
            api_kwargs=api_kwargs,
            client=client,
            repetition_penalty=repetition_penalty
        )
        
        # 把 API 的判断结果回填到对应位置
        idx = 0  # API 结果列表的指针
        for i in range(len(correctness_list)):
            if not correctness_list[i]:
                # 这个位置是规则判断为错的，用 API 结果覆盖
                if api_correctness_list[idx] is not None:
                    correctness_list[i] = api_correctness_list[idx]
                idx += 1
    
    return correctness_list


def compute_score_w_prev_correctness(predicts: List[str],
                                     correctness_list: List[float],
                                     format_weight: float = 0.1,
                                     length_penalty_weight=0.001,
                                     resp_lengths=None,
                                     ref_resp_lengths=None) -> List[Dict[str, float]]:
    """
    使用预先计算好的正确性列表，批量计算最终奖励分数（RL 训练主奖励函数）。
    
    与 compute_score 的区别：
    - compute_score 内部自己做规则判断
    - 本函数接收外部传入的 correctness_list（通常来自 rule_then_api_batch_judge 的结果）
    
    总分公式：
    overall = (1 - format_weight) × accuracy_score
            + format_weight × format_score
            - length_penalty_weight × max(0, resp_length - ref_length)
    
    参数：
    - predicts：模型生成的回答列表（用于计算格式分）
    - correctness_list：预先判断好的正确性列表（1.0=正确, 0.0=错误, -1.0=重复惩罚）
    - format_weight：格式分权重（默认 0.1）
    - length_penalty_weight：长度惩罚系数（默认 0.001）
    - resp_lengths：生成回答的 token 长度
    - ref_resp_lengths：参考长度（用于计算长度惩罚）
    
    返回：
    - scores：每个样本的分数字典列表
    """
    scores = []
    
    # 把参考长度转为 tensor
    ref_resp_lengths = torch.tensor(ref_resp_lengths)
    
    # 计算长度惩罚（只惩罚比参考更长的回答）
    if resp_lengths is not None and ref_resp_lengths is not None:
        length_penalty = torch.where(
            torch.logical_and(resp_lengths > ref_resp_lengths, ref_resp_lengths != 0),
            resp_lengths - ref_resp_lengths,
            torch.zeros_like(resp_lengths)
        )
    else:
        length_penalty = torch.zeros(len(predicts))
    
    # 遍历每个样本
    for i, (predict, correctness) in enumerate(zip(predicts, correctness_list)):
        # 清理格式：去掉 < > / 两侧的多余空格
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)
        
        # 计算格式分
        format_score = format_reward(predict)
        
        # 根据 correctness（预先判断的正确性）设置准确性分数
        if correctness == 1.0:
            # 回答正确：准确性分 = 1.0
            # （注意：这里原本有"如果用了 latent 给额外奖励"的设计，但两个分支都是 1.0，目前未区分）
            if use_latent_reward(predict):
                accuracy_score = 1.0
            else:
                accuracy_score = 1.0
        else:
            # 回答错误（0.0）或触发重复惩罚（-1.0）：直接用 correctness 值作为准确性分
            # -1.0 表示模型输出了大量重复无意义内容，给予负分惩罚
            accuracy_score = correctness
        
        # 计算综合总分
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score
                         + format_weight * format_score
                         - length_penalty_weight * length_penalty[i],
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
    
    return scores
