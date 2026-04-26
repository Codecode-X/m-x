"""
本文件的作用：批量 API 判断模块（多线程并发版）

在 RL 训练的评分环节，当规则判断认为答案"错误"时，
本模块调用外部大模型 API（Gemini/DeepSeek）来进行二次验证：
- 有些答案规则无法判断等价（例如 "two" vs "2"），需要语义理解
- 有些回答格式特殊，规则提取失败，API 可以整体理解

本文件提供两个核心函数：
1. judge_wrap_fn：构建评判 prompt（支持普通模式和重复惩罚模式）
2. api_batch_judge：多线程并发批量调用 API，高效处理整个 batch

工作原理：
    对 batch 中每个"规则判断失败"的样本，用 ThreadPoolExecutor 并发调用 API
    → 最多重试 5 次，超时/失败则视为不正确（0 分）
    → 如果开启 repetition_penalty，还会判断回答是否存在无意义重复（给 -1 分）
"""

# 导入类型注解工具
from typing import List, Optional, Dict, Tuple
# 导入 API 调用统一入口（Gemini/DeepSeek 的封装）
from tools.custom_api import get_api_response
# 导入异常追踪工具（打印完整的错误调用栈）
import traceback
# 导入时间工具（用于计时和统计 API 调用耗时）
import time
# 调试工具（实际训练时不使用）
import pdb


def judge_wrap_fn(pred: Optional[str], gt: Optional[str], question: Optional[str],
                  repetition_penalty: bool = False) -> Tuple[str, str]:
    """
    构建 API 评判所需的 sys_prompt 和 user_prompt。
    
    支持两种模式：
    
    模式1（repetition_penalty=False，普通模式）：
    - 只判断答案是否正确
    - API 回复：'yes'（正确）或 'no'（错误）
    
    模式2（repetition_penalty=True，重复惩罚模式）：
    - 判断答案是否正确（1 分）
    - 如果错误，还需判断是否包含无意义重复内容（-1 分）
    - 正常的错误答案给 0 分，重复性错误给 -1 分
    
    参数：
    - pred：模型的预测答案
    - gt：标准答案
    - question：题目文本
    - repetition_penalty：是否开启重复惩罚模式
    
    返回：
    - (sys_prompt, user_prompt)：系统提示和用户消息的元组
    """
    
    if not repetition_penalty:
        # ── 普通模式：只判断对错 ──
        
        # 系统提示：告诉 API 它是一个严格的答案判断者
        sys_prompt = (
            "You are a strict answer judge. Given the question, a model's predicted answer, and the ground-truth answer, "
            "determine if the prediction is correct. Consider semantic equivalence, case/format variations, "
            "and numeric equivalence if applicable. Only reply with 'yes' or 'no'."
        )
        # 用户消息：提供题目、预测答案、标准答案，要求 API 判断
        user_prompt = (
            f"Question: {question if question is not None else ''}\n"
            f"Predicted Answer: {pred if pred is not None else ''}\n"
            f"Ground Truth Answer: {gt if gt is not None else ''}\n"
            "Does the predicted answer exactly or semantically match the ground-truth? Reply 'yes' or 'no'."
        )
    
    else:
        # ── 重复惩罚模式：判断对错 + 是否有无意义重复 ──
        
        # 系统提示：包含三个输出选项（1/0/-1）和两个重复示例
        sys_prompt = (
            "You are a strict answer judge. Given the question, a model's predicted answer, and the ground-truth answer, you should:\n"
            "1. Determine if the prediction is correct. Consider semantic equivalence, case/format variations, "
            "and numeric equivalence if applicable. If the prediction is correct, reply with '1'.\n"
            
            # 如果答案错误，还需判断是否有无意义重复：
            "2. If the prediction is incorrect, then determine if the prediction contains repeatedly illogical contents. Here are two examples:\n"
            
            # 重复示例1：同一句话反复出现（模型陷入循环）
            "Example (1) 'First, observe the pattern in the top row of the image.  The pattern in the top row is  increasing by one row each time.  The pattern in the bottom row is  increasing by one column each time.  The pattern in the bottom row is  increasing by one column each time.  The pattern in the bottom row is  increasing by one column each time. ...'\n"
            
            # 重复示例2：\boxed{} 内容不断重复变化（答案来回跳变）
            "Example (2) 'First, observe the pattern in the top row of the provided image.  The pattern in the top row is  \boxed{A}.  The pattern in the bottom row is  \boxed{D}.  The pattern in the middle row is  \boxed{B}.  The pattern in the bottom row is  \boxed{C}.  The pattern in the middle row is  \boxed{A}.  The pattern in the bottom row is  \boxed{D}.  The pattern in the middle row is  \boxed{B}. ...'\n"
            
            # 判断规则：有重复给 -1，无重复给 0
            "If the prediction doesn't contain such contents, reply with '0'. Else, reply with '-1'.\n"
            "Remember, you are only allowed to output '1', '0', or '-1', do not output anything else."
        )
        
        # 用户消息：提供题目、预测答案、标准答案
        user_prompt = (
            f"Question: {question if question is not None else ''}\n"
            f"Predicted Answer: {pred if pred is not None else ''}\n"
            f"Ground Truth Answer: {gt if gt is not None else ''}\n"
            "Your output: "
        )
    
    return sys_prompt, user_prompt


def _api_call_wrapper(
    api_name: str,
    pred: Optional[str],
    gt: Optional[str],
    question: Optional[str],
    dataset_name: str,
    client=None,
    api_kwargs: Optional[dict] = None,
    repetition_penalty: bool = False
) -> Optional[bool]:
    """
    对单个样本执行 API 判断，有重试机制（最多 5 次）。
    
    返回值含义：
    - 1.0：API 判断正确
    - 0.0：API 判断错误（或 API 无法判断，默认为错误）
    - -1.0：触发重复惩罚（repetition_penalty 模式下，回答有大量无意义重复）
    - None：所有重试都失败（调用方会把它当 0 处理）
    
    参数：
    - api_name：使用的 API 模型名称
    - pred：模型预测答案
    - gt：标准答案
    - question：题目文本
    - dataset_name：数据集名称（用于日志追踪）
    - client：API 客户端（可复用，避免反复建连）
    - api_kwargs：额外的 API 调用参数
    - repetition_penalty：是否使用重复惩罚模式
    
    返回：1.0 / 0.0 / -1.0 / None
    """
    
    # 快速过滤：如果预测或标准答案为空，直接判断为错误（不浪费 API 调用额度）
    if pred is None or gt is None or str(pred).strip() == "":
        return False
    
    # 最多尝试 5 次（防止临时性网络故障导致全部失败）
    attempts = 5
    for atpt in range(attempts):
        try:
            # 根据当前样本构建评判 prompt
            sys_prompt, user_prompt = judge_wrap_fn(pred, gt, question, repetition_penalty)
            
            # 调用外部 API（通过 custom_api.py 的统一接口）
            # 注意：这里只传一个 prompt（列表长度为 1），因为是单个样本
            responses = get_api_response(
                api_name, sys_prompt, [user_prompt],
                client=client, **(api_kwargs or {})
            )
            
            # 验证返回的响应是否有效（非空字符串）
            if responses and isinstance(responses[0], str) and responses[0].strip():
                t = responses[0].strip().lower()  # 统一转小写方便匹配
                
                if not repetition_penalty:
                    # ── 普通模式：解析 yes/no ──
                    if "yes" in t and "no" not in t:
                        # 明确说 yes，判断正确
                        return 1.0
                    if "no" in t and "yes" not in t:
                        # 明确说 no，判断错误
                        return 0.0
                    # yes 和 no 都出现，或都没出现 → 模糊响应，视为错误
                    print(f"Neither 'yes' nor 'no' in the API response. Will set the judgment to be incorrect. The API response is: {responses[0]}")
                    return 0.0
                
                else:
                    # ── 重复惩罚模式：解析 1/0/-1 ──
                    if "1" in t and "0" not in t and "-1" not in t:
                        # 只有 1，判断正确
                        return 1.0
                    elif "0" in t and "1" not in t and "-1" not in t:
                        # 只有 0，判断错误（普通错误）
                        return 0.0
                    elif "-1" in t and "0" not in t:
                        # 触发重复惩罚：打印前 1500 个字符帮助调试
                        pred_partial = pred[:1500]
                        print(f"[Repetitive pred]={pred_partial}...")
                        return -1.0
                    # 格式不规范，视为普通错误
                    print(f"Invalid API response. Will set the judgment to be incorrect. The API response is: {responses[0]}")
                    return 0.0
            
            # 响应为空或无效：打印提示并重试
            print(f"Failed to obtain valid API judgement. Will retry for the {atpt + 1} time...")
            if isinstance(responses[0], str):
                print(f"The API response is: {responses[0]}")
            continue  # 继续下一次重试
        
        except Exception as e:
            # 打印完整异常堆栈（便于调试），然后继续重试
            traceback.print_exc()
            print(f"API judge error: {e}")
            continue
    
    # 所有重试都失败了：返回 None，让调用方决定如何处理
    return None


def _strip_boxed_instruction(q: str) -> str:
    """
    从题目文本中去掉"请把答案写在 \\boxed{} 里"等格式要求的指令。
    
    目的：去掉这些指令后，题目内容更纯粹，API 判断时不会被格式要求干扰。
    
    参数：q - 原始题目文本
    返回：去掉格式指令后的题目文本
    """
    if not isinstance(q, str):
        # 非字符串类型直接返回（防御性处理）
        return q
    return (
        q.replace("Put the letter of your choice within \\boxed{}.", "")
         .replace("Put your final answer within \\boxed{}.", "")
         .replace("Given the answer in a single word and put it within \\boxed{}.", "")
         .strip()  # 去掉首尾空白
    )


def api_batch_judge(
    questions: List[Optional[str]],
    preds: List[Optional[str]],
    gts: List[Optional[str]],
    *,
    api_name: Optional[str] = 'gemini-2.5-pro',   # 默认使用 Gemini 2.5 Pro
    api_max_workers: int = 32,                      # 默认最多 32 个并发线程
    api_kwargs: Optional[Dict] = None,              # 额外 API 参数（如 temperature）
    client=None,                                    # 可复用的 API 客户端
    dataset_name: str = "",                         # 数据集名称（用于日志）
    repetition_penalty: bool = False                # 是否开启重复惩罚
) -> List[int]:
    """
    多线程并发批量 API 评判。
    
    对一批（question, pred, gt）三元组，并发地调用 API 判断每个预测是否正确。
    这是 rule_then_api_batch_judge 的第二阶段（只处理规则判断失败的样本）。
    
    并发策略：
    - 使用 ThreadPoolExecutor（线程池），因为 API 调用是 I/O 密集型任务
    - 最大并发数默认 32，可通过环境变量 API_JUDGE_WORKERS 覆盖
    - 每个任务独立重试，不影响其他任务
    
    参数：
    - questions：题目列表（可能含 None）
    - preds：预测答案列表（可能含 None）
    - gts：标准答案列表（可能含 None）
    - api_name：API 模型名称（"gemini-2.5-pro" / "deepseek-chat" 等）
    - api_max_workers：最大并发线程数
    - api_kwargs：传给 API 的额外参数
    - client：预建的 API 客户端（None 时内部创建）
    - dataset_name：数据集名称（便于日志追踪）
    - repetition_penalty：是否判断重复性错误（True 时给 -1 分）
    
    返回：
    - results：每个样本的正确性列表（1.0/0.0/-1.0），与输入列表等长
    """
    
    # 局部导入（避免循环依赖）
    import os
    import concurrent.futures as cf
    import traceback
    
    # 记录开始时间（用于最后统计总耗时）
    start_time = time.time()
    
    # 检查输入列表长度是否一致
    if not (len(questions) == len(preds) == len(gts)):
        raise ValueError("Length mismatch: `questions`, `preds`, and `gts` must have the same length.")
    
    # 总样本数
    n = len(preds)
    
    # 初始化结果列表（默认全部为 0，即"不正确"）
    results: List[int] = [0] * n
    
    # 预处理题目文本：去掉 boxed 格式指令，让 API 更容易理解题目本意
    try:
        questions_wo_inst = [_strip_boxed_instruction(q) for q in questions]
    except NameError:
        # 如果 _strip_boxed_instruction 函数不可用，直接用原始题目
        questions_wo_inst = questions
    
    # 从环境变量读取最大并发数（允许在不改代码的情况下调整并发量）
    try:
        max_workers = int(os.environ.get("API_JUDGE_WORKERS", api_max_workers))
    except Exception:
        max_workers = api_max_workers
    
    # ── 多线程并发调用 API ──
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        
        # 为每个样本提交一个独立的 API 调用任务（并发）
        for i in range(n):
            fut = ex.submit(
                _api_call_wrapper,               # 每个线程执行的函数
                api_name,                        # API 模型名称
                preds[i],                        # 预测答案
                gts[i],                          # 标准答案
                questions_wo_inst[i],            # 去掉格式指令后的题目
                dataset_name,                    # 数据集名称
                client=client,                   # API 客户端（多线程共享，通常是线程安全的）
                api_kwargs=api_kwargs,            # 额外 API 参数
                repetition_penalty=repetition_penalty  # 重复惩罚开关
            )
            # 保存 (样本索引, Future对象) 对，用于后续收集结果
            futs.append((i, fut))
        
        # 等待并收集所有 API 调用的结果
        for i, fut in futs:
            try:
                # fut.result() 会阻塞直到该任务完成，并返回 _api_call_wrapper 的返回值
                r = fut.result()
                # 把结果存入对应位置（None 表示所有重试都失败，视为 0）
                results[i] = r
            except Exception:
                # 理论上 _api_call_wrapper 已经处理了所有异常，这里只是双重保险
                traceback.print_exc()
                print("WARNING: API judge fail, set the correctness to be 0")
                results[i] = 0
    
    # 统计并打印总耗时
    end_time = time.time()
    print(f"[api_batch_judge] Completed {n} samples in {end_time - start_time:.2f} seconds using API '{api_name}'")
    
    return results
