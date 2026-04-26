"""
本文件的作用：封装 API 调用接口，支持 Gemini 和 DeepSeek 两种外部大模型 API。

在 RL 训练中，当规则判断无法确定答案是否正确时，需要调用外部大模型（Gemini 或 DeepSeek）
来做语义级别的判断（例如："a cute bear" 和 "teddy bear" 是否等价？）。

本文件提供统一的接口：
1. build_gemini_client()：创建 Gemini API 客户端
2. get_gemini_response()：批量调用 Gemini API 获取回复
3. build_deepseek_client()：创建 DeepSeek API 客户端
4. get_deepseek_response()：批量调用 DeepSeek API 获取回复
5. get_api_response()：统一入口，根据模型名称分派到对应的 API

使用前提：
- 需要在环境变量中设置 GOOGLE_API_KEY（Gemini）或 DEEPSEEK_API_KEY（DeepSeek）
- 需要安装：`pip install openai google-generativeai tqdm`
"""

# 注释：使用前请安装 OpenAI SDK：pip3 install openai
# （OpenAI SDK 兼容 DeepSeek API，所以这里用 OpenAI 客户端访问 DeepSeek）

# 导入 OpenAI 客户端（用于 DeepSeek，因为 DeepSeek API 兼容 OpenAI 格式）
from openai import OpenAI
# 导入进度条工具（此处有注释掉的使用，遗留导入）
from tqdm import tqdm
# 导入操作系统接口，用于读取环境变量（API Key）
import os
# 导入 JSON 处理工具（此处未直接使用，遗留导入）
import json
# 导入 Google Generative AI SDK（用于访问 Gemini API）
from google import genai
# 导入日志模块（用于静默第三方库的日志输出）
import logging

# 静默相关第三方库的详细日志（避免 API 调用时打印大量 HTTP 请求日志）
for name in ["openai", "openai._client", "httpx", "httpcore"]:
    logging.getLogger(name).setLevel(logging.WARNING)


# Gemini API 的默认生成配置
gemini_generation_config = {
    "max_output_tokens": 9000,  # 最大输出 9000 个 token
    "temperature": 0.3,         # 较低温度，输出更确定（适合判断任务）
    "top_p": 1.0                # Top-P 为 1.0，不做核采样截断
}


def build_gemini_client():
    """
    创建 Gemini API 客户端。
    
    从环境变量 GOOGLE_API_KEY 读取 API Key，
    这是 Google AI Studio / Vertex AI 的 Gemini 密钥。
    
    使用方法：
    ```bash
    export GOOGLE_API_KEY="your-key-here"
    ```
    
    返回：
    - genai.Client 对象（可用于调用 Gemini 模型）
    
    异常：
    - ValueError：如果环境变量未设置
    """
    # 从环境变量读取 API Key
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # 如果没有设置，抛出有明确提示的错误
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in environment.")
    
    # 用 API Key 创建 Gemini 客户端
    client = genai.Client(api_key=api_key)
    return client


def get_gemini_response(client, sys_prompt, user_prompts, temperature=0.3, model_name="gemini-2.5-pro-exp-02"):
    """
    批量调用 Gemini API，获取多个 prompt 的回复。
    
    注意：此函数是串行调用（for 循环），适合小批量使用。
    大批量时应使用 api_judge.py 里的多线程版本。
    
    参数：
    - client：build_gemini_client() 返回的客户端对象
    - sys_prompt：系统提示（角色设定，如 "You are a judge..."）
    - user_prompts：用户消息列表（每个元素是一个字符串，对应一个评判任务）
    - temperature：生成温度（默认 0.3，适合判断任务）
    - model_name：使用的 Gemini 模型名（默认 gemini-2.5-pro-exp-02）
    
    返回：
    - responses：回复文本列表，与 user_prompts 一一对应
    """
    responses = []
    
    # 遍历每个 prompt（串行调用）
    for user_prompt in user_prompts:
        # 构造 Gemini API 要求的消息格式
        # Gemini 把 system prompt 和 user prompt 合并为一条用户消息（不支持独立的 system role）
        contents = [
            {"role": "user", "parts": [{"text": sys_prompt + "\n" + user_prompt}]}
        ]
        
        # 复制默认生成配置，避免修改全局配置
        gen_cfg = dict(gemini_generation_config)
        # 用传入的 temperature 覆盖默认值
        gen_cfg["temperature"] = temperature
        
        try:
            # 调用 Gemini API 生成回复
            response = client.models.generate_content(
                model=model_name,               # 模型名称
                contents=contents,              # 输入内容
                generation_config={
                    "max_output_tokens": gen_cfg["max_output_tokens"],
                    "temperature": gen_cfg["temperature"],
                    "top_p": gen_cfg["top_p"],
                },
            )
            
            # 从响应中提取文本内容（拼接所有 parts 的文本）
            # response.candidates[0] 是最佳候选回复
            # .content.parts 是回复的各个文本片段（通常只有一个）
            text = "".join(part.text for part in response.candidates[0].content.parts)
        
        except Exception as e:
            # API 调用失败时，返回带错误标记的字符串（而不是崩溃）
            # 这样上层代码可以识别并跳过这个失败的结果
            text = f"[API calling error, at tools.custom_api.py:get_gemini_response] {str(e)}"
        
        responses.append(text)
    
    return responses


def build_deepseek_client():
    """
    创建 DeepSeek API 客户端。
    
    DeepSeek 使用与 OpenAI 兼容的 API 格式，
    所以可以直接用 OpenAI SDK，只需修改 base_url 和 api_key。
    
    使用方法：
    ```bash
    export DEEPSEEK_API_KEY="your-key-here"
    ```
    
    返回：
    - OpenAI 客户端对象（已配置 DeepSeek 的 URL 和 API Key）
    """
    # 从环境变量读取 DeepSeek API Key（如果没有设置则为 None，API 调用会失败）
    return OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY', None),  # DeepSeek API Key
        base_url="https://api.deepseek.com"                 # DeepSeek API 的 base URL
    )


def get_deepseek_response(client, sys_prompt, user_prompts, temperature=0.3, model_name="deepseek-chat"):
    """
    批量调用 DeepSeek API，获取多个 prompt 的回复。
    
    参数：
    - client：build_deepseek_client() 返回的客户端
    - sys_prompt：系统提示（告诉模型它是评判助手）
    - user_prompts：用户消息列表
    - temperature：生成温度（默认 0.3）
    - model_name：DeepSeek 模型名称（"deepseek-chat" 或 "deepseek-reasoner"）
    
    返回：
    - responses：回复文本列表，与 user_prompts 一一对应
    """
    model = model_name
    responses = []
    
    try:
        # 遍历每个 prompt 串行调用
        # （注：原代码有一段被注释掉的 tqdm 进度条版本，实际不使用）
        for user_prompt in user_prompts:
            response = client.chat.completions.create(
                model=model,               # 模型名称
                messages=[
                    {"role": "system", "content": sys_prompt},   # 系统提示
                    {"role": "user", "content": user_prompt},    # 用户消息
                ],
                temperature=temperature,   # 生成温度
                stream=False,              # 不使用流式输出
            )
            # 提取第一个候选回复的文本
            responses.append(response.choices[0].message.content)
    
    except Exception as e:
        # 打印错误信息（但不崩溃，返回已有的部分结果）
        print(f"Deepseek API judge error: {e}")
    
    return responses


def get_api_response(api_model_name, sys_prompt, user_prompts, client=None, temperature=0.3):
    """
    统一的 API 调用入口：根据模型名称，分派到对应的 API 函数。
    
    这是对外暴露的统一接口，调用方不需要关心用的是 Gemini 还是 DeepSeek。
    
    参数：
    - api_model_name：模型名称，支持：
        - "gemini-2.5-pro"：使用 Google Gemini API
        - "deepseek-chat"：使用 DeepSeek 聊天模型
        - "deepseek-reasoner"：使用 DeepSeek 推理模型
    - sys_prompt：系统提示
    - user_prompts：用户消息列表
    - client：可选，如果已有客户端对象就复用（避免重复创建），否则内部新建
    - temperature：生成温度（默认 0.3）
    
    返回：
    - 回复文本列表
    
    异常：
    - ValueError：如果 api_model_name 不在支持列表中
    """
    
    if api_model_name == "gemini-2.5-pro":
        # 使用 Gemini API
        if client is None:
            # 如果没有传入已有客户端，新建一个
            client = build_gemini_client()
        return get_gemini_response(client, sys_prompt, user_prompts, temperature)
    
    elif api_model_name in ["deepseek-chat", "deepseek-reasoner"]:
        # 使用 DeepSeek API
        if client is None:
            client = build_deepseek_client()
        return get_deepseek_response(client, sys_prompt, user_prompts, temperature, model_name=api_model_name)
    
    else:
        # 不支持的模型名称
        raise ValueError(f"Unsupported API model name: {api_model_name}")
