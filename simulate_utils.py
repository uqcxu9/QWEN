QWEN
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import seaborn as sns
import re
import os
import multiprocessing
import scipy
import json

save_path = '/data/QWEN/ACL24-EconAgent/results/'

brackets = list(np.array([0, 97, 394.75, 842, 1607.25, 2041, 5103])*100/12)
quantiles = [0, 0.25, 0.5, 0.75, 1.0]

from datetime import datetime
world_start_time = datetime.strptime('2001.01', '%Y.%m')

prompt_cost_1k, completion_cost_1k = 0.001, 0.002

def prettify_document(document: str) -> str:
    # Remove sequences of whitespace characters (including newlines)
    cleaned = re.sub(r'\s+', ' ', document).strip()
    return cleaned


def get_completion(dialogs, temperature=0, max_tokens=100):
    import time
    import torch
    
    model_path = "/data/models/Qwen3-8B"
    
    # ===== 使用 Transformers =====
    if not hasattr(get_completion, '_model'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("首次加载 Qwen 模型（使用 Transformers）...")
        
        get_completion._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        get_completion._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        print("Transformers 模型加载完成！")
    
    tokenizer = get_completion._tokenizer
    model = get_completion._model
    
    max_retries = 20
    for i in range(max_retries):
        try:
            # 将 dialogs 转换为 prompt
            prompt = tokenizer.apply_chat_template(
                dialogs,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # 生成配置
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.9
            
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **gen_kwargs
                )
            
            # 解码
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # 清理显存
            del inputs, outputs
            torch.cuda.empty_cache()
            
            return response, 0
            
        except Exception as e:
            if i < max_retries - 1:
                print(f"⚠️  重试 {i+1}/{max_retries}: {e}")
                torch.cuda.empty_cache()
                time.sleep(6)
            else:
                print(f"❌ 失败: {type(e).__name__}: {e}")
                return "Error", 0


def get_multiple_completion(dialogs, num_cpus=1, temperature=0, max_tokens=100):
    results = []
    
    for i, dialog in enumerate(dialogs):
        print(f"🔄 处理 {i+1}/{len(dialogs)}...", end=" ")
        response, cost = get_completion(dialog, temperature=temperature, max_tokens=max_tokens)
        results.append((response, cost))
        print(f"✅")
        
        # 每个请求后清理显存
        import torch
        torch.cuda.empty_cache()
    
    total_cost = sum([cost for _, cost in results])
    return [response for response, _ in results], total_cost

def format_numbers(numbers):
    return '[' + ', '.join('{:.2f}'.format(num) for num in numbers) + ']'

def format_percentages(numbers):
    return '[' + ', '.join('{:.2%}'.format(num) for num in numbers) + ']'
