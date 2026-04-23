from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from image_preprocess import preprocess
from llm import DebugQwen2_5_VLForConditionalGeneration
import torch
from utils.print_kvcache import print_kv
from utils.warmup import warmup
from utils.image_token_heatmap import visualize_token_heatstrip
import threading
from Qwen_Transformer.kv_cache import CPUKVCache
from utils.ganrao import start_cpu_background_compute

from Qwen_Transformer.attention import layer_attn_cum,imgage_token_attn_collect,image_token_num
import numpy as np
import matplotlib.pyplot as plt
import time

import nvtx
import argparse





device = "cuda:0"

model = DebugQwen2_5_VLForConditionalGeneration.from_pretrained(
    "../models/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map=device
)

processor = AutoProcessor.from_pretrained("../models/Qwen2.5-VL-7B-Instruct")

#后续实验的时候，可以根据这个格式传message （用脚本
messages = [
    {
        "role": "user",
        "content": [
            # {"type": "image", "image": "../images/demo.jpeg"},
            # {"type": "image", "image": "../images/demo.jpeg"},
            {"type": "image", "image": "../images/demo.jpeg"},
            # {"type": "image", "image": "../images/R.jpg"},
            # {"type": "image", "image": "../images/1600.jpg"},
            # {"type": "image", "image": "../images/dogs.png"},
            # {"type": "image", "image": "../images/dogs.png"},     
            # {"type": "image", "image": "../images/dogs.png"},
            # {"type": "image", "image": "../images/dogs.png"},  
            
            # {"type": "image", "image": "../images/dogs.png"},
            # {"type": "image", "image": "../images/dogs.png"},     
            # {"type": "image", "image": "../images/dogs.png"},
            # {"type": "image", "image": "../images/dogs.png"},  
            
            # {"type": "image", "image": "../images/dogs.png"},
            # {"type": "image", "image": "../images/dogs.png"},     
            # {"type": "image", "image": "../images/dogs.png"},
            # {"type": "image", "image": "../images/dogs.png"},  
            # {"type": "image", "image": "../images/dogs.png"},
            # {"type": "image", "image": "../images/dogs.png"},     
            # {"type": "image", "image": "../images/dogs.png"},
            # {"type": "image", "image": "../images/dogs.png"},  
                        
            # {"type": "image", "image": "../images/dogs.png"},
            # {"type": "image", "image": "../images/dogs.png"},     
            # {"type": "image", "image": "../images/dogs.png"},
            # {"type": "image", "image": "../images/dogs.png"},  
            # {"type": "image", "image": "../images/dogs.png"},
            # {"type": "image", "image": "../images/dogs.png"},     
            # {"type": "image", "image": "../images/dogs.png"},
            # {"type": "image", "image": "../images/dogs.png"},  
            # {"type": "image", "image": "../images/dogs.png"},
            # {"type": "image", "image": "../images/dogs.png"},     
            # {"type": "image", "image": "../images/dogs.png"},
            # {"type": "image", "image": "../images/dogs.png"},  
            # {"type": "image", "image": "../images/dogs.png"},
            # {"type": "image", "image": "../images/dogs.png"},     
            # {"type": "image", "image": "../images/dogs.png"},
            # {"type": "image", "image": "../images/dogs.png"}, 

                        
            # {"type": "image", "image": "../images/cats2.jpg"},
            {"type": "text", "text": "Please provide a comprehensive and highly detailed description of the given image, capturing every visible aspect and nuance that can be observed. Your description should go beyond a simple identification of the main subject and should instead cover a broad range of visual elements. These include, but are not limited to: the primary and secondary subjects within the frame, their relative positions, postures, and expressions; the background environment, its objects, textures, and spatial arrangement; the color palette used throughout the image and any notable patterns, contrasts, or gradients; the quality and direction of lighting and shadows; and any discernible artistic style, perspective, or focal point."},
            # {"type": "text", "text": "Describe the image."},
        ],
    },
]


text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

#图片预处理 resize
pre_time_start = time.time()
image_inputs = preprocess(messages)



inputs = processor(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
    # padding_side='left' 
)

#统一单机用cuda
inputs = inputs.to(device)
torch.cuda.synchronize()

pre_time_end = time.time()
print(f"preprocess time: {pre_time_end - pre_time_start}")

#预热模型
# warmup(model)

time.sleep(0.3)
print("正式开始")

past_key_values = None  # 初始没有 KV Cache
rope_deltas = None

num_layers = model.config.num_hidden_layers
max_len = 12000
cpu_kv_cache = CPUKVCache(num_layers=num_layers, max_len=max_len,queue_maxsize=256)


# # 模拟后台CPU在对图片进行resize和conv
# stop, cpu_threads_list = start_cpu_background_compute(threads=1)


# Prefill 阶段
# with torch.no_grad():
#     with nvtx.annotate("prefill", color="red"):
#         output = model(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             pixel_values=inputs.get("pixel_values", None),
#             image_grid_thw=inputs.get("image_grid_thw", None),
#             cache_position=torch.tensor([0], device=device),
#             # cpu_kv_cache=cpu_kv_cache,
#             cpu_kv_cache=None,
#             is_prefill = True,
#         )

#     past_key_values = output.past_key_values
#     rope_deltas = output.rope_deltas
#     cache_position = torch.tensor([1], device=device)

#     next_token = torch.argmax(output.logits[:, -1, :], dim=-1)
#     # print("prefill完成")
#     print(processor.decode(next_token[0], skip_special_tokens=True), end="", flush=True)


def tmp_decoding(*args):
    print("接收到的参数:", args)
    #这边能不能直接跳过prefill阶段，模拟生成KV cache
    tasks_submitted = []
    for i in range(num_layers):
        key_tensor = torch.randn(1, 28, 1024, 128)
        value_tensor = torch.randn(1, 28, 1024, 128)
        
        key_tensor = key_tensor.to(torch.bfloat16)
        value_tensor = value_tensor.to(torch.bfloat16)

        success = cpu_kv_cache.submit_append(i, key_tensor, value_tensor, block=True)
        if not success:
            print(f"Failed to submit task for layer {i}")
        tasks_submitted.append(success)

    # 等待所有任务完成
    cpu_kv_cache._q.join()  # 等待队列中的所有任务完成

    rope_deltas = torch.tensor([[-240]], device=device)
    next_token = torch.tensor([785], device=device)
    cache_position = torch.tensor([1], device=device)




    # 增量 Decode 阶段  轮数自定义
    i = 0
    for _ in range(1024):
        with torch.no_grad():
            with nvtx.annotate(f"decoding{i}", color="yellow"):
                model_inputs = model.prepare_inputs_for_generation(
                    input_ids=next_token.unsqueeze(1),
                    # past_key_values=past_key_values,
                    rope_deltas=rope_deltas,
                    cache_position=cache_position,
                    cpu_kv_cache=cpu_kv_cache,
                    # cpu_kv_cache=None,
                    is_prefill = False,
                )
            i = i + 1
            
            output = model(**model_inputs)
            next_token = torch.argmax(output.logits[:, -1, :], dim=-1)

            past_key_values = output.past_key_values

            # print_kv(past_key_values)

            rope_deltas = output.rope_deltas
            cache_position += 1


            print(processor.decode(next_token[0], skip_special_tokens=True), end="", flush=True)

            # 检查是否是结束符
            if next_token[0].item() == 151645:
                break


def main():
    parser = argparse.ArgumentParser(description='Here is an AD.')
    # 添加可选参数
    parser.add_argument('-m', '--mode', type=int, choices=[1,2,3,4,5],
                        help='Please choose the baseline:\n'
                            '    1. FlexGen\n'
                            '    2. FlexGen+DSA\n'
                            '    3. InfiniGen\n'
                            '    4. ShadowKV\n'
                            '    5. MServe')
    
    args = parser.parse_args()
    
    # 开始推理
    tmp_decoding(args.mode)

if __name__ == '__main__':
    main()
