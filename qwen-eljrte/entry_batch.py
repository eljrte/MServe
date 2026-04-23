from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from image_preprocess import preprocess
from llm import DebugQwen2_5_VLForConditionalGeneration
import torch
from utils.print_kvcache import print_kv
from utils.warmup import warmup
import threading

import time

import nvtx

model = DebugQwen2_5_VLForConditionalGeneration.from_pretrained(
    "../models/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained("../models/Qwen2.5-VL-7B-Instruct")

#后续实验的时候，可以根据这个格式传message （用脚本
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "../images/demo.jpeg"},
            {"type": "image", "image": "../images/demo.jpeg"},
            # {"type": "image", "image": "../images/demo.jpeg"},
            # {"type": "image", "image": "../images/demo.jpeg"},
            # {"type": "image", "image": "../images/demo.jpeg"},
            {"type": "text", "text": "Describe the picture."},
        ],
    },
]


text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

#图片预处理 resize
image_inputs = preprocess(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
    # padding_side='left' 
)

#统一单机用cuda0
inputs = inputs.to("cuda:0")


#预热模型
warmup(model)

past_key_values = None  # 初始没有 KV Cache
rope_deltas = None

# vit_thread = launch_visual_async(model, dummy_pixel_values, dummy_grid_thw)
# time.sleep(1.5)
# tmp_start = time.time()
# vit_thread.join()
# Prefill 阶段
with torch.no_grad():
    seq_len = 2196
    for layer_idx in range(num_layers):
        mock_k = torch.randn(B, 28, seq_len, 128, dtype=torch.bfloat16, device=device)
        mock_v = torch.randn(B, 28, seq_len, 128, dtype=torch.bfloat16, device=device)
        success_count = cpu_kv_caches.submit_append_batched(layer_idx, mock_k, mock_v, block=True)
        print(f"Layer {layer_idx}: Successfully submitted ")
        if layer_idx % 5 == 0:
            cpu_kv_caches.sync()
            del mock_k, mock_v
            torch.cuda.empty_cache()
# past_key_values = [
#     [
#         torch.randn(B, 28,1000, 128, dtype=torch.bfloat16, device=device),
#         torch.randn(B, 28, 1000, 128, dtype=torch.bfloat16, device=device),
#     ] for _ in range(num_layers)
# ]
rope_deltas = torch.full((B, 1), -240, dtype=torch.long, device=device)  
next_token = torch.full((B,), 785, dtype=torch.long, device=device)
cache_position = torch.full((B,), 1, dtype=torch.long, device=device)



# batch_size = 12
# num_heads = 8
# seq_len = 2196  # 假设我们要预填 10 个 token 的缓存
# head_dim = 128
# past_key_values = DynamicCache()

# # 2. 循环为每一层填充数据
# for i in range(num_layers):
#     # 注意：每一层的 key/value 通常是不同的数值
#     # 这里为了演示，每一层都随机生成新的 tensor
#     key_states = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
#     value_states = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    
#     # 【关键】这里传入当前的 layer_idx = i
#     past_key_values.update(key_states, value_states, layer_idx=i)
# print(num_layers)
# rope_deltas = torch.full((batch_size, 1), -240, dtype=torch.long, device=device)  
# next_token = torch.full((batch_size,), 785, dtype=torch.long, device=device)
# cache_position = torch.full((batch_size,), 1, dtype=torch.long, device=device)

# 增量 Decode 阶段  轮数自定义
i = 0
for _ in range(128):

    # if i == 15:
    #     vit_proc = subprocess.Popen([
    #         "bash", "-c",
    #         f"CUDA_VISIBLE_DEVICES=0 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=70 "
    #         f"/home/dg123/miniconda3/bin/python ./qwen-eljrte/entry_vit_new.py"
    #      ])
    
    with torch.no_grad():
        with nvtx.annotate(f"decoding{i}", color="yellow"):
            model_inputs = model.prepare_inputs_for_generation(
                input_ids=next_token.unsqueeze(1),
                past_key_values=past_key_values,
                rope_deltas=rope_deltas,
                cache_position=cache_position
            )
        i = i + 1
        

        decode_start = time.time()
        output = model(**model_inputs)
        torch.cuda.synchronize()
        decode_end = time.time()
        # print(f"decode time: {decode_end - decode_start}")

        next_token = torch.argmax(output.logits[:, -1, :], dim=-1)

        past_key_values = output.past_key_values

        # print_kv(past_key_values)

        rope_deltas = output.rope_deltas
        cache_position += 1


        # print(processor.decode(next_token[0], skip_special_tokens=True), end="", flush=True)
        # 检查是否是结束符
        # if next_token[0].item() == 151645:
        #     break






