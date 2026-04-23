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
    {"role": "user", "content": "请写一首五言绝句，来描述盛夏"},
]


text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

#图片预处理 resize
image_inputs = preprocess(messages)

inputs = processor(
    text=[text],
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

# Prefill 阶段
with torch.no_grad():
    with nvtx.annotate("prefill", color="red"):
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values", None),
            image_grid_thw=inputs.get("image_grid_thw", None),
            cache_position=torch.tensor([0], device="cuda")
        )


    past_key_values = output.past_key_values
    rope_deltas = output.rope_deltas
    cache_position = torch.tensor([1], device="cuda")

    next_token = torch.argmax(output.logits[:, -1, :], dim=-1)
    # print(processor.decode(next_token[0], skip_special_tokens=True), end="", flush=True)



# 增量 Decode 阶段  轮式自定义
i = 0
for _ in range(255):
    with torch.no_grad():
        with nvtx.annotate(f"decoding{i}", color="yellow"):
            model_inputs = model.prepare_inputs_for_generation(
                input_ids=next_token.unsqueeze(1),
                past_key_values=past_key_values,
                rope_deltas=rope_deltas,
                cache_position=cache_position
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
        # if next_token[0].item() == 151645:
        #     break






