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
            {
                "type": "image",
                "image": "../images/demo.jpeg" 
            },
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



# step 1: clone 图像张量
dummy_pixel_values = inputs.get("pixel_values", None).clone()
dummy_grid_thw = inputs.get("image_grid_thw", None).clone()

dummy_pixel_values.to("cuda:0")
dummy_grid_thw.to("cuda:0")
def launch_visual_async(model, pixel_values, grid_thw):
    def run_visual():
        torch.cuda.set_device(pixel_values.device)

        # 创建独立 CUDA Stream
        stream = torch.cuda.Stream(device=pixel_values.device)

        with torch.cuda.stream(stream):
            with torch.no_grad():
                vit_start = time.time()
                with nvtx.annotate("vit", color="blue"):
                    _ = model.visual(pixel_values, grid_thw=grid_thw)

            # 同步 stream，确保所有视觉 kernel 执行完毕
                stream.synchronize()
                vit_end = time.time()
                print(f"vit time (in stream): {vit_end - vit_start:.4f}s")

    t = threading.Thread(target=run_visual)
    t.start()
    print("vit 开始")
    return t





with open("prompt.txt", "r", encoding="utf-8") as f:
    text_content = f.read()
text_message = [
    {"role": "user", "content": text_content},
]
# Preparation for inference
# 先暂时使用huggingdace提供的模板
text = processor.apply_chat_template(
    text_message, tokenize=False, add_generation_prompt=True
)

#图片预处理 resize
image_inputs = preprocess(text_message)

inputs = processor(
    text=[text],
    # images=image_inputs,
    padding=True,
    return_tensors="pt",
    # padding_side='left' 
)

#统一单机用cuda0
inputs = inputs.to("cuda:0")

inputs["cache_position"] = torch.tensor([0], device="cuda:0")


#预热模型
warmup(model)

past_key_values = None  # 初始没有 KV Cache
rope_deltas = None
# vit_thread = launch_visual_async(model, dummy_pixel_values, dummy_grid_thw)
# vit_thread.join()
# time.sleep(1.6)
tmp_start = time.time()
# vit_thread.join()
# Prefill 阶段
with torch.no_grad():
    torch.cuda.synchronize()
    prefill_start = time.time()
    with nvtx.annotate("prefill", color="red"):
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values", None),
            image_grid_thw=inputs.get("image_grid_thw", None),
            cache_position=torch.tensor([0], device="cuda")
        )
        torch.cuda.synchronize()
        prefill_end = time.time()
        print(f"prefill time: {prefill_end - prefill_start}")
    # vit_thread.join()
    # tmp_end = time.time()
    # print(f"total time: {tmp_end - tmp_start}")

    past_key_values = output.past_key_values
    rope_deltas = output.rope_deltas
    cache_position = torch.tensor([1], device="cuda")

    next_token = torch.argmax(output.logits[:, -1, :], dim=-1)
    # print(processor.decode(next_token[0], skip_special_tokens=True), end="", flush=True)


# vit_thread = launch_visual_async(model, dummy_pixel_values, dummy_grid_thw)
# 增量 Decode 阶段  轮式自定义
i = 0
# for _ in range(255):
#     with torch.no_grad():
#         with nvtx.annotate(f"decoding{i}", color="yellow"):
#             model_inputs = model.prepare_inputs_for_generation(
#                 input_ids=next_token.unsqueeze(1),
#                 past_key_values=past_key_values,
#                 rope_deltas=rope_deltas,
#                 cache_position=cache_position
#             )
#         i = i + 1
        

#         decode_start = time.time()
#         output = model(**model_inputs)
#         torch.cuda.synchronize()
#         decode_end = time.time()
#         print(f"decode time: {decode_end - decode_start}")

#         next_token = torch.argmax(output.logits[:, -1, :], dim=-1)

#         past_key_values = output.past_key_values

#         # print_kv(past_key_values)

#         rope_deltas = output.rope_deltas
#         cache_position += 1


        # print(processor.decode(next_token[0], skip_special_tokens=True), end="", flush=True)
        # 检查是否是结束符
        # if next_token[0].item() == 151645:
        #     break






