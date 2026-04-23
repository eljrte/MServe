import torch

def warmup(model):

    # 预热模型
    dummy_input = {
        "input_ids": torch.ones((1, 4), dtype=torch.long, device="cuda:0"),  # 4 个 token
        "attention_mask": torch.ones((1, 4), device="cuda"),
        "pixel_values": None,
        "cache_position": torch.tensor([0], device="cuda"),
    }
    with torch.no_grad():
        _ = model(**dummy_input)
        torch.cuda.synchronize()
