import math
import torch
import time

@torch.no_grad()
def dual_attention_filter(
    attn: torch.Tensor,
    v_token_start: int,
    img_seq: int,
    global_thr: float = 1.0,
    individual_thr: float = 0.0,
    min_keep_ratio: float = 0.35,
    use_fp16: bool = False,
):
    """
    加速版：
      - 只对 head 做一次均值 (复用)
      - 复用 head-mean 后的 [text, img] 块进行 sum / amax
      - 避免不必要的中间张量与 Python 循环
    形状:
      attn: [B, H, T, T]
    返回:
      important_indices: LongTensor, shape [M]
      text_seq: int
    """
    # torch.cuda.synchronize()
    # dual_attn_start = time.time()

    assert attn.dim() == 4, "attn must be [B, H, T, T]"
    B, H, T, _ = attn.shape
    assert B >= 1

    # 仅取 batch 0（与原逻辑一致）
    # 使用切片而非多次 narrow，获得 view 而非拷贝
    # total_seq = sys + img + text
    text_seq = T - img_seq - v_token_start
    img_beg, img_end = v_token_start, v_token_start + img_seq
    txt_beg, txt_end = img_end, img_end + text_seq

    # [H, text, img] 的 view
    block = attn[0, :, txt_beg:txt_end, img_beg:img_end]

    # 可选半精度（减少带宽、可能更快；若数值要求严格可关闭）
    if use_fp16 and block.dtype == torch.float32:
        block = block.half()

    # 对 head 求均值 -> [text, img]
    head_mean = block.mean(dim=0)

    # 全局分数（按 text 维求和）-> [img]
    # 等价于：attn_sum = head_mean.sum(dim=0)
    attn_sum = torch.sum(head_mean, dim=0)

    # 个体阈值分数：在 text 维取最大 -> [img]
    text_max = torch.amax(head_mean, dim=0)

    # 双阈值候选剔除
    # 注意：global_thr 是相对于总和的比例阈值，沿用你的定义
    candidate_mask = (attn_sum < attn_sum.sum() * global_thr) & (text_max < individual_thr)

    all_indices = torch.arange(img_seq, device=attn.device)
    important_indices = all_indices[~candidate_mask]

    # torch.cuda.synchronize()
    # dual_attn_end = time.time()
    # print(f"dual_attn_filter cost time: {dual_attn_end - dual_attn_start}")

    # 至少保留 min_keep_ratio
    # min_keep = max(1, math.ceil(img_seq * float(min_keep_ratio)))
    # min_keep = min(min_keep, img_seq)

    # if important_indices.numel() < min_keep:
    #     # 直接基于 attn_sum 选 topk 填充
    #     # (不做 unique+排序的额外开销，先用布尔掩码再补足更快)
    #     need = min_keep - important_indices.numel()

    #     # 在被剔除的集合里选 topk
    #     # mask 出被剔除的索引
    #     drop_indices = all_indices[candidate_mask]
    #     # 对应的分数
    #     drop_scores = attn_sum[candidate_mask]
    #     # 可能出现 drop_indices 为空（例如阈值很严），防御性处理
    #     if drop_indices.numel() > 0:
    #         topk = min(need, drop_indices.numel())
    #         fill_idx = drop_indices[torch.topk(drop_scores, k=topk, largest=True).indices]
    #         important_indices = torch.cat([important_indices, fill_idx])

        # # 若仍不足（极端情况），从所有 token 里补足剩余 topk（避免重复）
        # if important_indices.numel() < min_keep:
        #     need2 = min_keep - important_indices.numel()
        #     # 用布尔表避免 unique/sort
        #     keep_mask = torch.zeros(img_seq, dtype=torch.bool, device=attn.device)
        #     keep_mask[important_indices] = True
        #     # 在未选中的里再选
        #     remain_scores = attn_sum.clone()
        #     remain_scores[keep_mask] = -float("inf")
        #     extra = torch.topk(remain_scores, k=need2, largest=True).indices
        #     important_indices = torch.cat([important_indices, extra])

        # # 最终做一次排序，方便下游使用
        # important_indices, _ = torch.sort(important_indices)


    return important_indices.to(torch.long), int(text_seq)