import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union,Set

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache

from transformers.utils import add_start_docstrings, logging, replace_return_docstrings
from configuration_qwen2_5_vl import Qwen2_5_VLConfig

import time

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import nvtx
from .kv_cache import CPUKVCache
import torch.nn.functional as F
import threading
from .dual_attention_filter import dual_attention_filter

import time, threading
from concurrent.futures import ThreadPoolExecutor
import cv2

from scipy.stats import spearmanr

import torch.cuda.nvtx as nvtx

logger = logging.get_logger(__name__)

def visualize_attention_map(image_path, attn_scores, token_num=None, 
                              alpha=0.7, patch_size=14, group_size=4,layer_num=0,lunshu_cnt=0):
    """简化版本，只显示最重要的结果"""
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    if isinstance(attn_scores, torch.Tensor):
        attn_scores = attn_scores.detach().cpu().numpy()
    
    if attn_scores.ndim > 1:
        attn_scores = attn_scores.flatten()
    
    token_num = len(attn_scores) if token_num is None else token_num
    
    # 计算网格
    img_aspect = w / h
    best_h, best_w = 1, token_num
    for h_div in range(1, int(np.sqrt(token_num)) + 1):
        if token_num % h_div == 0:
            w_div = token_num // h_div
            if abs(w_div/h_div - img_aspect) < abs(best_w/best_h - img_aspect):
                best_h, best_w = h_div, w_div
    
    # 重塑和归一化
    attn_map = attn_scores[:best_h*best_w].reshape(best_h, best_w)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    # 创建区块级别的热力图
    block_h = h // best_h
    block_w = w // best_w
    block_map = np.zeros((h, w))
    
    for i in range(best_h):
        for j in range(best_w):
            y_start = i * block_h
            y_end = min((i + 1) * block_h, h)
            x_start = j * block_w
            x_end = min((j + 1) * block_w, w)
            block_map[y_start:y_end, x_start:x_end] = attn_map[i, j]
    
    # 创建热力图和叠加
    heatmap = np.uint8(255 * block_map)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title(f"Original\n{w}x{h}")
    axes[0].axis('off')
    
    axes[1].imshow(attn_map, cmap='jet')
    axes[1].set_title(f"Attention Grid\n{best_h}x{best_w}")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (α={alpha})")
    axes[2].axis('off')
    
    # 在叠加图上绘制区块边界
    for i in range(best_h + 1):
        y = i * block_h
        axes[2].axhline(y=y, color='white', linewidth=1, alpha=0.5)
    for j in range(best_w + 1):
        x = j * block_w
        axes[2].axvline(x=x, color='white', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"attention_map_{layer_num}_{lunshu_cnt}.png", dpi=300, bbox_inches="tight")
    
    return overlay


def top_token_set_per_layer(attn_layer: torch.Tensor, threshold: float = 0.95,image_len=0) -> Set[int]:
    """
    attn_layer: [H, Q, K] 或 [H, 1, K]
    返回：达到阈值所需的 token 下标集合（Python set）
    """
    assert attn_layer.ndim >= 2, f"unexpected shape: {attn_layer.shape}"
    # 将非 token 维（最后一维以外）全部平均，得到 [K]
    reduce_dims = tuple(range(attn_layer.ndim - 1))
    token_scores = attn_layer.mean(dim=reduce_dims)  # -> [K]

    # 降序排序，做前缀和占比
    scores_sorted, indices_sorted = torch.sort(token_scores, descending=True)
    cumsum = torch.cumsum(scores_sorted, dim=0)
    total = cumsum[-1].clamp_min(1e-12)
    ratio = cumsum / total

    # 第一个使累计占比 ≥ threshold 的位置
    k = int(torch.searchsorted(ratio, threshold).item()) + 1
    top_indices = indices_sorted[:k].tolist()

    # if all(idx > len(token_scores)-100 or idx <=15+image_len for idx in top_indices):
    #     print("hellp")
    # else:
    #     print("damn")
    
    # 过滤掉 index > K-10  过滤掉recent 这些在gpu上
    K_total = attn_layer.shape[-1]
    # print(15+image_len)
    top_indices = [idx for idx in top_indices if idx >= 15+image_len]
    
    return set(top_indices),top_indices


def top_token_set_per_layer_num(attn_layer: torch.Tensor, top_k: int = 50,image_len=0) -> Set[int]:
    """
    attn_layer: [H, Q, K] 或 [H, 1, K]
    返回：前 top_k 的 token 下标集合（Python set）
    """
    assert attn_layer.ndim >= 2, f"unexpected shape: {attn_layer.shape}"
    # 将非 token 维（最后一维以外）全部平均，得到 [K]
    reduce_dims = tuple(range(attn_layer.ndim - 1))
    token_scores = attn_layer.mean(dim=reduce_dims)  # -> [K]

    if top_k > token_scores.shape[0]:
        top_k = token_scores.shape[0]

    # 按分数降序取前 top_k
    score_sorted, indices_sorted = torch.sort(token_scores, descending=True)
    top_indices = indices_sorted[:top_k].tolist()

    total_score = score_sorted[:top_k].sum().item()

    # if all(idx > len(token_scores)-150 or idx <=15+image_len for idx in top_indices):
    #     print("hellp")
    # else:
    #     print("damn")
    # print(total_score)

    top_indices = [idx for idx in top_indices if idx >= 15+image_len]

    return set(top_indices),top_indices

def jaccard_similarity(a: Set[int], b: Set[int]) -> float:
    if not a and not b:  # 两个空集，定义为 1.0
        return 1.0
    return len(a & b) / max(1, len(a | b))

def containment_score(prev_set: Set[int], curr_set: Set[int], empty_policy: str = "one") -> float:
    """
    计算 'curr_set 是否被包含在 prev_set' 的程度：|prev∩curr| / |curr|
    empty_policy:
      - "one": 若 curr_set 为空，按真子集的“真包含”视作 1.0（真空蕴含）
      - "zero": 若 curr_set 为空，返回 0.0
      - "nan":  若 curr_set 为空，返回 NaN（上层自行忽略）
    """
    if len(curr_set) == 0:
        return 1.0 if empty_policy == "one" else (0.0 if empty_policy == "zero" else math.nan)
    return len(prev_set & curr_set) / len(curr_set)

def dual_attention_filter_adaptive(attn, v_token_start, img_seq, target_ll=0.75, 
                                   max_iter=25, eps=1e-6):
    """
    自适应地选择全局阈值 gamma_th（对应 attn_sum）和个体阈值 alpha_th（对应 attn.max），
    使得 (V <= gamma_th) & (I <= alpha_th) 的比例 ≈ target_ll。

    返回：
      important_indices: 需要保留的视觉 token 索引（相对图像段起点的局部索引）
      text_seq: 文本 token 数
      (gamma_th, alpha_th): 实际采用的两个阈值（数值阈，不是比例）
    """
    # 1) 形状裁剪：取 文本→图像 的 cross-attn 子矩阵
    # attn: [batch, head, total_seq, total_seq] 或 [head, total_seq, total_seq]
    if attn.dim() == 4:
        attn = attn[0]  # 只取 batch 0
    nhead, total_seq, _ = attn.shape
    # 估算 text_seq
    text_seq = total_seq - img_seq - v_token_start

    # [nhead, text_seq, img_seq]
    attn = attn.narrow(1, v_token_start + img_seq, text_seq).narrow(2, v_token_start, img_seq)

    # 2) 聚合：head 平均 -> [text_seq, img_seq]
    attn = attn.mean(dim=0)

    # 3) 两个一维指标：
    #    V：全局关注度（沿 text 求和）  [img_seq]
    #    I：个体峰值（沿 text 取最大） [img_seq]
    V = attn.sum(dim=0)                # global
    I = attn.max(dim=0).values         # individual

    # 4) 设定 individual 轴的目标分位数（经验：pi = sqrt(target_ll)）
    #    这样二维独立近似下，LL ≈ pv * pi ≈ target_ll
    pi = V.new_tensor(target_ll).sqrt().clamp_(0.0 + eps, 1.0 - eps)

    # 为了鲁棒：若所有值相等，避免 quantile 退化
    def safe_quantile(x, q):
        if torch.allclose(x, x[0], atol=0, rtol=0):
            return x[0]
        return torch.quantile(x, q.item())

    # 给定 V 的分位数 pv 和固定 pi，计算左下象限占比
    def ll_ratio_for(pv_q):
        gamma_th = safe_quantile(V, pv_q)   # 全局阈值（数值）
        alpha_th = safe_quantile(I, pi)     # 个体阈值（数值）
        ll = ((V <= gamma_th) & (I <= alpha_th)).float().mean()
        return ll, gamma_th, alpha_th

    # 5) 在 pv∈[0,1] 上二分，使 LL 比例 ≈ target_ll
    lo, hi = V.new_tensor(0.0), V.new_tensor(1.0)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        ll_ratio, gamma_th, alpha_th = ll_ratio_for(mid)
        if (ll_ratio < target_ll):
            lo = mid
        else:
            hi = mid

    # 6) 最终阈值（数值阈）
    pv_star = (lo + hi) / 2
    ll_ratio, gamma_th, alpha_th = ll_ratio_for(pv_star)

    # 7) 生成 mask（True 表示“应当丢弃”的视觉 token）
    candidate_mask = (V <= gamma_th) & (I <= alpha_th)

    # 8) 重要 token = 取反
    all_indices = torch.arange(img_seq, device=attn.device)
    important_indices = all_indices[~candidate_mask]

    return important_indices, int(text_seq), (gamma_th, alpha_th, ll_ratio.item())

def plot_overlap_heatmap(attn_scores: torch.Tensor, k: int = 100,layer_idx: int = 0):
    """
    绘制 overlap@k 热力图
    attn_scores: [H, 1, T] attention 分数张量
    k: top-k 大小
    """
    assert attn_scores.dim() == 3 and attn_scores.size(1) == 1, "shape 必须是 [H, 1, T]"
    H, _, T = attn_scores.shape
    scores = attn_scores.squeeze(1)  # [H, T]

    kk = min(k, T)
    topk_idx = torch.topk(scores, kk, dim=1, largest=True, sorted=False).indices  # [H, kk]

    # 构造 bool 矩阵 [H, T]，表示是否是 top-k
    B = torch.zeros_like(scores,  dtype=torch.float32)
    B.scatter_(1, topk_idx, True)

    # |∩| = B @ B^T ；overlap = |∩| / k
    inter = B @ B.t()                  # [H, H] float32
    overlap_k = inter / float(kk)      # [H, H]

    # 画图需搬到 CPU
    M = overlap_k.detach().cpu()


    # 绘制热力图
    plt.figure(figsize=(8, 6))
    im = plt.imshow(M, cmap='hot', interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ttl = f'Head Overlap@{kk} Heatmap'
    if layer_idx is not None:
        ttl += f' (Layer {layer_idx})'
    plt.title(ttl)
    plt.xlabel('Head')
    plt.ylabel('Head')
    if H <= 64:  # 太多的话就别打满刻度
        plt.xticks(range(H))
        plt.yticks(range(H))
    plt.tight_layout()
    plt.savefig(f'overlap_heatmap_k{k}_layer{layer_idx}.png')   


import torch

import torch, math

def head_entropy(attn_weights: torch.Tensor, already_prob: bool = False,
                 valid_mask: torch.Tensor | None = None, eps: float = 1e-12):
    """
    基于熵评估 head 重要性（低熵=高重要性）
    attn_weights: [H, 1, T]，可以是 logits/score 或已softmax后的概率
    already_prob: 如果传入的是概率，设为 True（只做归一化）；否则按 token 维做 softmax
    valid_mask: 可选，[T] 的 bool 张量，True 表示有效 token（会在熵与归一化里仅统计有效数）
    返回:
      importance: [H]，越大越重要 (= 1 - H_norm)
      entropy: [H]，自然对数底
      entropy_norm: [H]，H/log(T_eff)
      order: [H]，按重要性降序的下标
    """
    assert attn_weights.dim() == 3 and attn_weights.size(1) == 1, "shape 必须是 [H, 1, T]"
    H, _, T = attn_weights.shape
    x = attn_weights.squeeze(1).to(torch.float32)  # [H, T]

    if valid_mask is not None:
        assert valid_mask.shape == (T,) and valid_mask.dtype == torch.bool
        if already_prob:
            x = x.masked_fill(~valid_mask, 0.0)
        else:
            x = x.masked_fill(~valid_mask, float('-inf'))

    # 关键修正：默认强制按 token 维 softmax
    if already_prob:
        p = x.clamp_min(0)
        p = p / (p.sum(dim=-1, keepdim=True) + eps)
    else:
        p = torch.softmax(x, dim=-1)

    # 熵
    entropy = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=-1)  # [H]

    # 有效 token 数用于归一化
    T_eff = int(valid_mask.sum().item()) if valid_mask is not None else T
    maxH = math.log(max(T_eff, 1))
    entropy_norm = (entropy / (maxH + eps)).clamp(0, 1)

    importance = (1.0 - entropy_norm).clamp(0, 1)
    order = torch.argsort(importance, descending=True)
    return importance, entropy, entropy_norm, order


import torch
import matplotlib.pyplot as plt

def plot_hidden_dim_scores(q: torch.Tensor,
                           k: torch.Tensor,
                           reduce: str = "mean",   # "mean" | "sum" | "rms"
                           norm: str | None = None, # None | "zscore" | "minmax"
                           layer_idx: int = 0
                           ):
    """
    可视化每个 hidden 维度的得分（不返回值）
    q, k: [B, H, T, Dh]
    reduce:
      - "mean":  score[d] = mean_{B,H,T}(|q|+|k|)
      - "sum":   score[d] = sum_{B,H,T}(|q|+|k|)   (受长度影响)
      - "rms":   score[d] = sqrt(mean(q^2) + mean(k^2)) / sqrt(2)
    norm:
      - None:    不做归一化
      - "zscore":(x-mean)/std，便于看离群
      - "minmax":(x-min)/(max-min)，压到[0,1]
    """
    # print(q.shape)
    assert q.shape == k.shape and q.dim() == 4, "q/k 需要形状 [B,H,T,Dh] 且一致"
    q = q.detach().float()
    k = k.detach().float()

    if reduce == "mean":
        scores = (q.abs().mean(dim=(0,1,2)) + k.abs().mean(dim=(0,1,2)))  # [Dh]
    elif reduce == "sum":
        scores = (q.abs().sum(dim=(0,1,2)) + k.abs().sum(dim=(0,1,2)))    # [Dh]
    elif reduce == "rms":
        scores = torch.sqrt(q.pow(2).mean(dim=(0,1,2)) + k.pow(2).mean(dim=(0,1,2))) / (2 ** 0.5)
    else:
        raise ValueError("reduce 仅支持 'mean' | 'sum' | 'rms'")

    # 可选归一化，便于跨层/样本对比或突出离群
    if norm is not None:
        x = scores
        if norm == "zscore":
            scores = (x - x.mean()) / (x.std(unbiased=False) + 1e-12)
        elif norm == "minmax":
            scores = (x - x.min()) / (x.max() - x.min() + 1e-12)
        else:
            raise ValueError("norm 仅支持 None | 'zscore' | 'minmax'")

    s = scores.detach().cpu().numpy()
    D = s.shape[0]

    plt.figure(figsize=(10, 4))
    plt.plot(range(D), s)  # 不指定颜色
    plt.xlabel("Hidden dimension index")
    ylabel = {
        "mean": "|q|+|k| (mean over B,H,T)",
        "sum":  "|q|+|k| (sum over B,H,T)",
        "rms":  "RMS(q,k) over B,H,T"
    }[reduce]

    if norm == "zscore":
        ylabel += "  [z-score]"
    elif norm == "minmax":
        ylabel += "  [min-max 0~1]"
    plt.ylabel(ylabel)
    plt.title("Per-dimension magnitude")
    plt.tight_layout()
    plt.savefig(f"hidden_dim_scores_demo_{layer_idx}.png")
    plt.close()


def save_hidden_dim_scores_per_head(
    q: torch.Tensor,
    k: torch.Tensor,
    reduce: str = "mean",     # "mean" | "sum" | "rms"
    norm: Optional[str] = None,  # None | "zscore" | "minmax"
    layer_idx: int = 0,
    heads: Optional[List[int]] = None,   # 指定子集 head，可选
    out_prefix: str = "hidden_dim_scores"
):
    """
    为每个 head 生成一张图片，展示该 head 的 per-dimension 分数曲线。
    不返回值；图片保存到 /mnt/data 下。

    形状：q, k: [B, H, T, Dh]
    分数定义（对每个 head 单独）：
        - "mean":  score[d] = mean_{B,T}(|q|+|k|)
        - "sum":   score[d] = sum_{B,T}(|q|+|k|)   (受长度影响)
        - "rms":   score[d] = sqrt(mean(q^2)+mean(k^2)) / sqrt(2)
    归一化（可选，逐 head 单独进行）：
        - "zscore": (x - mean) / std
        - "minmax": (x - min) / (max - min)
    """
    assert q.shape == k.shape and q.dim() == 4, "q/k 必须是 [B,H,T,Dh] 且一致"
    q = q.detach().float()
    k = k.detach().float()
    B, H, T, D = q.shape

    if heads is None:
        heads = list(range(H))

    # 逐 head 处理与绘图
    saved_files = []
    for h in heads:
        qh = q[:, h, :, :]  # [B,T,D]
        kh = k[:, h, :, :]  # [B,T,D]

        if reduce == "mean":
            scores = (qh.abs().mean(dim=(0,1)) + kh.abs().mean(dim=(0,1)))      # [D]
        elif reduce == "sum":
            scores = (qh.abs().sum(dim=(0,1)) + kh.abs().sum(dim=(0,1)))        # [D]
        elif reduce == "rms":
            scores = torch.sqrt(qh.pow(2).mean(dim=(0,1)) + kh.pow(2).mean(dim=(0,1))) / (2 ** 0.5)
        else:
            raise ValueError("reduce 仅支持 'mean' | 'sum' | 'rms'")

        # 逐 head 归一化（可选）
        if norm is not None:
            x = scores
            if norm == "zscore":
                scores = (x - x.mean()) / (x.std(unbiased=False) + 1e-12)
            elif norm == "minmax":
                scores = (x - x.min()) / (x.max() - x.min() + 1e-12)
            else:
                raise ValueError("norm 仅支持 None | 'zscore' | 'minmax'")

        s = scores.detach().cpu().numpy()
        # 单独一张图（遵循：每张图一个图表，不设置颜色）
        plt.figure(figsize=(10, 4))
        plt.plot(range(D), s)
        plt.xlabel("Hidden dimension index")
        ylabel = {
            "mean": "|q|+|k| (mean over B,T)",
            "sum":  "|q|+|k| (sum over B,T)",
            "rms":  "RMS(q,k) over B,T"
        }[reduce]
        if norm == "zscore":
            ylabel += "  [z-score]"
        elif norm == "minmax":
            ylabel += "  [min-max 0~1]"
        plt.ylabel(ylabel)
        plt.title(f"Layer {layer_idx} - Head {h} - Per-dimension magnitude")
        plt.tight_layout()

        out_path = f"./dim_outlier/{out_prefix}_L{layer_idx}_H{h}.png"
        plt.savefig(out_path)
        plt.close()



image_attn_cum = None
layer_attn_cum = [0] * 28
image_token_num = 0
record_top_token_per_layer = [set() for _ in range(28)]  # 每层的 top token 集合
imgage_token_attn_collect = [0]*3578  #256dog 391cats2 3577demo
prefill_top_image_token = [set() for _ in range(28)]  # 每层的 prefill 阶段重要图像 token 集合
lunshu_cnt = 0
layer1_diff = set()
tmp_record = [set() for _ in range(28)]
layer0_3_attn_accu = None
helper = [0]
important_image = [None]*28
shallow_layers_attn_score = None
last_attn_score = None
prev_top_tokens = None
import config_variable
critical_tokens_set_i = None
critical_tokens_set_i_1=None
critical_tokens_set_i_2 = None
critical_tokens_set_i_3 = None

critical_tokens_set_layer_i = None
critical_tokens_set_layer_i_1 = None
critical_tokens_set_layer_i_2 = None

lfu_cache = None
lru_cache = None


attn_score = None
attn_score2 = None
attn_score3 = None
prev_iter_tokens=None
curr_iter_tokens=None
def compute_token_importance(
    query: torch.Tensor,
    key: torch.Tensor,
    method: str = "attention_score",
    top_k_ratio: float = 0.3,
    min_tokens: int = 128,
) -> torch.Tensor:
    """
    计算 token 重要性并返回要保留的 token 索引。

    Args:
        query: [B, H, Q, D] Query 张量
        key: [B, H, T, D] Key 张量（可能包含历史和当前）
        method: 重要性计算方法 ("attention_score" | "norm")
        top_k_ratio: 保留的 token 比例
        min_tokens: 最少保留的 token 数

    Returns:
        indices: [num_selected] 重要 token 的索引（在 CPU 上）
    """
    B, H, Q, D = query.shape
    _, _, T, _ = key.shape

    if method == "attention_score":
        # 使用 query-key 相似度作为重要性指标
        # 只对最后一个 query 计算（decoding 阶段 Q=1）
        q = query[:, :, -1:, :]  # [B, H, 1, D]

        # 计算 attention score (不用 sqrt 缩放，只用于排序)
        scores = torch.matmul(q, key.transpose(-2, -1))  # [B, H, 1, T]
        scores = scores.squeeze(2)  # [B, H, T]

        # 跨 head 取平均
        scores = scores.mean(dim=1)  # [B, T]

        # 取 top-k
        k = max(min_tokens, int(T * top_k_ratio))
        k = min(k, T)

        _, indices = torch.topk(scores, k, dim=-1, largest=True, sorted=False)
        indices = indices.squeeze(0)  # [k]

    elif method == "norm":
        # 使用 key 的 L2 norm 作为重要性指标
        norms = key.norm(dim=-1).mean(dim=1)  # [B, T]
        k = max(min_tokens, int(T * top_k_ratio))
        k = min(k, T)
        _, indices = torch.topk(norms, k, dim=-1, largest=True, sorted=False)
        indices = indices.squeeze(0)
    else:
        raise ValueError(f"Unknown importance method: {method}")

    # 确保索引在 CPU 上且是 long 类型
    if indices.device.type != "cpu":
        indices = indices.cpu()

    return indices.long()


def sparse_sdpa_attention(
    query: torch.Tensor,
    key_full: torch.Tensor,
    value_full: torch.Tensor,
    key_sparse: torch.Tensor,
    value_sparse: torch.Tensor,
    sparse_indices: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    稀疏注意力计算：对选中的重要 token 计算精确注意力，其余 token 使用局部 attention 近似。
    简化版本：直接使用选中的 token 计算 attention，忽略剩余 token（假设它们贡献较小）。

    Args:
        query: [B, H, Q, D]
        key_full: [B, H, T_full, D] 完整的 key（备用）
        value_full: [B, H, T_full, D] 完整的 value（备用）
        key_sparse: [B, H, T_sparse, D] 选中的重要 token 的 key
        value_sparse: [B, H, T_sparse, D] 选中的重要 token 的 value
        sparse_indices: [T_sparse] 选中 token 的索引（在 key_full 中的位置）
        attn_mask: 可选的 attention mask（简化处理）
        dropout_p: dropout 概率
        is_causal: 是否使用因果 mask（解码阶段 Q=1 时忽略）

    Returns:
        attn_output: [B, H, Q, D]
    """
    B, H, Q, D = query.shape
    _, _, T_sparse, _ = key_sparse.shape

    # 简化版本：只使用选中的 token 计算 attention
    # 这对于解码阶段 (Q=1) 是合理的，因为当前 token 主要关注重要的历史 token

    # 1. 计算选中 token 的 attention scores
    attn_scores = torch.matmul(query, key_sparse.transpose(-2, -1)) / math.sqrt(D)  # [B, H, Q, T_sparse]

    # 2. 可选：添加一个小的 bias 来补偿未选中 token 的贡献
    # 这可以通过缩放来实现：让选中 token 的 attention 和接近 1

    # 3. Softmax 归一化（在选中的 token 上）
    attn_prob = torch.softmax(attn_scores, dim=-1)

    # 4. 计算输出
    attn_output = torch.matmul(attn_prob, value_sparse)  # [B, H, Q, D]

    if dropout_p > 0:
        attn_output = F.dropout(attn_output, p=dropout_p, training=True)

    return attn_output


def sdpa_with_scores(
    q, k, v,
    attn_mask=None,        # 形状可 broadcast 到 [..., Tq, Tk]；bool 或加性(-inf)都行
    is_causal=False,
    dropout_p=0.0,
    training=False,
    layer_idx = None,  # 用于调试
    image_len= None,  # 用于调试，图像 token 的长度
):

    global image_token_num,imgage_token_attn_collect
    if image_len > 0:
        image_token_num = image_len

    # q,k,v: [B, H, Tq/Tk, D]
    B, H, Tq, D = q.shape
    Tk = k.size(-2)

    # 用 fp32 做 softmax 更稳
    qf = q.float()
    kf = k.float()

    # 1) 打分 logits
    attn_logits = torch.matmul(qf, kf.transpose(-2, -1)) * (1.0 / math.sqrt(D))  # [B,H,Tq,Tk]

    # 2) 掩码
    if is_causal:
        # 上三角置 -inf：注意 Tq 与 Tk 可不等
        causal = torch.ones((Tq, Tk), dtype=torch.bool, device=attn_logits.device).triu(1)
        attn_logits = attn_logits.masked_fill(causal, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_logits = attn_logits.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_logits = attn_logits + attn_mask  # 加性掩码一般是 0 或 -inf

    # 3) softmax（fp32），再按需要 cast 回去
    attn_prob = torch.softmax(attn_logits, dim=-1)


    with torch.no_grad():
        B, H, Q, K = attn_prob.shape
        sample_heads = H
        sample_query = Q
        global lunshu_cnt,critical_tokens_set_layer_i,critical_tokens_set_layer_i_1,critical_tokens_set_layer_i_2,attn_score,attn_score2,attn_score3


        # global lfu_cache
        # if sample_query >1 and layer_idx == 14:
        #     # print(image_token_num)
        #     # 初始化LRU cache 先不考虑prefill的东西
        #     lfu_cache = dict()
        #     #将lfu_cache先填visual tokens
        #     text_to_image_attention = attn_prob[:, :, -5:, 15:15+image_token_num].mean(dim=(0,1,2))
        #     # print(text_to_image_attention.shape)
        #     #找出最高的60个
        #     top_values, top_indices = torch.topk(text_to_image_attention, k=100)
        #     lfu_cache = {idx.item(): 1 for idx in top_indices}
        #     #加入idx1-14
        #     for i in range(14):
        #         lfu_cache[i] = 1
        #     #加入15+image-15+image+20
        #     for i in range(15+image_token_num,15+image_token_num+20):
        #         lfu_cache[i] = 1
        
        # global lunshu_cnt
        # if layer_idx == 14 and sample_query == 1:
        #     lunshu_cnt+=1
        #     attn_weights_cpu = attn_prob[0, :sample_heads,:sample_query, :].float()
        #     scores_sorted, indices = torch.sort(attn_weights_cpu.mean(dim=0).squeeze(0),descending=True)
        #     cumsum_w = torch.cumsum(scores_sorted, dim=0)
            
        #     threshold = 0.9
        #     cumulative_percentage = cumsum_w / cumsum_w[-1]  # 归一化到总和为1
        #     indices_90_percent = indices[cumulative_percentage <= threshold]
            

            # # # 下面测试LRU 命中率
            # if lunshu_cnt ==1:
            #     #初始化LRU  cache
            #     for idx in indices_90_percent:
            #         if len(lfu_cache) <300:
            #             lfu_cache[idx.item()] = lunshu_cnt
            #         else:
            #             break
            # else:
            # #计算LRU cache命中率
            # hit_count = 0
            # for idx in indices_90_percent:
            #     if idx.item() in lfu_cache:
            #         hit_count += 1
            # hit_rate = hit_count / len(indices_90_percent)
            # with open("lru_cache_hit_rate_mserve_visual_2_90.txt", "a") as file:
            #     file.write(f"{hit_rate}\n")

            # # 更新LRU cache
            # for idx in indices_90_percent:
            #     if idx.item() not in lfu_cache and len(lfu_cache) >=250:
            #         # 找到最久未使用的key
            #         lru_key = min(lfu_cache, key=lfu_cache.get)
            #         del lfu_cache[lru_key]
            #         lfu_cache[idx.item()] = lunshu_cnt
            #     elif idx.item() not in lfu_cache and len(lfu_cache) <250:
            #         lfu_cache[idx.item()] = lunshu_cnt
            #     else:
            #         # 更新使用时间
            #         lfu_cache[idx.item()] = lunshu_cnt


        global critical_tokens_set_layer_i,lunshu_cnt,prev_iter_tokens,curr_iter_tokens

         # 第一次使用时初始化
        if lunshu_cnt == 0:
            prev_iter_tokens = {}
            curr_iter_tokens = {}
            lunshu_cnt+=1


        if sample_query == 1:
            # 当 layer 0 开始时，说明进入了新一轮迭代
            if layer_idx == 0:
                prev_iter_tokens = curr_iter_tokens.copy()
                curr_iter_tokens = {}

            # 当前层 attention score
            attn_weights_cpu = attn_prob[0, :sample_heads, :sample_query, :].float()
            attn_score = attn_weights_cpu.mean(dim=0).squeeze(0)   # [num_tokens]

            # 取当前层 top 10%
            _, indices = torch.sort(attn_score, descending=True)
            num_tokens = attn_score.shape[0]
            top_k = max(1, int(num_tokens * 0.1))
            top_indices = indices[:top_k]
            current_layer_critical_tokens = set(top_indices.tolist())

            # 先把当前层结果存下来，供后续层使用
            curr_iter_tokens[layer_idx] = current_layer_critical_tokens

            # layer 0 没有 layer i-1，无法做你说的这个并集预测
            if layer_idx > 0:
                prev_layer_i_tokens = prev_iter_tokens.get(layer_idx, set())         # 上一次迭代的 layer i
                curr_layer_i_minus_1_tokens = curr_iter_tokens.get(layer_idx - 1, set())  # 本次迭代的 layer i-1

                # 并集预测集合
                # predicted_tokens = prev_layer_i_tokens | curr_layer_i_minus_1_tokens
                predicted_tokens = prev_layer_i_tokens


                # 和当前 layer i 的真实集合比较
                overlap = len(predicted_tokens & current_layer_critical_tokens)
                similarity = overlap / len(current_layer_critical_tokens)

                with open("per_iteration_similarity_past.txt", "a") as f:
                    f.write(f"{similarity}\n")


        # if sample_query == 1:
        #     # 下方验证同一次迭代，相邻层的 critical tokens 相似性
        #     attn_weights_cpu = attn_prob[0, :sample_heads, :sample_query, :].float()
        #     attn_score = attn_weights_cpu.mean(dim=0).squeeze(0)   # [num_tokens]

        #     # 按注意力分数从高到低排序
        #     scores_sorted, indices = torch.sort(attn_score, descending=True)

        #     # 取前 10% 的 token
        #     num_tokens = attn_score.shape[0]
        #     top_k = max(1, int(num_tokens * 0.1))   # 至少取 1 个
        #     top_indices = indices[:top_k]
        #     current_layer_critical_tokens = set(top_indices.tolist())

        #     if layer_idx == 0:
        #         lunshu_cnt += 1
        #         critical_tokens_set_layer_i = current_layer_critical_tokens
        #     else:
        #         # 计算相似性
        #         diff = len(critical_tokens_set_layer_i & current_layer_critical_tokens)
        #         similarity = diff / len(current_layer_critical_tokens)

        #         with open("per_iteration_similarity_4_6_4.txt", "a") as f:
        #             f.write(f"{similarity}\n")

        #         critical_tokens_set_layer_i = current_layer_critical_tokens


        # if sample_query == 1:
        #     # 下方验证同一次迭代，相隔三层的critical tokens 相似性
        #     if layer_idx == 0:
        #         lunshu_cnt+=1
        #         attn_weights_cpu = attn_prob[0, :sample_heads,:sample_query, :].float()

        #         attn_score = attn_weights_cpu.mean(dim=0).squeeze(0)

        #         scores_sorted, indices = torch.sort(attn_weights_cpu.mean(dim=0).squeeze(0),descending=True)
        #         cumsum_w = torch.cumsum(scores_sorted, dim=0)
                
        #         threshold = 0.93
        #         cumulative_percentage = cumsum_w / cumsum_w[-1]  # 归一化到总和为1
        #         indices_90_percent = indices[cumulative_percentage <= threshold]
        #         critical_tokens_set_layer_i = set(indices_90_percent.tolist())
        #     else:
        #         attn_weights_cpu = attn_prob[0, :sample_heads,:sample_query, :].float()

        #         attn_score2 = attn_weights_cpu.mean(dim=0).squeeze(0)   

        #         scores_sorted, indices = torch.sort(attn_weights_cpu.mean(dim=0).squeeze(0),descending=True)
        #         cumsum_w = torch.cumsum(scores_sorted, dim=0)

        #         threshold = 0.93
        #         cumulative_percentage = cumsum_w / cumsum_w[-1]  # 归一化到总和为1
        #         indices_90_percent = indices[cumulative_percentage <= threshold]
        #         current_layer_critical_tokens = set(indices_90_percent.tolist())

        #         # # 计算相似性
        #         # intersection = len(current_layer_critical_tokens & critical_tokens_set_layer_i)
        #         # union = len(current_layer_critical_tokens)
        #         diff = len(critical_tokens_set_layer_i&current_layer_critical_tokens)
        #         similarity = diff / len(current_layer_critical_tokens)
        #         # if union == 0:
        #         #     similarity = 0  # 如果两个集合都为空，则相似度为0
        #         # else:
        #         #     similarity = intersection / union

        #         with open("per_iteration_similarity_4_6_2.txt", "a") as f:
        #             f.write(f"{similarity}\n")
                
        #         critical_tokens_set_layer_i = current_layer_critical_tokens


    


        # 下方验证iteration之间的critical tokens 相似性
        # if layer_idx == 14 and sample_query == 1:
        #     lunshu_cnt+=1
        #     attn_weights_cpu = attn_prob[0, :sample_heads,:sample_query, :].float()
        #     scores_sorted, indices = torch.sort(attn_weights_cpu.mean(dim=0).squeeze(0),descending=True)
        #     cumsum_w = torch.cumsum(scores_sorted, dim=0)
            
        #     threshold = 0.9
        #     cumulative_percentage = cumsum_w / cumsum_w[-1]  # 归一化到总和为1
        #     indices_90_percent = indices[cumulative_percentage <= threshold]

        #     global critical_tokens_set_i, critical_tokens_set_i_1, critical_tokens_set_i_2, critical_tokens_set_i_3
        #     if lunshu_cnt >=5:
        #         current_layer_critical_tokens = set(indices_90_percent.tolist())
        #         intersection = len(current_layer_critical_tokens & critical_tokens_set_i)
        #         union = len(current_layer_critical_tokens | critical_tokens_set_i)
        #         if union == 0:
        #             similarity = 0  # 如果两个集合都为空，则相似度为0
        #         else:
        #             similarity = intersection / union

        #         with open("similarity_layer2.txt", "a") as file:
        #             file.write(f"{similarity}\n")

        #         critical_tokens_set_i = critical_tokens_set_i_1
        #         critical_tokens_set_i_1 = critical_tokens_set_i_2
        #         critical_tokens_set_i_2 = critical_tokens_set_i_3
        #         critical_tokens_set_i_3 = set(indices_90_percent.tolist())

            
        #     if lunshu_cnt == 1:
        #         critical_tokens_set_i = set(indices_90_percent.tolist())
        #     if lunshu_cnt == 2:
        #         critical_tokens_set_i_1 = set(indices_90_percent.tolist())
        #     if lunshu_cnt == 3:
        #         critical_tokens_set_i_2 = set(indices_90_percent.tolist())
        #     if lunshu_cnt == 4:
        #         critical_tokens_set_i_3 = set(indices_90_percent.tolist())    

        #     print(f"轮数:{lunshu_cnt},critical_tokenset_i:{len(critical_tokens_set_i)}")
            
            

            





        
        # image冗余
        # if sample_query >1:

        #     config_variable.image_token_set = [0] * image_token_num

        # if layer_idx == 14 and sample_query == 1:
        #     #获取attention score累加到90%的token id
        #     attn_weights_cpu = attn_prob[0, :sample_heads,:sample_query, :].float()
        #     scores_sorted, indices = torch.sort(attn_weights_cpu.mean(dim=0).squeeze(0),descending=True)
        #     cumsum_w = torch.cumsum(scores_sorted, dim=0)
            
        #     threshold = 0.9
        #     cumulative_percentage = cumsum_w / cumsum_w[-1]  # 归一化到总和为1
        #     indices_90_percent = indices[cumulative_percentage <= threshold]

        #     config_variable.total_critical_token_num+= len(indices_90_percent)
        #     print(f"达到90%激活次数所需的token数量: {config_variable.total_critical_token_num}")

        #     #获取是image的token id
        #     image_token_indices = indices_90_percent[(indices_90_percent >= 15) & (indices_90_percent < 15+image_token_num)]
        #     for idx in image_token_indices:
        #         config_variable.image_token_set[idx-15]+=1
            
        #     total_sum = sum(config_variable.image_token_set)

        #     threshold = total_sum * 0.8
        #     sorted_indices = sorted(range(len(config_variable.image_token_set)), key=lambda i: config_variable.image_token_set[i], reverse=True)
        #     cumulative_sum = 0
        #     num_tokens_for_90_percent = 0

        #     for idx in sorted_indices:
        #         cumulative_sum += config_variable.image_token_set[idx]
        #         num_tokens_for_90_percent += 1
                
        #         if cumulative_sum >= threshold:
        #             break
            
        #     # print(f"达到90%激活次数所需的图像token数量: {num_tokens_for_90_percent}")




        #     non_zero_count = sum(1 for x in config_variable.image_token_set if x != 0)

        #     if total_sum != 0:
        #         ratio = non_zero_count / total_sum
        #     else:
        #         ratio = 0  # 如果总和为0，则比率为0

            # print(f"非零部分数量占比: {ratio}, 非零部分数量: {non_zero_count}, 总和: {total_sum}")



        # 可视化热力图
        # global lunshu_cnt
        # lunshu_cnt += 1
        # if Q>1 and layer_idx==14:
        #     text_to_image_attention = attn_prob[:, :, 15+image_token_num:, 15:15+image_token_num]
        #     avg_per_head = text_to_image_attention.mean(dim=(0,1,2))
        #     # text_to_image_attention = attn_prob[:, :, -1, 15:15+image_token_num]
        #     # avg_per_head = text_to_image_attention.mean(dim=(0,1))
        #     avg_values = avg_per_head.cpu().numpy()

        #     visualize_attention_map("../images/dog2.png", avg_values,layer_num = layer_idx,token_num=image_token_num,lunshu_cnt=lunshu_cnt)
        # elif Q==1 and layer_idx==14:
        #     text_to_image_attention = attn_prob[:, :, :, 15:15+image_token_num]
        #     avg_per_head = text_to_image_attention.mean(dim=(0,1,2))
        #     avg_values = avg_per_head.cpu().numpy()
        #     visualize_attention_map("../images/dog2.png", avg_values,layer_num = layer_idx,token_num=image_token_num,lunshu_cnt=lunshu_cnt)



        # sample_query_start = 0 if Q == 0 else 15+image_token_num  # 只取前几个 query
        

        # attn_weights_cpu = attn_prob[0, :sample_heads,:sample_query, :].float()

        # global lunshu_cnt, image_attn_cum
        # #这边开始测试CDF曲线
        # if sample_query == 1 and layer_idx == 14:

        #     scores_sorted, indices = torch.sort(attn_weights_cpu.mean(dim=0).squeeze(0),descending=True)

        #     cumsum_scores = torch.cumsum(scores_sorted, dim=0)
        #     total_score = cumsum_scores[-1].clamp(min=1e-12)
        #     ratio = cumsum_scores / total_score

        #     # 找到达到95%的位置
        #     k95 = int(torch.searchsorted(ratio, 0.90).item()) + 1

        #     # 获取前95% attention的indices
        #     top_95_indices = indices[:k95]

        #     # 定义图像token的范围
        #     image_start = 15
        #     image_end = 15 + image_token_num

        #     # 找到在图像token范围内的indices
        #     image_indices_in_95 = top_95_indices[(top_95_indices >= image_start) & (top_95_indices < image_end)]
        #     # 计算占比
        #     image_tokens_in_95_count = len(image_indices_in_95)
        #     image_tokens_ratio = image_tokens_in_95_count / image_token_num if image_token_num > 0 else 0

        #     # print(f"达到95% attention需要的token总数: {k95}")
        #     print(f"其中图像token数量 (15:{15+image_token_num}): {image_tokens_in_95_count}")
        #     print(f"图像token占全部image token数量的占比: {image_tokens_ratio:.4f} ({image_tokens_in_95_count}/{image_token_num})")
            # print(f"图像token占前95% attention token的占比: {image_tokens_in_95_count/k95:.4f}")

        #     if image_attn_cum is None:
        #         image_attn_cum = attn_weights_cpu.mean(dim=0).squeeze(0)
        #     else:
        #         image_attn_cum += attn_weights_cpu.mean(dim=0).squeeze(0)

        #         print(image_attn_cum.shape)



        #     scores = attn_weights_cpu.mean(dim=0).squeeze(0)  # [num_image_tokens]
        #     scores = scores / scores.sum()                    # 归一化为概率分布
        #     scores_sorted, indices = torch.sort(scores, descending=True)
        #     cumsum_w = torch.cumsum(scores_sorted, dim=0)

        #     if lunshu_cnt == 16:
        #         arr = cumsum_w.cpu().numpy()
        #         # 保存到文件
        #         with open("cdf_round16.txt", "w") as f:
        #             for v in arr:
        #                 f.write(f"{v:.6f}\n")
            
        #         with open("cdf_values_qwen2.txt", "w") as f:
        #             for v in cumsum_w:
        #                 f.write(f"{v.item():.6f}\n")
            
        #     torch.cuda.synchronize()
            


        #     lunshu_cnt+=1

        #     xs = torch.arange(1, len(cumsum_w)+1) / len(cumsum_w)
        #     ys = cumsum_w

        #     xs = xs.cpu().numpy()
        #     ys = ys.cpu().numpy()


        #     fig, ax = plt.subplots(figsize=(7,3))
        #     ax.plot(xs*100, ys*100, color="darkorange", linewidth=2, label="Image tokens CDF")

        #     # 1) 只保留横向网格线
        #     ax.grid(axis="y", linestyle="--", alpha=0.5)

        #     # 2) 画 y=90% 与曲线的交点及“到交点为止”的横虚线
        #     target = 0.90
        #     idx = np.searchsorted(ys, target)
        #     if 0 < idx < len(ys):
        #         # 线性插值，求交点 x*
        #         x0, y0 = xs[idx-1], ys[idx-1]
        #         x1, y1 = xs[idx],     ys[idx]
        #         x_star = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
        #     else:
        #         # 边界情况（曲线一开始就>=target 或永远不到 target）
        #         x_star = xs[0] if idx == 0 else xs[-1]

        #     # 横虚线：0 -> x*
        #     ax.hlines(y=target*100, xmin=0, xmax=x_star*100, linestyles="--", linewidth=1, colors="gray")

        #     # 竖虚线：交点 -> x轴
        #     ax.vlines(x=x_star*100, ymin=0, ymax=target*100, linestyles="--", linewidth=1, colors="gray")

        #     # 在 x 轴标注 ratio
        #     ax.text(x_star*100, -5, f"{x_star*100:.1f}%", 
        #             ha="center", va="top", fontsize=10)

        #             # 在 y 轴标注 90%
        #     ax.text(-2, target*100, "90%", ha="right", va="center", fontsize=10)

            
        #     plt.xlabel("Top-k token ratio (%)")
        #     plt.ylabel("Cumulative attention (%)")
        #     plt.xlim(0, 100)
        #     plt.ylim(0, 100)
        #     plt.grid(linestyle="--", alpha=0.5)
        #     plt.legend(loc="lower right")
        #     plt.tight_layout()
        #     plt.savefig(f"image_token_cdf_layer{layer_idx}_{lunshu_cnt}.png")
        #     lunshu_cnt+=1






        # 建议放循环外的轻量超参
        # LAMBDA_LAYER = 0.7   # 分层EMA，越接近当前层权重越大
        # ENT_TAU      = 0.5   # 头部权重温度（越小越偏向低熵头）
        # SHARP_POW    = 1.5   # 分数锐化幂次 (>1 提升高分token占优)

        # global shallow_layers_attn_score, tmp_record, lunshu_cnt
        # lunshu_cnt += 1

        # if sample_query == 1:
        #     # 取当前层视图 [H, Q, T]
        #     cur_attn = attn_prob[0, :sample_heads, :sample_query, :]

        #     if layer_idx == 0:
        #         shallow_layers_attn_score = cur_attn.clone()
        #         tmp_record[layer_idx] = set()
        #     else:
        #         # —— 当前层真实 top-k（你的 k=150 保持不变）——
        #         k = 150
        #         current_score = cur_attn.mean(dim=(0, 1))               # [T]
        #         current_topk  = torch.topk(current_score, k).indices    # [k]
        #         set_current   = set(current_topk.tolist())

        #         # 上一次记录（可能不存在）
        #         prev_seen = tmp_record[layer_idx]
        #         new_set   = set_current - prev_seen                     # 分母集合（保持不变）

        #         if lunshu_cnt >= 2:
        #             # ========= 用“浅层”预测 =========
        #             # 1) 归一化得到浅层概率，避免头/查询规模差异
        #             denom = shallow_layers_attn_score.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # [H,Q,1]
        #             shallow_prob = shallow_layers_attn_score / denom                              # [H,Q,T]

        #             # 2) 低熵头加权（越自信权重越大）
        #             ent = -(shallow_prob * shallow_prob.clamp_min(1e-9).log()).sum(dim=-1)   # [H,Q]
        #             head_conf = (-ent.mean(dim=1) / ENT_TAU).softmax(dim=0)                  # [H]

        #             # 3) 聚合出浅层分数：先均值掉 query，再按头权重聚合
        #             p_qmean = shallow_prob.mean(dim=1)                                       # [H,T]
        #             shallow_score = (head_conf.unsqueeze(-1) * p_qmean).sum(dim=0)           # [T]

        #             # 4) 分数锐化（拉开高分差距）
        #             if SHARP_POW != 1.0:
        #                 shallow_score = shallow_score.pow(SHARP_POW)

        #             # 5) 邻域增强（容忍±1位置漂移）
        #             neigh = torch.maximum(shallow_score.roll(1), shallow_score.roll(-1))
        #             shallow_score = torch.maximum(shallow_score, neigh)

        #             # 6) 动态选择浅层 topk，至少覆盖分母规模
        #             k_shallow = max(50, len(new_set), 1)
        #             shallow_topk = torch.topk(shallow_score, k_shallow).indices
        #             set_shallow  = set(shallow_topk.tolist())

        #             # ======== 你的“Jaccard”公式（保持不变）========
        #             denom_sz = max(len(new_set), 1)   # 防止除0
        #             jaccard = len(set_shallow & new_set) / denom_sz

        #             # print(f"{len(new_set)}")  # 你原来的打印
        #             # 如需打印得分：
        #             print(f"[layer {layer_idx}] Jaccard(top-new={denom_sz}): {jaccard:.4f}")

        #         # 记录当前层真实集合，供下一次差分
        #         tmp_record[layer_idx] = set_current

        #         # 放在对比之后再更新，避免信息泄露
        #         shallow_layers_attn_score = (
        #             LAMBDA_LAYER * shallow_layers_attn_score +
        #             (1.0 - LAMBDA_LAYER) * cur_attn
        #         )


        # global shallow_layers_attn_score,tmp_record,lunshu_cnt
        # lunshu_cnt+=1
        # # initial: 0~15
        # initial_idx = set(range(0, 16))

        # # recent: K-20 ~ K-1
        # recent_idx = set(range(max(0, K-50), K))





        # # 合并
        # gpu_fixed = initial_idx | recent_idx
        # if sample_query == 1:    
        #     if layer_idx == 0:
        #         # shape: [heads, query, tokens]
        #         shallow_layers_attn_score = attn_prob[0, :sample_heads, :sample_query, :].clone()
        #     else:
                
        #         current_score = attn_prob[0, :sample_heads, :sample_query, :].mean(dim=(0,1))
                
        #         k = 150
                
        #         current_topk = torch.topk(current_score, k).indices.cpu().numpy()

        #         set_current = set(current_topk.tolist())

        #         if lunshu_cnt >=2 and layer_idx >=3:

        #             # cur_attn: [H, Q, T]
        #             prob = shallow_layers_attn_score / shallow_layers_attn_score.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # 归一化
        #             ent = -(prob * prob.clamp_min(1e-9).log()).sum(dim=-1).mean(dim=1)   # [H] 每个head的平均熵
        #             # 取前 top_h 个最小熵的 head
        #             top_h = 3
        #             important_heads = ent.topk(top_h, largest=False).indices  # [top_h]


        #             # 只用这些 head 做平均
        #             shallow_score = shallow_layers_attn_score[important_heads].mean(dim=(0,1))  # [T]
        #             # shallow_score = shallow_layers_attn_score.mean(dim=(0,1))  # shape: [tokens]
        #             shallow_topk = torch.topk(shallow_score, 150).indices
                    
        #             set_shallow = set(shallow_topk.tolist())

        #             # src_cpu = shallow_layers_attn_score.detach().to('cpu', non_blocking=False)

        #             # torch.cuda.synchronize()
        #             # import time
        #             # start_2 = time.time()
                    

        #             write_index = set_shallow - gpu_fixed - (tmp_record[layer_idx] - gpu_fixed)
        #             #写入的index 
        #             write_index_tensor = torch.tensor(sorted(list(write_index)), dtype=torch.long).to("cpu")


        #             # torch.cuda.synchronize()
        #             # end_2 = time.time()
        #             # print(f"其他操作耗时: {end_2 - start_2:.6f} 秒")

        #             # pinned memory 预分配 (注意 pin_memory=True)
        #             # shape = [H, Q, len(write_index)]

        #             # pinned_buf = torch.empty(
        #             #     (sample_heads, sample_query, len(write_index_tensor)),
        #             #     dtype=shallow_layers_attn_score.dtype,
        #             #     pin_memory=True
        #             # )


        #             # print(f"{len(write_index_tensor)}")
                    
                    
        #             # torch.cuda.synchronize()
        #             # import time
        #             # start = time.time()

        #             # # 将对应 index 写入 pinned memory
        #             # torch.index_select(
        #             #     src_cpu , dim=-1, index=write_index_tensor, out=pinned_buf
        #             # )

        #             # torch.cuda.synchronize()
        #             # end = time.time()
        #             # print(f"写入 pinned mem 耗时: {end - start:.6f} 秒")
                                        

        #             jaccard = len((set_shallow - gpu_fixed - tmp_record[layer_idx]) & (set_current - tmp_record[layer_idx] - gpu_fixed)) \
        #                     / max(len(set_current - tmp_record[layer_idx] - gpu_fixed), 1)

        #             # with open

        # #             # jaccard = len(set_shallow & (set_current-tmp_record[layer_idx])) / len(set_current-tmp_record[layer_idx])

        #             # 这里可以打印原本需要维护多少
        #             # with open("original_write.txt","a") as f:
        #             #     f.write(f"{len(set_current-tmp_record[layer_idx] - set_shallow - gpu_fixed)}\n")
                    
        #             # print(f"{len(set_current-tmp_record[layer_idx])}")

        #             # print(f"Jaccard similarity of top-{k}: {jaccard:.4f}, {len((set_shallow - gpu_fixed - tmp_record[layer_idx]) & (set_current - tmp_record[layer_idx] - gpu_fixed))}, {max(len(set_current - tmp_record[layer_idx] - gpu_fixed), 1)},{len(set_shallow & set_current)}")

        #         tmp_record[layer_idx] = set_current

        #         shallow_layers_attn_score = shallow_layers_attn_score*0.5 +  attn_prob[0, :sample_heads, :sample_query, :]



        ## 实验8 统计image token的attention和随着decoding的变化
        # if sample_query == 1 and layer_idx==14:
        #     attn_image = attn_prob[0, :,0,:]
        #     # attn_image = attn_image.mean(dim=0).sum().item() * 100
        #     attn_score = attn_image.mean(dim=0)
        #     print(attn_score.shape)

        #     top_k = 80
        #     top_indices = torch.topk(attn_score, k=min(top_k, len(attn_score))).indices
        #     current_top_tokens = set(top_indices.cpu().numpy())

        #     global prev_top_tokens
        #     if prev_top_tokens is None:
        #         prev_top_tokens = current_top_tokens
        #     else:
        #         intersection = prev_top_tokens.intersection(current_top_tokens)
        #         union = prev_top_tokens.union(current_top_tokens)
        #         sim = len(intersection) / len(union)    
        #         print(f"{sim:.4f}")



            # with open("image_receive_attn.txt", "a") as f:
            #     f.write(f"{attn_image}\n")
            


        ##实验7 在这下方尝试使用dual-attention filter
        # if sample_query > 1:
        #     global important_image

        #     # torch.cuda.synchronize()
        #     # dual_attn_start = time.time()
        #     important_image_indices , _ = dual_attention_filter(attn_prob,15,image_token_num,0.001,0.01)
            # torch.cuda.synchronize()
            # dual_attn_end = time.time()
            # print(f"dual attn time: {dual_attn_end - dual_attn_start}")

        #     important_image[layer_idx] = important_image_indices

        # if sample_query == 1:
        #     # 假设 batch_size = 1
        #     b_idx = 0
        #     q_idx = 0

        #     # 取出当前 query 的注意力分布，先对 head 取平均
        #     # attn_prob: [B, H, Q, K]
        #     attn_mean = attn_prob[b_idx, :, q_idx, :].mean(dim=0)  # [K]

        #     # 取 top-150 的 key token
        #     topk = 200
        #     top_values, top_indices = torch.topk(attn_mean, k=topk, dim=-1)  # [150]

        #     # 图像 token 的范围
        #     img_start = 15
        #     img_end = 15 + image_token_num
        #     # print(image_token_num)

        #     # 选出 top150 里面的图像 token
        #     img_in_topk = top_indices[(top_indices >= img_start) & (top_indices < img_end)]

        #     # 转换成相对索引 (0 ~ image_token_num-1)
        #     img_in_topk_relative = img_in_topk - img_start

        #     # 从保存的 important_image 中取出当前层的结果
        #     important_indices = important_image[layer_idx]  # tensor([...])
        #     # print(len(important_indices))

        #     # 计算交集 (哪些图像 token 同时在 top150 和 important_image 里)
        #     overlap = torch.isin(img_in_topk_relative, important_indices)

        #     if(len(img_in_topk_relative) > 0):
        #         with open("image_overlap.txt", "a") as f:
        #             f.write(f"{overlap.sum().item() / len(img_in_topk_relative)}\n")

            # if(len(img_in_topk_relative) > 0):
            #     print(f"layer:{layer_idx} {overlap.sum().item() / len(img_in_topk_relative)},   {len(img_in_topk_relative)}")

            


        

        # 实验5 尝试找outlier 在prefill阶段
        # if sample_query > 1:
            # plot_hidden_dim_scores(q, k, reduce="mean", norm="zscore",layer_idx=layer_idx)
            # save_hidden_dim_scores_per_head(q,k,reduce="mean",norm="zscore",layer_idx=layer_idx)
            # print(q.shape)
            # print(k.shape)

        

        # 实验4 验证不同head的top集合相似度
        # result = plot_overlap_heatmap(attn_weights_cpu, k=(int)(image_token_num*0.1),layer_idx=layer_idx)
            # importance, entropy, entropy_norm, order = head_entropy(attn_weights_cpu,already_prob=True)
            # for h in range(entropy.numel()):
            #     print(f"Head {h:02d} - 熵: {entropy[h]:.4f}  (归一化: {entropy_norm[h]:.4f}, 重要性: {importance[h]:.4f})")

        # 实验3 验证图片prefill 阶段的 dual-attention filter
        # if sample_query > 1:
        #     attn = attn_prob[0]
        #     v_token_start = 15
        #     img_seq = image_token_num
        #     text_seq = sample_query - img_seq - v_token_start
        #     attn = attn.narrow(1, v_token_start+img_seq, text_seq).narrow(2, v_token_start, img_seq) # [nhead, text_seq, img_seq] 
            
        #     attn = attn.mean(dim=0)  # [nhead, text_seq, img_seq] -> [text_seq, img_seq]
        #     attn_sum = attn.sum(dim=0)  # [text_seq, img_seq] -> [img_seq], global-attention
            

        #     # V = attn.sum(dim=0)              # = attn_sum
        #     # I = attn.max(dim=0).values

        #     # # 自适应得到 gamma_th / alpha_th
        #     # _, _, (gamma_th, alpha_th, ll_ratio) = dual_attention_filter_adaptive(
        #     #     attn_prob[0],              # 你的 cross-attention 张量
        #     #     v_token_start,     # 视觉 token 在序列中的起始位置
        #     #     img_seq,           # 视觉 token 数量
        #     #     target_ll=0.6,    # 目标左下象限比例（可调，比如 0.7~0.8）
        #     #     max_iter=25,       # 二分搜索迭代次数（默认够用）
        #     #     eps=1e-6           # 防止 quantile 出现除0问题
        #     # )

        #     # candidate_mask = (V <= gamma_th) & (I <= alpha_th)

        #     # candidate_mask = (attn_sum < attn_sum.sum(dim=0)*global_thr) & (attn.max(dim=0)[0] < individual_thr)
        #     # all_indices = torch.arange(img_seq, device=attn.device)

        #     # Important token index needs to be retained
        #     # important_indices = all_indices[~candidate_mask]


        #     topk_num = min(1000, attn_sum.size(0))  # 防止 img_seq < 100
        #     important_indices = torch.topk(attn_sum, topk_num, largest=True).indices
        #     prefill_top_image_token[layer_idx] = set(important_indices.tolist())


        # print(attn_weights_cpu.shape)
        ### 下方是实验1  每一层激活的kv与第0层的比较 集合间的关系 包含度，可以用于浅层管理深层的pinned mem
        # if sample_query == 1:
        #     global lunshu_cnt,layer1_diff,layer0_3_attn_accu,helper
        # #     # if layer_idx <=2:
        # #     #     top_token_cur_layer = top_token_set_per_layer(attn_weights_cpu, threshold=0.9 )
        # #     # else:          # if layer_idx <=2:
        # #     #     top_token_cur_layer = top_token_set_per_layer(attn_weights_cpu, threshold=0.95 )

        # #     # if layer_idx <=2:
        # #     #     # top_token_cur_layer,tmp_record[layer_idx] = top_token_set_per_layer(attn_weights_cpu, threshold=0.8,image_len=image_token_num)
        # #     #     top_token_cur_layer,tmp_record[layer_idx] = top_token_set_per_layer_num(attn_weights_cpu, top_k=30,image_len=image_token_num)
        # #     #     if layer_idx == 0:
        # #     #         layer0_3_attn_accu = attn_weights_cpu
        # #     #     else:
        # #     #         layer0_3_attn_accu += attn_weights_cpu
        # #     #         _,helper = top_token_set_per_layer_num(layer0_3_attn_accu,top_k=80,image_len=image_token_num)

        # #     # else:
        #     if layer_idx>=5:
        #         top_token_cur_layer,tmp_record[layer_idx] = top_token_set_per_layer(attn_weights_cpu, threshold=0.9,image_len=image_token_num)
        #         # top_token_cur_layer,tmp_record[layer_idx] = top_token_set_per_layer_num(attn_weights_cpu, top_k=100,image_len=image_token_num)

        #     if layer_idx >=5:
        #         # jaccard_similarity_value = jaccard_similarity(
        #         #     record_top_token_per_layer[layer_idx],top_token_cur_layer
        #         # )
        #         # print(f"第{layer_idx}层，与第{layer_idx-1}层的 Jaccard 相似度: {jaccard_similarity_value:.4f}")

        #         #比较当前层被0 1层包含的程度
        #         # containment_score_value = containment_score(
        #         #     record_top_token_per_layer[1]|record_top_token_per_layer[2], top_token_cur_layer, empty_policy="one"
        #         # )
        #         # print(f"{len(record_top_token_per_layer[1])}")

        #         #比较与相邻decoding的相似度
        #         if layer_idx == 14:
        #             containment_score_value = containment_score(
        #                 record_top_token_per_layer[layer_idx], top_token_cur_layer, empty_policy="one"
        #             )
        #             # print(containment_score_value)
        #             with open(f"adjacent_decoding_same_layer_critical_token_demp_layer14.txt","a", encoding="utf-8") as f:
        #                 f.write(f"{containment_score_value:.4f}\n")


        #         # containment_score_value = containment_score(
        #         #     top_token_cur_layer,record_top_token_per_layer[1]|record_top_token_per_layer[0], empty_policy="one"
        #         # )

        #         # print(len(record_top_token_per_layer[1]-top_token_cur_layer))


        #         # diff_set = top_token_cur_layer - record_top_token_per_layer[layer_idx]
        #         # positions2 = [i for i, val in enumerate(tmp_record[layer_idx-1])
        #         #     if val in record_top_token_per_layer[1]]
        #         # with open(f"tmp11.txt","a", encoding="utf-8") as f:
        #         #     f.write(f"positions: {positions2}\n")
                
        #         # print(f"第{layer_idx}层len(diff_set):{len(diff_set)}，前一层大小为{len(record_top_token_per_layer[layer_idx])}")

        #         # positions = [i for i, val in enumerate(tmp_record[layer_idx-1])
        #         #  if val in top_token_cur_layer - record_top_token_per_layer[layer_idx]]
        #         # # count = len(diff_set & (set(tmp_record[1])|set(tmp_record[0])))
        #         # # print("count:",count)
        #         # with open(f"tmp15.txt","a", encoding="utf-8") as f:
        #         #     f.write(f"positions: {positions}\n")


        #         # not_pos = [i for i,val in enumerate(helper) if val not in record_top_token_per_layer[layer_idx] and val not in set(positions)]
        #         # with open(f"tmp24.txt","a", encoding="utf-8") as f:
        #         #     f.write(f"not_pos: {not_pos}\n")
        #         # print(len(not_pos))


        #         # print(len(record_top_token_per_layer[1]))

        #         # print(f"{len(top_token_cur_layer)}，{len(record_top_token_per_layer[0])}")

        #         # print(f"{len((record_top_token_per_layer[1]|record_top_token_per_layer[0])-record_top_token_per_layer[layer_idx])}")
                

        #         #图像image和prefill阶段的token的包含度
        #         # containment_score_value = containment_score(
        #         #     prefill_top_image_token[layer_idx], top_token_cur_layer, empty_policy="one"
        #         # )
        
        #     # if layer_idx==1:
        #     #     layer1_diff = top_token_cur_layer - record_top_token_per_layer[layer_idx]

        #         record_top_token_per_layer[layer_idx] = top_token_cur_layer

            # if lunshu_cnt % 20 == 0:
            #     if layer_idx > 3 and lunshu_cnt>0:
            #         containment_score_value = containment_score(
            #             record_top_token_per_layer[layer_idx], top_token_cur_layer, empty_policy="one"
            #         )
            #         print(f"第{layer_idx}层，对第层的包含度: {containment_score_value:.4f}")
            #     record_top_token_per_layer[layer_idx] = top_token_cur_layer
            # if layer_idx == 0:
            #     lunshu_cnt += 1





        ### 下方是实验2  用于统计，各类 token的注意力占比
        # # 
        # part_score = attn_weights_cpu.mean(dim=0).sum().item()

        # #如果想要看占比（相对该 query 所有 token 的总注意力）

        # full_score_sum = attn_prob[0, :sample_heads, :sample_query, :].float().mean(dim=0).sum().item()
        
        # partial_score_ratio = part_score / full_score_sum

        # # print(f"第{layer_idx}层，图像 token attn score 占比: {partial_score_ratio:.4f}")

        # # global layer_attn_cum
        # # layer_attn_cum[layer_idx] += partial_score_ratio

        # # print(f"图像 token attn score 总和: {image_score_sum:.4f}")
        # with open("attn_ratio_log_cats_txt.txt", "a", encoding="utf-8") as f:  # "a" 表示追加写入
        #     f.write(f"第{layer_idx}层 ，text token attn score 占比: {partial_score_ratio:.4%}\n")




        #补充2.1实验 假如prefill阶段 最后一个token
        # if sample_query ==1:
        #     last_q = attn_weights_cpu[:,-1,:]
        #     scores = last_q.mean(dim=0)
        #     #这是取prefill的最后一个token
        #     # attn_weights_cpu = attn_prob[0, :sample_heads, -1, :].float()
        #     # scores_sorted, indices = torch.sort(attn_weights_cpu.mean(dim=0).squeeze(0),descending=True)

        #     scores_sorted, indices = torch.sort(scores, descending=True)
        #     cumsum_scores = torch.cumsum(scores_sorted, dim=0)
        #     ratio = cumsum_scores / cumsum_scores[-1].clamp(min=1e-12)

        #     # step4: 找到第一次达到 >= 95% 的位置
        #     k95 = int(torch.searchsorted(ratio, 0.9).item()) + 1

        #     print(f"第{layer_idx}层，达到95%需要的token数: {k95}，占比: {k95 / K:.4f} ({k95}/{K})")
        #     with open("critical_token_num.txt","a",encoding="utf-8") as f:
        #         f.write(f"prefill 第{layer_idx}层，达到95%需要的token数: {k95}，占比: {k95 / K:.4f} ({k95}/{K})\n")


        #下方是实验2.1  用于统计，对于某类token，各层需要多少 token 才能达到 98% 的注意力覆盖  聚焦于decoding阶段
        # 3) 排序 + 累计和
        # if sample_query == 1:
        #     scores_sorted, indices = torch.sort(attn_weights_cpu.mean(dim=0).squeeze(0),descending=True)
        #     cumsum_scores = torch.cumsum(scores_sorted, dim=0)
        #     ratio = cumsum_scores / cumsum_scores[-1].clamp(min=1e-12)

        #     # step4: 找到第一次达到 >= 95% 的位置
        #     k95 = int(torch.searchsorted(ratio, 0.90).item()) + 1

        # #     # print(f"第{layer_idx}层，达到95%需要的token数: {k95}，占比: {k95 / K:.4f} ({k95}/{K})")
        #     with open("critical_token_num.txt","a",encoding="utf-8") as f:
        #         f.write(f"第{layer_idx}层，达到95%需要的token数: {k95}，占比: {k95 / K:.4f} ({k95}/{K})\n")
        # #     layer_attn_cum[layer_idx] += k95

        #     for idx in indices[:k95]:
        #         imgage_token_attn_collect[idx] += 1
            
            # with open("tmp.txt","a",encoding="utf-8") as f:
            #     f.write(f"imgage_token_attn_collect: {imgage_token_attn_collect}\n")

        

        #下方是实验2.2  用于统计image token,激活次数（占整体，上方是占imagetoken自身)
        # 
        # cnt_image_tmp = 0
        # cnt_text_tmp = 0
        # if sample_query == 1:
        #     scores_sorted, indices = torch.sort(attn_weights_cpu.mean(dim=0).squeeze(0),descending=True)
        #     cumsum_scores = torch.cumsum(scores_sorted, dim=0)
        #     ratio = cumsum_scores / cumsum_scores[-1].clamp(min=1e-12)

        #     # print(cumsum_scores[-1])

        #     # step4: 找到第一次达到 >= 95% 的位置
            
            # k95 = int(torch.searchsorted(ratio, 0.95).item()) + 1
        #     print(f"第{layer_idx}层，达到95%需要的token数: {k95}")
        #     # print(15+image_token_num)
            # for idx in indices[:k95]:
        # #         # print(idx)
                # if idx >=15 and idx <15+image_token_num:
                    # cnt_image_tmp +=1
        #         if idx > 15 + image_token_num:
        #             cnt_text_tmp +=1

        #     print(f"第{layer_idx}层，image token 激活次数: {cnt_text_tmp}")
                    # print("hello")
                    # imgage_token_attn_collect[idx-15] += 1
        #     with open("tmp.txt","a",encoding="utf-8") as f:
        #         f.write(f"imgage_token_attn_collect: {imgage_token_attn_collect}\n")




    
    # print(attn_prob.dtype)

    if dropout_p and training:
        attn_prob = F.dropout(attn_prob, p=dropout_p, training=True)

    # 4) 乘 V 得到输出
    attn_out = torch.matmul(attn_prob.type_as(v), v)  # [B,H,Tq,D]

    return attn_out, attn_prob, attn_logits

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)



def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2_5_VLAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)


        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"attn_output should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



last_attn_input = None
class Qwen2_5_VLSdpaAttention(Qwen2_5_VLAttention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        cpu_kv_cache: Optional[CPUKVCache] = None,     #new
        is_prefill:Optional[bool] = None,
        layer_idx: Optional[int] = None,
        image_len: Optional[int] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 这个是使用原始的attention 不知道为什么使用原始的attn输出内容不太对 
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2_5_VLModel is using Qwen2_5_VLSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        

        # 测量 输入hidden_states的相似度
        # global last_attn_input

        # if layer_idx >=1:
        #     qf = hidden_states.flatten(start_dim=1) 
        #     lf = last_attn_input.flatten(start_dim=1)
        #     qf_n = F.normalize(qf, p=2, dim=1, eps=1e-12)
        #     lf_n = F.normalize(lf, p=2, dim=1, eps=1e-12)
        #     cos_sim = (qf_n * lf_n).sum(dim=1).item()
        #     print(f"第{layer_idx}层余弦相似度:", cos_sim)

        # last_attn_input = hidden_states 

        bsz, q_len, _ = hidden_states.size()
        # print(hidden_states.shape)


        # nvtx.range_push("My project Layer")
        # torch.cuda.synchronize()
        # proj_start = time.time()
        query_states = self.q_proj(hidden_states)
        
        # global last_attn_input

        # if layer_idx >=1:
        #     qf = query_states.flatten(start_dim=1) 
        #     lf = last_attn_input.flatten(start_dim=1)
        #     qf_n = F.normalize(qf, p=2, dim=1, eps=1e-12)
        #     lf_n = F.normalize(lf, p=2, dim=1, eps=1e-12)
        #     cos_sim = (qf_n * lf_n).sum(dim=1).item()
        #     print(f"第{layer_idx}层余弦相似度:", cos_sim)

        # last_attn_input = query_states 





        # print(query_states.shape)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # torch.cuda.synchronize()
        # proj_end = time.time()
        # print(f"proj time: {proj_end - proj_start}")

        # torch.cuda.synchronize()
        # attn_start = time.time()


        if is_prefill:

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            if cpu_kv_cache is not None:
                ok = cpu_kv_cache.submit_append(self.layer_idx, key_states, value_states, block=False)

            # if cpu_kv_cache is not None:
            #     threading.Thread(
            #     target=cpu_kv_cache.append,
            #     args=(self.layer_idx, key_states, value_states),
            #     daemon=True
            # ).start()
        else:
            key_states_new = repeat_kv(key_states, self.num_key_value_groups)
            value_states_new = repeat_kv(value_states, self.num_key_value_groups)

            # === Dynamic Sparse Attention 配置 ===
            use_sparse_attn = True  # 可以通过参数控制是否启用
            sparse_top_k_ratio = 0.3  # 保留 30% 的重要 token
            sparse_min_tokens = 128   # 最少保留 128 个 token

            if cpu_kv_cache is not None and use_sparse_attn:
                # 1. 先获取当前层的完整 KV（用于计算重要性）
                key_full, value_full = CPUKVCache.get_full_kv(past_key_value, key_states_new, value_states_new)

                # 2. 计算 token 重要性（使用 query 和 key 的相似度）
                # query_states: [B, H, Q, D], key_full: [B, H, T, D]
                with torch.no_grad():
                    important_indices = compute_token_importance(
                        query_states,
                        key_full,
                        method="attention_score",
                        top_k_ratio=sparse_top_k_ratio,
                        min_tokens=sparse_min_tokens,
                    )

                # 3. 异步获取选中的重要 token 的 KV（使用UVA零拷贝）
                sparse_result = cpu_kv_cache.get_sparse_async_uva(
                    self.layer_idx,
                    query_states.device,
                    important_indices,
                )

                if sparse_result is not None:
                    key_sparse, value_sparse, done_event = sparse_result

                    # 4. 等待 H2D 完成
                    torch.cuda.current_stream().wait_event(done_event)

                    # 5. 使用稀疏注意力计算
                    attn_output = sparse_sdpa_attention(
                        query_states,
                        key_full,           # 完整 KV（备用）
                        value_full,
                        key_sparse,         # 重要 token 的 KV
                        value_sparse,
                        important_indices.to(query_states.device),
                        dropout_p=self.attention_dropout if self.training else 0.0,
                    )

                    # 跳过常规的 SDPA 计算
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.view(bsz, q_len, self.hidden_size)
                    attn_output = self.o_proj(attn_output)
                    return attn_output, None, past_key_value
                else:
                    # 稀疏加载失败，回退到完整 KV
                    key_states, value_states = key_full, value_full

            elif cpu_kv_cache is not None:
                # 这个是把以前的kv和新的kv拼接起来
                key_states,value_states = CPUKVCache.get_full_kv(past_key_value, key_states_new, value_states_new)

            elif past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
                    
                



        # if cpu_kv_cache is not None:
            # ev, k_buf, v_buf, _ = cpu_kv_cache.append_async(self.layer_idx, key_states, value_states)
        # if cpu_kv_cache is not None and is_prefill:
        #     threading.Thread(
        #         target=cpu_kv_cache.append,
        #         args=(self.layer_idx, key_states, value_states),
        #         daemon=True
        #     ).start()
        
        # time.sleep(0.5)
        
        # if cpu_kv_cache is not None:
        #     event = cpu_kv_cache.append_async(self.layer_idx, key_states, value_states)
        #     cpu_kv_cache.last_append_event = event





        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)



        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False
        # print(query_states.shape)
        # print(key_states.shape)

       
        # attn_output = sdpa_with_scores(
        #     query_states,
        #     key_states,
        #     value_states,
        #     attn_mask=causal_mask,
        #     dropout_p=self.attention_dropout if self.training else 0.0,
        #     is_causal=is_causal,
        #     layer_idx = layer_idx,
        #     image_len = image_len,
        # )[0]

        # print(key_states.shape)
        # nvtx.range_push("My Attn Layer")
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        # nvtx.range_pop()


        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        
        # nvtx.range_push("My outproj Layer")
        attn_output = self.o_proj(attn_output)
        # nvtx.range_pop()

        # torch.cuda.synchronize()
        # attn_end = time.time()
        # print(f"attn time: {attn_end - attn_start}")

        return attn_output, None, past_key_value


QWEN2_5_VL_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLAttention,
    "sdpa": Qwen2_5_VLSdpaAttention,
}