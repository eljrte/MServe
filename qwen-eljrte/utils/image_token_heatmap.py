import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
def visualize_token_heatstrip(
    counts,                 # list/np.ndarray/torch.Tensor, 一维非负
    title="Token Activation Heatmap",
    pool_size=16,           # 池化窗口大小；=1 表示不池化
    pool_mode="mean",       # "mean" 或 "max"
    use_log=True,           # True: 对数增强 log1p；False: 线性
    gamma=None,             # 幂次增强（例如 1.2/1.5）；None 表示不用
    clip_percentiles=(1,99),# 分位裁剪增强对比
    cmap="coolwarm",        # 颜色图
    use_lognorm=False       # True 用 LogNorm（需非负且>0，可在内部+1）
):
    # ---- 转为 numpy 并清洗 ----
    try:
        import torch
        if isinstance(counts, torch.Tensor):
            counts = counts.detach().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(counts).astype(np.float64).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x[x < 0] = 0

    # ---- 1D 池化 ----
    if pool_size > 1:
        # padding 到 pool_size 的倍数
        n = len(x)
        pad = (-n) % pool_size
        if pad:
            x = np.pad(x, (0, pad), mode='edge')
        x = x.reshape(-1, pool_size)
        if pool_mode == "max":
            x = x.max(axis=1)
        else:
            x = x.mean(axis=1)

    # ---- 可选数值增强（log / gamma）----
    x_vis = x.copy()
    if use_log:
        x_vis = np.log1p(x_vis)   # 缓和大值，拉开低值
    if gamma is not None:
        # 归一化到 [0,1] 后做幂，再放回原尺度
        lo, hi = x_vis.min(), x_vis.max()
        if hi > lo:
            z = (x_vis - lo) / (hi - lo)
            z = z ** gamma
            x_vis = z * (hi - lo) + lo

    # ---- 分位裁剪增强对比 ----
    if clip_percentiles is not None:
        p1, p2 = np.percentile(x_vis, clip_percentiles)
        if p2 > p1:
            vmin, vmax = p1, p2
        else:
            vmin, vmax = None, None
    else:
        vmin = vmax = None

    # ---- 绘制一维热度条 ----
    heat = x_vis.reshape(1, -1)  # 1×M
    plt.figure(figsize=(12, 1.8))

    if use_lognorm:
        # LogNorm 需要 >0，这里整体 +1 以避免 0
        im = plt.imshow(heat + 1.0, aspect='auto', cmap=cmap,
                        norm=LogNorm(vmin=max(1.0, (vmin if vmin is not None else (heat+1).min())),
                                     vmax=(vmax if vmax is not None else (heat+1).max())))
    else:
        im = plt.imshow(heat, aspect='auto', cmap=cmap,
                        vmin=vmin, vmax=vmax, interpolation='nearest')

    plt.colorbar(im, label="Activation (enhanced)")
    plt.yticks([])  # 隐藏Y轴
    plt.xlabel(f"Token Index (pooled, size={pool_size})" if pool_size>1 else "Token Index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("image_token_activation_heatmap.png", dpi=300)
    plt.close()