"""
简化的KV Cache传输基准测试（纯PyTorch，无需编译）

运行:
    python benchmark_simple.py
"""

import torch
import time
import numpy as np

# 配置
BATCH_SIZE = 1
NUM_HEADS = 28
SEQ_LEN = 8192
HEAD_DIM = 128
NUM_SELECTED = 2048
NUM_ITERATIONS = 50
WARMUP = 10
DEVICE = "cuda:0"


def create_pinned_kv_cache():
    """创建CPU pinned KV Cache"""
    k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16).pin_memory()
    v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16).pin_memory()
    return k, v


def generate_scattered_indices():
    """生成零散索引"""
    # 确保分散：在整个范围内均匀分布
    chunk_size = SEQ_LEN // NUM_SELECTED
    indices = []
    for i in range(NUM_SELECTED):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, SEQ_LEN)
        idx = np.random.randint(start, max(end, start + 1))
        indices.append(idx)
    # 打乱顺序
    indices = torch.tensor(indices, dtype=torch.long)
    return indices[torch.randperm(len(indices))]


# ========== 方法1: 分散Memcpy ==========
def method_scattered_memcpy(k_cache, v_cache, indices):
    """对每个token单独Memcpy"""
    k_gpu = torch.empty(BATCH_SIZE, NUM_HEADS, NUM_SELECTED, HEAD_DIM,
                        dtype=k_cache.dtype, device=DEVICE)
    v_gpu = torch.empty(BATCH_SIZE, NUM_HEADS, NUM_SELECTED, HEAD_DIM,
                        dtype=v_cache.dtype, device=DEVICE)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for i, idx in enumerate(indices):
        k_gpu[:, :, i, :].copy_(k_cache[:, :, idx, :], non_blocking=True)
        v_gpu[:, :, i, :].copy_(v_cache[:, :, idx, :], non_blocking=True)

    torch.cuda.synchronize()
    return k_gpu, v_gpu, (time.perf_counter() - start) * 1000


# ========== 方法2: Gather+Memcpy ==========
def method_gather_memcpy(k_cache, v_cache, indices):
    """CPU gather后单次传输"""
    k_cpu_gathered = torch.empty(BATCH_SIZE, NUM_HEADS, NUM_SELECTED, HEAD_DIM,
                                  dtype=k_cache.dtype, pin_memory=True)
    v_cpu_gathered = torch.empty(BATCH_SIZE, NUM_HEADS, NUM_SELECTED, HEAD_DIM,
                                  dtype=v_cache.dtype, pin_memory=True)

    torch.cuda.synchronize()
    start = time.perf_counter()

    # CPU gather
    torch.index_select(k_cache, dim=2, index=indices, out=k_cpu_gathered)
    torch.index_select(v_cache, dim=2, index=indices, out=v_cpu_gathered)

    # 单次传输
    k_gpu = k_cpu_gathered.to(DEVICE, non_blocking=True)
    v_gpu = v_cpu_gathered.to(DEVICE, non_blocking=True)

    torch.cuda.synchronize()
    return k_gpu, v_gpu, (time.perf_counter() - start) * 1000


# ========== 方法3: 模拟UVA (Multi-Stream) ==========
def method_multistream_memcpy(k_cache, v_cache, indices):
    """
    使用多stream模拟UVA效果
    将indices分成多组，每组用不同stream并行处理
    """
    k_gpu = torch.empty(BATCH_SIZE, NUM_HEADS, NUM_SELECTED, HEAD_DIM,
                        dtype=k_cache.dtype, device=DEVICE)
    v_gpu = torch.empty(BATCH_SIZE, NUM_HEADS, NUM_SELECTED, HEAD_DIM,
                        dtype=v_cache.dtype, device=DEVICE)

    # 使用4个stream
    num_streams = 4
    streams = [torch.cuda.Stream(device=DEVICE) for _ in range(num_streams)]

    torch.cuda.synchronize()
    start = time.perf_counter()

    # 每个stream处理一部分indices
    chunk_size = (NUM_SELECTED + num_streams - 1) // num_streams
    for stream_idx in range(num_streams):
        s = stream_idx * chunk_size
        e = min(s + chunk_size, NUM_SELECTED)

        with torch.cuda.stream(streams[stream_idx]):
            for i in range(s, e):
                idx = indices[i].item()
                k_gpu[:, :, i, :].copy_(k_cache[:, :, idx, :], non_blocking=True)
                v_gpu[:, :, i, :].copy_(v_cache[:, :, idx, :], non_blocking=True)

    for s in streams:
        s.synchronize()

    torch.cuda.synchronize()
    return k_gpu, v_gpu, (time.perf_counter() - start) * 1000


# ========== 方法4: 预取+Gather ==========
def method_prefetch_gather(k_cache, v_cache, indices):
    """
    预先把数据从pinned memory读到CPU cache（warmup），再gather
    测试CPU cache的影响
    """
    # 预读一次（warmup CPU cache）
    _ = k_cache.sum() + v_cache.sum()

    return method_gather_memcpy(k_cache, v_cache, indices)


def run_benchmark():
    print("=" * 70)
    print("KV Cache传输基准测试 (纯PyTorch)")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  Seq Len: {SEQ_LEN}, Selected: {NUM_SELECTED}")
    print(f"  Cache: {2 * BATCH_SIZE * NUM_HEADS * SEQ_LEN * HEAD_DIM * 2 / 1024**3:.2f} GB")
    print(f"  Selected Data: {2 * BATCH_SIZE * NUM_HEADS * NUM_SELECTED * HEAD_DIM * 2 / 1024**2:.2f} MB")

    # 创建数据
    print("\n创建CPU pinned KV Cache...")
    k_cache, v_cache = create_pinned_kv_cache()
    print(f"  K shape: {k_cache.shape}, pinned: {k_cache.is_pinned()}")

    # 生成零散索引
    print(f"\n生成零散索引 ({NUM_SELECTED} tokens)...")
    indices = generate_scattered_indices()
    sorted_idx = torch.sort(indices).values
    gaps = sorted_idx[1:] - sorted_idx[:-1]
    print(f"  范围: [{indices.min().item()}, {indices.max().item()}]")
    print(f"  平均间隔: {gaps.float().mean().item():.1f}")

    # 测试
    results = {
        "scattered_memcpy": [],
        "gather_memcpy": [],
        "multistream_memcpy": [],
        "prefetch_gather": [],
    }

    print(f"\n预热 ({WARMUP} iterations)...")
    for _ in range(WARMUP):
        method_scattered_memcpy(k_cache, v_cache, indices)
        method_gather_memcpy(k_cache, v_cache, indices)
        method_multistream_memcpy(k_cache, v_cache, indices)
        method_prefetch_gather(k_cache, v_cache, indices)

    print(f"正式测试 ({NUM_ITERATIONS} iterations)...")

    print("\n[1/4] 分散Memcpy...")
    for _ in range(NUM_ITERATIONS):
        _, _, t = method_scattered_memcpy(k_cache, v_cache, indices)
        results["scattered_memcpy"].append(t)

    print("[2/4] Gather+Memcpy...")
    for _ in range(NUM_ITERATIONS):
        _, _, t = method_gather_memcpy(k_cache, v_cache, indices)
        results["gather_memcpy"].append(t)

    print("[3/4] Multi-Stream (模拟UVA)...")
    for _ in range(NUM_ITERATIONS):
        _, _, t = method_multistream_memcpy(k_cache, v_cache, indices)
        results["multistream_memcpy"].append(t)

    print("[4/4] Prefetch+Gather...")
    for _ in range(NUM_ITERATIONS):
        _, _, t = method_prefetch_gather(k_cache, v_cache, indices)
        results["prefetch_gather"].append(t)

    # 结果
    print("\n" + "=" * 70)
    print("结果")
    print("=" * 70)

    data_size_mb = 2 * BATCH_SIZE * NUM_HEADS * NUM_SELECTED * HEAD_DIM * 2 / 1024**2

    for name, times in results.items():
        arr = np.array(times)
        bw = data_size_mb / (arr.mean() / 1000)
        print(f"\n{name}:")
        print(f"  平均: {arr.mean():.3f} ms (std: {arr.std():.3f})")
        print(f"  带宽: {bw:.2f} MB/s")

    # 对比
    print("\n" + "=" * 70)
    print("对比 (相对Gather+Memcpy)")
    print("=" * 70)
    baseline = np.mean(results["gather_memcpy"])
    for name, times in results.items():
        ratio = np.mean(times) / baseline
        marker = "(baseline)" if name == "gather_memcpy" else ""
        if name == "multistream_memcpy":
            marker += " <- 类似UVA效果"
        print(f"  {name}: {ratio:.2f}x {marker}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    run_benchmark()
