"""
比较三种KV Cache传输方式的性能：
1. 分散Memcpy: 对每个零散KV单独发起DMA传输
2. Gather+Memcpy: CPU先gather到连续buffer，再单次传输
3. GPU UVA Kernel: GPU直接通过UVA访问零散的CPU pinned memory (CUDA C++)

编译CUDA扩展:
    python setup_cuda_uva.py build_ext --inplace

运行方式:
    python benchmark_kv_transfer.py

要求:
    - CUDA GPU
    - PyTorch with CUDA support
    - nvcc编译器
"""

import torch
import time
import numpy as np
from typing import Tuple, Optional
import argparse
import sys

# 配置参数
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_HEADS = 8
DEFAULT_SEQ_LEN = 8192
DEFAULT_HEAD_DIM = 128
DEFAULT_NUM_SELECTED = 2048
DEFAULT_NUM_ITERATIONS = 100
DEFAULT_WARMUP_ITERATIONS = 10

# 尝试导入CUDA UVA扩展
try:
    import cuda_uva
    CUDA_UVA_AVAILABLE = True
    print("CUDA UVA extension loaded successfully!")
except ImportError as e:
    CUDA_UVA_AVAILABLE = False
    print(f"Warning: CUDA UVA extension not available: {e}")
    print("Method 3 (True UVA) will be skipped. To enable, run: python setup_cuda_uva.py build_ext --inplace")


def create_pinned_kv_cache(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """创建CPU pinned memory中的KV Cache"""
    k_cache = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype).pin_memory()
    v_cache = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype).pin_memory()
    return k_cache, v_cache


def generate_scattered_indices(
    seq_len: int,
    num_selected: int,
    seed: int = 42,
) -> torch.Tensor:
    """生成零散的随机索引（确保分散性）"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    indices = []
    min_gap = seq_len // (num_selected * 2)

    attempts = 0
    while len(indices) < num_selected and attempts < num_selected * 10:
        idx = np.random.randint(0, seq_len)
        if all(abs(idx - existing) >= min_gap for existing in indices):
            indices.append(idx)
        attempts += 1

    while len(indices) < num_selected:
        idx = np.random.randint(0, seq_len)
        if idx not in indices:
            indices.append(idx)

    indices = torch.tensor(indices, dtype=torch.long)
    indices = indices[torch.randperm(len(indices))]
    return indices


# ============ 方法1: 分散Memcpy ============
def method_scattered_memcpy(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    device: str = "cuda:0",
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """对每个选中的token单独发起Memcpy"""
    batch_size, num_heads, _, head_dim = k_cache.shape
    num_selected = len(indices)

    k_gpu = torch.empty(batch_size, num_heads, num_selected, head_dim,
                        dtype=k_cache.dtype, device=device)
    v_gpu = torch.empty(batch_size, num_heads, num_selected, head_dim,
                        dtype=v_cache.dtype, device=device)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for i, idx in enumerate(indices):
        k_gpu[:, :, i, :].copy_(k_cache[:, :, idx, :], non_blocking=True)
        v_gpu[:, :, i, :].copy_(v_cache[:, :, idx, :], non_blocking=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return k_gpu, v_gpu, elapsed


# ============ 方法2: Gather+Memcpy ============
def method_gather_then_memcpy(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    device: str = "cuda:0",
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """先在CPU上gather到连续buffer，再单次传输到GPU"""
    batch_size, num_heads, _, head_dim = k_cache.shape
    num_selected = len(indices)

    k_cpu_gathered = torch.empty(batch_size, num_heads, num_selected, head_dim,
                                  dtype=k_cache.dtype, pin_memory=True)
    v_cpu_gathered = torch.empty(batch_size, num_heads, num_selected, head_dim,
                                  dtype=v_cache.dtype, pin_memory=True)

    torch.cuda.synchronize()
    start = time.perf_counter()

    torch.index_select(k_cache, dim=2, index=indices, out=k_cpu_gathered)
    torch.index_select(v_cache, dim=2, index=indices, out=v_cpu_gathered)

    k_gpu = k_cpu_gathered.to(device, non_blocking=True)
    v_gpu = v_cpu_gathered.to(device, non_blocking=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return k_gpu, v_gpu, elapsed


# ============ 方法3: 真正的CUDA UVA Kernel ============
def method_cuda_uva(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    device: str = "cuda:0",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], float]:
    """
    使用CUDA UVA扩展，GPU直接读取CPU pinned memory

    这是真正的零拷贝（zero-copy）方法，不需要CPU gather
    """
    if not CUDA_UVA_AVAILABLE:
        return None, None, 0.0

    batch_size, num_heads, seq_len, head_dim = k_cache.shape

    torch.cuda.synchronize()
    start = time.perf_counter()

    # 调用CUDA扩展 - GPU直接通过UVA读取CPU pinned memory
    k_gpu, v_gpu = cuda_uva.uva_gather(
        k_cache, v_cache, indices,
        batch_size, num_heads, seq_len, head_dim
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return k_gpu, v_gpu, elapsed


# ============ 基准测试主函数 ============
def run_benchmark(
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_heads: int = DEFAULT_NUM_HEADS,
    seq_len: int = DEFAULT_SEQ_LEN,
    head_dim: int = DEFAULT_HEAD_DIM,
    num_selected: int = DEFAULT_NUM_SELECTED,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    device: str = "cuda:0",
) -> dict:
    """运行完整的基准测试"""

    print("=" * 80)
    print("KV Cache传输方式基准测试")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Num Heads: {num_heads}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Head Dimension: {head_dim}")
    print(f"  Num Selected Tokens: {num_selected}")
    print(f"  Total Cache Size: {2 * batch_size * num_heads * seq_len * head_dim * 2 / 1024**3:.2f} GB (K+V, bf16)")
    print(f"  Selected Data Size: {2 * batch_size * num_heads * num_selected * head_dim * 2 / 1024**2:.2f} MB")
    print(f"  Device: {device}")
    print(f"  Iterations: {num_iterations} (warmup: {warmup_iterations})")
    print(f"  CUDA UVA Available: {CUDA_UVA_AVAILABLE}")
    print()

    # 创建巨大的KV Cache
    print("创建CPU pinned KV Cache...")
    k_cache, v_cache = create_pinned_kv_cache(batch_size, num_heads, seq_len, head_dim)
    print(f"  K Cache shape: {k_cache.shape}, pinned: {k_cache.is_pinned()}")
    print(f"  V Cache shape: {v_cache.shape}, pinned: {v_cache.is_pinned()}")

    # 生成零散索引
    print(f"\n生成零散索引 ({num_selected} tokens)...")
    indices = generate_scattered_indices(seq_len, num_selected)

    # 验证索引的分散性
    sorted_indices = torch.sort(indices).values
    gaps = sorted_indices[1:] - sorted_indices[:-1]
    print(f"  索引范围: [{indices.min().item()}, {indices.max().item()}]")
    print(f"  平均间隔: {gaps.float().mean().item():.1f}")
    print(f"  最小间隔: {gaps.min().item()}")
    print(f"  最大间隔: {gaps.max().item()}")

    results = {
        "scattered_memcpy": [],
        "gather_memcpy": [],
    }
    if CUDA_UVA_AVAILABLE:
        results["cuda_uva"] = []

    # 预热
    print(f"\n预热 ({warmup_iterations} iterations)...")
    for _ in range(warmup_iterations):
        _, _, _ = method_scattered_memcpy(k_cache, v_cache, indices, device)
        _, _, _ = method_gather_then_memcpy(k_cache, v_cache, indices, device)
        if CUDA_UVA_AVAILABLE:
            _, _, _ = method_cuda_uva(k_cache, v_cache, indices, device)

    # 正式测试
    print(f"\n正式测试 ({num_iterations} iterations)...")

    # 方法1: 分散Memcpy
    print("\n[1/3] 测试分散Memcpy...")
    for i in range(num_iterations):
        _, _, elapsed = method_scattered_memcpy(k_cache, v_cache, indices, device)
        results["scattered_memcpy"].append(elapsed * 1000)

    # 方法2: Gather+Memcpy
    print("[2/3] 测试Gather+Memcpy...")
    for i in range(num_iterations):
        _, _, elapsed = method_gather_then_memcpy(k_cache, v_cache, indices, device)
        results["gather_memcpy"].append(elapsed * 1000)

    # 方法3: CUDA UVA (如果可用)
    if CUDA_UVA_AVAILABLE:
        print("[3/3] 测试CUDA UVA Kernel...")
        for i in range(num_iterations):
            _, _, elapsed = method_cuda_uva(k_cache, v_cache, indices, device)
            results["cuda_uva"].append(elapsed * 1000)
    else:
        print("[3/3] 跳过CUDA UVA Kernel (未编译)")

    # 统计结果
    print("\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)

    for method_name, times in results.items():
        times_arr = np.array(times)
        data_size_mb = 2 * batch_size * num_heads * num_selected * head_dim * 2 / 1024**2
        bandwidth_mbs = data_size_mb / (times_arr.mean() / 1000)

        print(f"\n{method_name}:")
        print(f"  平均时间: {times_arr.mean():.3f} ms")
        print(f"  中位数:   {np.median(times_arr):.3f} ms")
        print(f"  标准差:   {times_arr.std():.3f} ms")
        print(f"  最小值:   {times_arr.min():.3f} ms")
        print(f"  最大值:   {times_arr.max():.3f} ms")
        print(f"  带宽:     {bandwidth_mbs:.2f} MB/s")

    # 比较
    print("\n" + "=" * 80)
    print("性能比较 (相对于Gather+Memcpy)")
    print("=" * 80)
    baseline = np.mean(results["gather_memcpy"])
    for method_name, times in results.items():
        ratio = np.mean(times) / baseline
        marker = "(baseline)" if method_name == "gather_memcpy" else ""
        if method_name == "cuda_uva" and ratio < 1.0:
            marker += " <-- WINNER!"
        print(f"  {method_name}: {ratio:.2f}x {marker}")

    # 额外分析
    if "cuda_uva" in results:
        print("\n" + "=" * 80)
        print("UVA优势分析")
        print("=" * 80)
        gather_time = np.mean(results["gather_memcpy"])
        uva_time = np.mean(results["cuda_uva"])
        speedup = gather_time / uva_time
        print(f"  Gather+Memcpy 时间: {gather_time:.3f} ms")
        print(f"  CUDA UVA 时间:      {uva_time:.3f} ms")
        print(f"  加速比:             {speedup:.2f}x")

        if speedup > 1.0:
            print("\n  UVA性能优势:")
            print("  - 零拷贝：无需CPU gather操作")
            print("  - 并行性：GPU多线程并行访问分散内存")
            print("  - 无临时buffer：节省CPU内存带宽")
        else:
            print("\n  注意：UVA在此场景下可能没有优势，原因可能包括：")
            print("  - 索引过于分散，导致缓存不友好")
            print("  - PCIe带宽成为瓶颈")
            print("  - 小数据传输的开销")

    return results


def main():
    parser = argparse.ArgumentParser(description="KV Cache传输方式基准测试")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--num-selected", type=int, default=DEFAULT_NUM_SELECTED)
    parser.add_argument("--iterations", type=int, default=DEFAULT_NUM_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_ITERATIONS)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA not available!")
        return

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(args.device)}")

    run_benchmark(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        num_selected=args.num_selected,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        device=args.device,
    )


if __name__ == "__main__":
    main()
