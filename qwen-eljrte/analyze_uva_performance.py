"""
分析 CUDA UVA 性能优势

运行:
    python analyze_uva_performance.py
"""

import torch
import numpy as np
import time

# 测试不同参数下的 UVA vs Gather 性能
TEST_CONFIGS = [
    # (seq_len, num_selected, description)
    (4096, 512, "小序列，少token"),
    (4096, 1024, "小序列，多token"),
    (8192, 1024, "中序列，少token"),
    (8192, 2048, "中序列，多token"),
    (16384, 2048, "大序列，少token"),
    (16384, 4096, "大序列，多token"),
    (32768, 4096, "超大序列"),
]

# 测试不同分散度
def generate_indices_scattered(seq_len, num_selected, scatter_factor=1.0):
    """
    scatter_factor: 分散度
    - 0.0: 完全连续（最集中）
    - 1.0: 完全分散（最分散）
    """
    if scatter_factor == 0.0:
        # 连续索引
        start = np.random.randint(0, seq_len - num_selected)
        return torch.arange(start, start + num_selected, dtype=torch.long)
    else:
        # 分散索引
        chunk_size = int(seq_len / num_selected * (1.0 - scatter_factor * 0.5))
        indices = []
        for i in range(num_selected):
            start = min(i * chunk_size, seq_len - 1)
            end = min((i + 1) * chunk_size, seq_len)
            if end > start:
                idx = np.random.randint(start, end)
            else:
                idx = np.random.randint(0, seq_len)
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long)


def benchmark_transfer_methods(seq_len, num_selected, scatter_factor=0.5):
    """对比 Gather vs UVA"""
    BATCH = 1
    HEADS = 28
    DIM = 128
    ITERATIONS = 20
    WARMUP = 5

    device = "cuda:0"

    # 创建数据
    k_cache = torch.randn(BATCH, HEADS, seq_len, DIM, dtype=torch.bfloat16).pin_memory()
    v_cache = torch.randn(BATCH, HEADS, seq_len, DIM, dtype=torch.bfloat16).pin_memory()
    indices = generate_indices_scattered(seq_len, num_selected, scatter_factor)

    try:
        import cuda_uva
        has_uva = True
    except ImportError:
        has_uva = False
        return None

    # 预热
    for _ in range(WARMUP):
        # Gather方法
        k_g = torch.index_select(k_cache, 2, indices)
        v_g = torch.index_select(v_cache, 2, indices)
        k_gpu = k_g.to(device)
        v_gpu = v_g.to(device)

        # UVA方法
        k_u, v_u = cuda_uva.uva_gather(
            k_cache, v_cache, indices,
            BATCH, HEADS, seq_len, DIM
        )

    torch.cuda.synchronize()

    # 测试 Gather
    gather_times = []
    for _ in range(ITERATIONS):
        torch.cuda.synchronize()
        start = time.perf_counter()

        k_cpu = torch.index_select(k_cache, 2, indices)
        v_cpu = torch.index_select(v_cache, 2, indices)
        k_gpu = k_cpu.to(device)
        v_gpu = v_cpu.to(device)

        torch.cuda.synchronize()
        gather_times.append((time.perf_counter() - start) * 1000)

    # 测试 UVA
    uva_times = []
    for _ in range(ITERATIONS):
        torch.cuda.synchronize()
        start = time.perf_counter()

        k_gpu, v_gpu = cuda_uva.uva_gather(
            k_cache, v_cache, indices,
            BATCH, HEADS, seq_len, DIM
        )

        torch.cuda.synchronize()
        uva_times.append((time.perf_counter() - start) * 1000)

    gather_mean = np.mean(gather_times)
    uva_mean = np.mean(uva_times)
    speedup = gather_mean / uva_mean

    return {
        "seq_len": seq_len,
        "num_selected": num_selected,
        "scatter_factor": scatter_factor,
        "gather_ms": gather_mean,
        "uva_ms": uva_mean,
        "speedup": speedup,
    }


def analyze_scatter_impact():
    """分析分散度对性能的影响"""
    print("=" * 70)
    print("分散度对 UVA 性能的影响")
    print("=" * 70)
    print("\n测试配置: seq_len=8192, num_selected=2048")
    print()

    scatter_factors = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    for sf in scatter_factors:
        result = benchmark_transfer_methods(8192, 2048, sf)
        if result:
            results.append(result)
            print(f"分散度 {sf:.2f}: Gather={result['gather_ms']:.2f}ms, "
                  f"UVA={result['uva_ms']:.2f}ms, 加速比={result['speedup']:.2f}x")


def analyze_scale_impact():
    """分析不同规模下的性能"""
    print("\n" + "=" * 70)
    print("不同规模下的 UVA 性能")
    print("=" * 70)
    print()

    results = []
    for seq_len, num_selected, desc in TEST_CONFIGS:
        result = benchmark_transfer_methods(seq_len, num_selected, 0.5)
        if result:
            results.append(result)
            data_size = 2 * 1 * 28 * num_selected * 128 * 2 / 1024**2
            print(f"{desc:20s}: Gather={result['gather_ms']:.2f}ms, "
                  f"UVA={result['uva_ms']:.2f}ms, 加速比={result['speedup']:.2f}x, "
                  f"数据量={data_size:.1f}MB")


def main():
    print("CUDA UVA 性能分析")
    print("=" * 70)
    print()

    try:
        import cuda_uva
        print("✓ CUDA UVA 扩展已加载")
    except ImportError:
        print("✗ CUDA UVA 扩展未找到，请先编译:")
        print("  python setup_cuda_uva_v2.py build_ext --inplace")
        return

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 运行分析
    analyze_scatter_impact()
    analyze_scale_impact()

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
UVA 性能优势来源:
1. 零拷贝: 无需CPU参与gather操作
2. GPU并行: 多线程同时访问分散内存
3. 减少中间buffer: 节省CPU内存带宽

适用场景:
- 选中token分散在序列各处
- 需要减少CPU负载
- 大规模KV cache场景

注意事项:
- 需要pinned memory支持
- 索引极度分散时可能受PCIe延迟影响
- 小数据量时优势不明显
    """)


if __name__ == "__main__":
    main()
