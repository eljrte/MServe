# CUDA UVA 优化指南

## 你的测试结果

**UVA 比 Gather+Transfer 更快** - 这是预期结果！

## 性能优势来源

| 因素 | Gather+Transfer | CUDA UVA | 优势 |
|------|-----------------|----------|------|
| CPU参与 | 需要CPU index_select | 零拷贝 | CPU减负 |
| 内存访问 | 先读再写（临时buffer） | 直接读取 | 节省带宽 |
| 并行性 | 串行gather | GPU多线程并行 | 更快 |
| PCIe传输 | 2次拷贝（scatter+gather） | 1次直接访问 | 更高效 |

## 什么时候UVA最快？

### 1. 分散度适中（推荐）
```python
# 选中token分布在整个序列
indices = [100, 500, 1200, 3000, ...]  # 分散但不过于稀疏
```

### 2. 数据量适中
- **太少** (< 1MB): PCIe启动开销占主导
- **适中** (10MB-100MB): UVA优势最明显
- **太大** (> 1GB): 可能受PCIe带宽限制

### 3. 批次大小
- UVA对小batch更友好（无需batch维度gather）

## 进一步优化建议

### 1. 异步UVA传输
```python
# 当前实现是同步的，可以改为异步
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    k_gpu, v_gpu = cuda_uva.uva_gather(...)
# 同时做其他计算
```

### 2. 批量UVA请求
```python
# 如果有多层需要gather，可以合并kernel启动
# 修改CUDA kernel支持多layer同时处理
```

### 3. 预读取索引
```python
# 把索引提前放到GPU
indices_gpu = indices.cuda()
# 修改kernel接受GPU indices（减少PCIe流量）
```

### 4. 结合Dynamic Sparse Attention
```python
# 在你的sparse attention中使用UVA
important_indices = compute_token_importance(query, key)
k_sparse, v_sparse = cuda_uva.uva_gather(
    k_cache_cpu, v_cache_cpu, important_indices,
    batch_size, num_heads, seq_len, head_dim
)
```

## 实际集成到KV Cache

修改 `kv_cache.py` 的 `get_sparse_async` 方法：

```python
def get_sparse_async_uva(self, layer_idx, device, token_indices):
    """使用CUDA UVA获取稀疏KV"""
    k_cpu, v_cpu, _, _ = self.get_snapshot(layer_idx)
    if k_cpu is None:
        return None

    import cuda_uva
    B, H, T, D = k_cpu.shape

    # 直接使用UVA gather（无需CPU index_select）
    k_gpu, v_gpu = cuda_uva.uva_gather(
        k_cpu, v_cpu, token_indices,
        B, H, T, D
    )

    # 创建完成事件
    done_event = torch.cuda.Event(enable_timing=False)
    done_event.record()

    return k_gpu, v_gpu, done_event
```

## 性能调优参数

### 测试不同配置
```bash
python analyze_uva_performance.py
```

### 找到最佳selected token数
```python
# 实验：改变num_selected看性能变化
for num in [512, 1024, 2048, 4096]:
    time = benchmark_uva(num_selected=num)
    print(f"{num}: {time}ms")
```

## 注意事项

1. **内存对齐**: 确保head_dim是16或32的倍数（GPU内存访问对齐）
2. **索引范围**: UVA访问越界会导致segfault（kernel里已检查）
3. **多GPU**: UVA默认只在单GPU+iGPU/CPU场景有效，多GPU需统一地址空间

## 下一步

1. ✅ 确认UVA在你的场景下确实更快（已完成）
2. 🔧 集成到kv_cache.py的get_sparse_async
3. 📊 测试Dynamic Sparse Attention端到端性能
4. 🚀 考虑批量UVA请求优化多层gather
