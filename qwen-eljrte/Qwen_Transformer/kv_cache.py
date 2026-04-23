import torch
from typing import Optional, Tuple,List
import threading
import queue
import os
import time


device = "cuda:0"
class CPUKVCache:
    def __init__(self, num_layers: int, max_len: int,queue_maxsize: int = 256):
        self.num_layers = num_layers
        self.max_len = max_len
        self.cache = [None] * num_layers
        self.current_seq_len = 0

        # === GPU buffer 池：每层预分配固定的 GPU buffer，复用避免重复分配 ===
        # 形状会在第一次 get_async 时根据实际 KV 形状确定
        self._gpu_k_buffer: List[Optional[torch.Tensor]] = [None] * num_layers
        self._gpu_v_buffer: List[Optional[torch.Tensor]] = [None] * num_layers
        # 记录每层的容量 (seq_len) 和 实际有效长度
        self._gpu_buffer_capacity: List[int] = [0] * num_layers
        self._gpu_buffer_devices: List[Optional[torch.device]] = [None] * num_layers
        self._gpu_buffer_free_events: List[Optional[torch.cuda.Event]] = [None] * num_layers

        self._h2d_stream = torch.cuda.Stream(device=device)
        self._d2h_stream = torch.cuda.Stream(device=device)
        
        # 锁
        self._locks = {i: threading.Lock() for i in range(num_layers)}

        self.last_append_event = None  # 记录最新一次写入的完成事件
        self.pending_k = {}
        self.pending_v = {}

        self._seq_len: List[int] = [0] * num_layers
        self._versions: List[int] = [0] * num_layers

        self.idxs = torch.randint(0, 8000, (150,))
        self.rand_k = torch.randn(1, 28, 150, 128, dtype=torch.bfloat16)
        self.rand_v = torch.randn(1, 28, 150, 128, dtype=torch.bfloat16)

        # === 线程复用：单消费者队列 + 常驻线程 ===
        self._q: "queue.Queue[Tuple[int, torch.Tensor, torch.Tensor]]" = queue.Queue(maxsize=queue_maxsize)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, name="kv-worker", daemon=True)
        self._worker.start()


        #下方用来优化infinigen
        self.random_len = 8000
        self.choose_k = 8000               # 防越界
        self.idx = torch.randperm(self.random_len, dtype=torch.long)[:self.choose_k]
        self.k_sel_cpu = torch.empty(1,28,1600,128, dtype=torch.bfloat16, pin_memory=True)
        self.v_sel_cpu = torch.empty(1,28,1600,128, dtype=torch.bfloat16, pin_memory=True)

        self.k_gpu = torch.empty_like(self.k_sel_cpu, device=device)
        self.v_gpu = torch.empty_like(self.v_sel_cpu, device=device)


        # self.idx = torch.randperm(300, dtype=torch.long)[:200]
    def get_seq_len(self):
        return self.current_seq_len

    def get(self, layer_idx: int, device: torch.device):
        """获取某层历史缓存（搬回 GPU）"""
        if self.cache[layer_idx] is None:
            return None
        k, v = self.cache[layer_idx]
        # ✅ 使用 non_blocking + pinned memory 加速搬运
        return k.to(device,non_blocking=True), v.to(device,non_blocking=True)
        # return k.to(device), v.to(device)
    
    def get_dsa(self,layer_idx: int, device: torch.device):
        """模拟DSA"""

        torch.cuda.synchronize()
        start = time.time()
        B = len(self.cache)
        if B == 0:
            return None
        # k_gpu_list: List[torch.Tensor] = []
        # v_gpu_list: List[torch.Tensor] = []

        kv = self.cache[layer_idx]
    
        k_cpu, v_cpu = kv
        # kg = k_cpu.to(device, non_blocking=True)
        # vg = v_cpu.to(device, non_blocking=True)
        idx = torch.randperm(self.random_len, dtype=torch.long)[:self.choose_k]
        tmp_k_sel_cpu = torch.empty(1,28,8000,128, dtype=torch.bfloat16, pin_memory=True)
        tmp_v_sel_cpu = torch.empty(1,28,8000,128, dtype=torch.bfloat16, pin_memory=True)
        torch.index_select(k_cpu, dim=2, index=idx, out=tmp_k_sel_cpu)
        torch.index_select(v_cpu, dim=2, index=idx, out=tmp_v_sel_cpu)
        kg = tmp_k_sel_cpu.to(device, non_blocking=True)
        vg = tmp_v_sel_cpu.to(device, non_blocking=True)

        torch.cuda.synchronize()
        end = time.time()
        print(f"reload:{end-start}")
        return kg,vg


    # @torch.no_grad()
    # def get_async(
    #     self,
    #     layer_idx: int,
    #     device: torch.device,
    #     stack_on_gpu: bool = True,              # 若 True 且提供 compute_stream，则在 GPU 侧堆叠
    #     num_select: int = 150,                  # 每个样本随机挑选的 token 数上限
    #     return_valid_idx: bool = False,
    #     compute_stream: Optional[torch.cuda.Stream] = None,
    # ):
    #     """
    #     从 self.cache[b][layer_idx] 的 KV（CPU pinned）中：
    #     - 对每个有效样本，独立随机挑选 <= num_select 个位置
    #     - 将每个样本挑选结果分别拷入该样本专属的 pinned 小块
    #     - 分别在 _h2d_stream 上异步 H2D 到 GPU（不打包成一个大 batch 一起传）

    #     兼容 KV 形状：
    #     - 3D: [H, T, D]（序列维 dim=1）
    #     - 4D: [H, Q, T, D]（序列维 dim=2）
    #     要求同层不同样本的 H/Q/D 一致（T 可不同）。
    #     返回：
    #     - k_list, v_list, evt_list （均为按有效样本顺序的列表）
    #     - 若 return_valid_idx=True，额外返回 valid_idx（对应原始 batch 索引）
    #     - 若 stack_on_gpu=True 且传入 compute_stream，则再额外返回 (k_batched, v_batched)
    #     """
    #     B = len(self.cache)
    #     if B == 0:
    #         return None

    #     # 参考形状/类型锚定（除 T 外一致）
    #     ref_dim = None
    #     ref_dtype = None
    #     ref_shape_except_T = None  # 3D: (H, D)，4D: (H, Q, D)

    #     valid_entries = []  # list of (b, k_cpu, v_cpu, seq_dim, T_len)
    #     for b in range(B):
    #         kv = self.cache[b][layer_idx]
    #         if kv is None:
    #             continue
    #         k_cpu, v_cpu = kv
    #         assert k_cpu.is_pinned() and v_cpu.is_pinned(), f"KV at batch={b} must be pinned"

    #         seq_dim = 1
    #         shape_except_T = (k_cpu.size(0), k_cpu.size(2))          # (H, D)


    #         if ref_dim is None:
    #             ref_dim = k_cpu.dim()
    #             ref_dtype = k_cpu.dtype
    #             ref_shape_except_T = shape_except_T
    #         else:
    #             assert k_cpu.dim() == ref_dim, f"dim mismatch at batch={b}: got {k_cpu.dim()}, expect {ref_dim}"
    #             assert k_cpu.dtype == ref_dtype, f"dtype mismatch at batch={b}: got {k_cpu.dtype}, expect {ref_dtype}"
    #             assert shape_except_T == ref_shape_except_T, \
    #                 f"shape mismatch (except T) at batch={b}: got {shape_except_T}, expect {ref_shape_except_T}"

    #         T_len = k_cpu.size(seq_dim)
    #         if T_len <= 0:
    #             continue

    #         valid_entries.append((b, k_cpu, v_cpu, seq_dim, T_len))

    #     if len(valid_entries) == 0:
    #         return None

    #     # 逐样本独立挑选、独立 H2D
    #     k_list = []          # 每个元素是该样本的 GPU 张量
    #     v_list = []
    #     evt_list = []
    #     valid_idx = []       # 记录对应的原始 batch 下标

    #     for (b, k_cpu, v_cpu, seq_dim, T_len) in valid_entries:
    #         pick_len = min(num_select, T_len)

    #         # 随机不重复索引（CPU 上）
    #         # 注意：k_cpu/v_cpu 在 CPU 上（pinned），randperm 也在 CPU 上生成
    #         rand_idx = torch.randperm(T_len, device='cpu')[:pick_len]

    #         # 为该样本单独分配 pinned 小块，并从原始 pinned KV 中拣出

    #         H, D = ref_shape_except_T
    #         k_cpu_sel = torch.empty((H, pick_len, D), dtype=ref_dtype, pin_memory=True)
    #         v_cpu_sel = torch.empty((H, pick_len, D), dtype=ref_dtype, pin_memory=True)

    #         torch.index_select(k_cpu, dim=1, index=rand_idx, out=k_cpu_sel)
    #         torch.index_select(v_cpu, dim=1, index=rand_idx, out=v_cpu_sel)


    #         # 该样本单独一次异步 H2D
    #         with torch.cuda.stream(self._h2d_stream):
    #             k_gpu = k_cpu_sel.to(device, non_blocking=True)
    #             v_gpu = v_cpu_sel.to(device, non_blocking=True)
    #             k_gpu.record_stream(self._h2d_stream)
    #             v_gpu.record_stream(self._h2d_stream)

    #             evt = torch.cuda.Event(enable_timing=False)
    #             evt.record(self._h2d_stream)

    #         k_list.append(k_gpu)
    #         v_list.append(v_gpu)
    #         evt_list.append(evt)
    #         valid_idx.append(b)

    #     # 如需在 GPU 侧堆叠（不产生新的 H2D），就在 compute_stream 上等齐所有样本的 H2D 完成后再 stack
    #     # 注意：只在提供 compute_stream 时执行，以避免阻塞当前默认流
    #     stacked = None
    #     if stack_on_gpu and compute_stream is not None and len(k_list) > 0:
    #         compute_stream.wait_event(evt_list[0])  # 先等第一个，下面统一等全部
    #         for e in evt_list[1:]:
    #             compute_stream.wait_event(e)
    #         # 在 compute_stream 上做一次轻量的 stack（纯 GPU 操作，无拷贝到主机）
    #         with torch.cuda.stream(compute_stream):
    #             k_batched = torch.stack(k_list, dim=0)
    #             v_batched = torch.stack(v_list, dim=0)
    #         stacked = (k_batched, v_batched)

    #     # 组织返回
    #     if return_valid_idx:
    #         if stacked is not None:
    #             k_batched, v_batched = stacked
    #             return (k_list, v_list, evt_list, valid_idx, k_batched, v_batched)
    #         else:
    #             return (k_list, v_list, evt_list, valid_idx)
    #     else:
    #         if stacked is not None:
    #             k_batched, v_batched = stacked
    #             return (k_list, v_list, evt_list, k_batched, v_batched)
    #         else:
    #             return (k_list, v_list, evt_list)







    
    def get_async(self, layer_idx: int, device: torch.device):
        """
        在专用 H2D stream 上，把该层的 CPU pinned KV 异步拷到可复用的 GPU buffer。
        返回 (k_view, v_view, done_event)。
        """
        k_cpu, v_cpu, seq_len, ver = self.get_snapshot(layer_idx)
        if k_cpu is None:
            return None

        assert k_cpu.is_pinned() and v_cpu.is_pinned(), "KV must be pinned for async H2D"
        assert k_cpu.ndim == 4 and v_cpu.ndim == 4, "Expect [B, H, T, D]"

        dev = torch.device(device)
        B, H, T_cpu, D = k_cpu.shape

        need_alloc = (
            self._gpu_k_buffer[layer_idx] is None
            or self._gpu_v_buffer[layer_idx] is None
            or self._gpu_buffer_devices[layer_idx] != dev
            or self._gpu_k_buffer[layer_idx].dtype != k_cpu.dtype
            or self._gpu_v_buffer[layer_idx].dtype != v_cpu.dtype
            or self._gpu_k_buffer[layer_idx].shape[0] != B
            or self._gpu_k_buffer[layer_idx].shape[1] != H
            or self._gpu_k_buffer[layer_idx].shape[3] != D
            or self._gpu_buffer_capacity[layer_idx] < T_cpu
        )

        if need_alloc:
            alloc_len = max(self.max_len, T_cpu)
            self._gpu_k_buffer[layer_idx] = torch.empty((B, H, alloc_len, D), dtype=k_cpu.dtype, device=dev)
            self._gpu_v_buffer[layer_idx] = torch.empty((B, H, alloc_len, D), dtype=v_cpu.dtype, device=dev)
            self._gpu_buffer_capacity[layer_idx] = alloc_len
            self._gpu_buffer_devices[layer_idx] = dev
            self._gpu_buffer_free_events[layer_idx] = None

        k_gpu_buf = self._gpu_k_buffer[layer_idx]
        v_gpu_buf = self._gpu_v_buffer[layer_idx]
        k_gpu_view = k_gpu_buf[:, :, :T_cpu, :]
        v_gpu_view = v_gpu_buf[:, :, :T_cpu, :]

        # print(T_cpu)

        with torch.cuda.stream(self._h2d_stream):
            free_event = self._gpu_buffer_free_events[layer_idx]
            if free_event is not None:
                self._h2d_stream.wait_event(free_event)

            k_gpu_view.copy_(k_cpu, non_blocking=True)
            v_gpu_view.copy_(v_cpu, non_blocking=True)

            done_event = torch.cuda.Event(enable_timing=False)
            done_event.record(self._h2d_stream)

        return k_gpu_view, v_gpu_view, done_event


    # def get_async(self, layer_idx: int, device: torch.device):
    #     if self.cache[layer_idx] is None:
    #         return None
        
        
    #     k, v = self.cache[layer_idx]
    #     # print(k.shape, v.shape)

    #     # 按索引写入
    #     # torch.cuda.synchronize()
    #     # copy_start = time.time()
    #     # k.index_copy_(2, self.idxs, self.rand_k)
    #     # v.index_copy_(2, self.idxs, self.rand_v)
    #     # self.k_sel_cpu.resize_(0)
    #     # self.v_sel_cpu.resize_(0)

    #     # torch.index_select(k, 2, self.idx, out=self.k_sel_cpu)
    #     # torch.index_select(v, 2, self.idx, out=self.v_sel_cpu)
    #     # torch.cuda.synchronize()
    #     # copy_end = time.time()
    #     # with open("copy_time.txt", "a") as f:
    #     #     f.write(f"{copy_end - copy_start}\n")
    #     # k[:, :, :1, :] = torch.ones_like(k[:, :, :1, :])
    #     # v[:, :, :1, :] = torch.ones_like(v[:, :, :1, :])


    #     # 确保是 pinned mem
    #     assert k.is_pinned() and v.is_pinned(), "KV must be pinned for async H2D"

    #     with torch.cuda.stream(self._h2d_stream):
    #         # k_gpu = self.k_sel_cpu.to(device, non_blocking=True)
    #         # v_gpu = self.v_sel_cpu.to(device, non_blocking=True)
    #         k_gpu = k.to(device, non_blocking=True)
    #         v_gpu = v.to(device, non_blocking=True)
    #         k_gpu.record_stream(self._h2d_stream)
    #         v_gpu.record_stream(self._h2d_stream)

    #     return k_gpu, v_gpu


    # # InfiniGen的
    # def get_async(self, layer_idx: int, device: torch.device, num_select: int = 100):
    #     if self.cache[layer_idx] is None:
    #         return None
    #     k_cpu, v_cpu = self.cache[layer_idx]

    #     # 必须是 pinned host tensor 才能异步 H2D
    #     assert k_cpu.is_pinned() and v_cpu.is_pinned(), "KV must be pinned for async H2D"
    #     assert k_cpu.ndim == 4 and v_cpu.ndim == 4, "Expect [B, num_kv_heads, T, head_dim]"

    #     B, H, T, D = k_cpu.shape
    #     # if num_select >= T:
    #     #     idx = torch.arange(T, dtype=torch.long)  # 全量
    #     # else:
    #     #     # 生成升序随机索引（dim=2）
    #     #     idx = torch.randperm(T, dtype=torch.long)[:num_select]

    #     # ---- 在 CPU 侧（pinned）聚合出 [B, H, num_select, D] ----
    #     # 预分配 pinned staging（避免 index_select 产生非 pinned 临时张量）
        
    #     torch.cuda.synchronize()
    #     copy_start = time.time()
    #     # k_sel_cpu = k_cpu[:,:,self.idx,:]
    #     # v_sel_cpu = v_cpu[:,:,self.idx,:]
    #     # print(k_cpu.shape)
    #     k_sel_cpu_2 = torch.empty(1, 28, 400, 128, dtype=torch.bfloat16, pin_memory=True)
    #     v_sel_cpu_2 = torch.empty(1, 28, 400, 128, dtype=torch.bfloat16, pin_memory=True)
    #     torch.index_select(k_cpu, dim=2, index=self.idx, out=k_sel_cpu_2)
    #     torch.index_select(v_cpu, dim=2, index=self.idx, out=v_sel_cpu_2)
    #     torch.cuda.synchronize()
    #     copy_end = time.time()
    #     with open("copy_time.txt", "a") as f:
    #         f.write(f"{copy_end - copy_start}\n")
    #     # print("copy time: ", copy_end - copy_start)

    #     # ---- 异步 H2D：复用专用 H2D 流；用 copy_ 避免 GPU 端重复分配 ----
    #     with torch.cuda.stream(self._h2d_stream):
    #         torch.cuda.synchronize()
    #         transfer_start = time.time()
    #         self.k_gpu.copy_(self.k_sel_cpu, non_blocking=True)
    #         self.v_gpu.copy_(self.v_sel_cpu, non_blocking=True)
    #         torch.cuda.synchronize()
    #         transfer_end = time.time()
    #         with open("transfer_time.txt", "a") as f:
    #             f.write(f"{transfer_end - transfer_start}\n")
    #         # self.gpu = self.k_sel_cpu.to(device, non_blocking=True)
    #         # self.v_gpu = self.v_sel_cpu.to(device, non_blocking=True)
    #         self.k_gpu.record_stream(self._h2d_stream)
    #         self.v_gpu.record_stream(self._h2d_stream)

    # #     # 返回选择后的 KV 以及所用的时间步索引，便于对齐
    #     return self.k_gpu, self.v_gpu



    # def _get_h2d_stream(self, device: torch.device):
    #     """按目标 device 取/建专用 H2D stream，避免设备不匹配。"""
    #     if not hasattr(self, "_h2d_streams"):
    #         self._h2d_streams = {}
    #     dev_idx = torch.device(device).index or 0
    #     if dev_idx not in self._h2d_streams:
    #         self._h2d_streams[dev_idx] = torch.cuda.Stream(device=device)
    #     return self._h2d_streams[dev_idx]

    # def _ensure_sel_buffers(self, layer_idx:int, B:int, H:int, D:int, need_T:int,
    #                         device: torch.device, dtype_k: torch.dtype, dtype_v: torch.dtype):
    #     """按需扩容并复用：CPU pinned staging 与 GPU 接收缓冲。"""
    #     if not hasattr(self, "_sel_pool"):
    #         self._sel_pool = {}
    #     if layer_idx not in self._sel_pool:
    #         self._sel_pool[layer_idx] = {"k_cpu": None, "v_cpu": None,
    #                                     "k_gpu": None, "v_gpu": None, "cap_T": 0, "device": None}
    #     p = self._sel_pool[layer_idx]
    #     grow_shape = (p["k_cpu"] is None or p["cap_T"] < need_T or
    #                 p["k_cpu"].shape[0] != B or p["k_cpu"].shape[1] != H or p["k_cpu"].shape[-1] != D or
    #                 p["k_cpu"].dtype != dtype_k or p["v_cpu"].dtype != dtype_v)
    #     grow_dev = (p["k_gpu"] is None or p["device"] != torch.device(device) or
    #                 p["k_gpu"].shape[0] != B or p["k_gpu"].shape[1] != H or p["k_gpu"].shape[-1] != D or
    #                 p["k_gpu"].dtype != dtype_k or p["v_gpu"].dtype != dtype_v)
    #     if grow_shape:
    #         cap_T = max(need_T, max(64, p["cap_T"] * 2))
    #         p["k_cpu"] = torch.empty((B, H, cap_T, D), dtype=dtype_k, pin_memory=True)
    #         p["v_cpu"] = torch.empty((B, H, cap_T, D), dtype=dtype_v, pin_memory=True)
    #         p["cap_T"] = cap_T
    #     if grow_dev:
    #         if p["device"] != torch.device(device):
    #             # 设备改变：重建 GPU 缓冲（避免跨设备使用）
    #             p["k_gpu"] = None; p["v_gpu"] = None
    #         cap_T = max(need_T, p["cap_T"])
    #         p["k_gpu"] = torch.empty((B, H, cap_T, D), dtype=dtype_k, device=device)
    #         p["v_gpu"] = torch.empty((B, H, cap_T, D), dtype=dtype_v, device=device)
    #         p["device"] = torch.device(device)
    #     return (p["k_cpu"], p["v_cpu"], p["k_gpu"], p["v_gpu"], p["cap_T"])

    # @torch.no_grad()
    # def get_async(self, layer_idx: int, device: torch.device, num_select: int = 50):
    #     """
    #     从 CPU pinned KV 中 **无放回均匀随机** 选 num_select 个时间步，异步 H2D 到 GPU。
    #     返回: (k_gpu_view, v_gpu_view, idx, done_event)
    #     """
    #     # 建议：若已实现 get_snapshot()，用它拿一致版本
    #     # k_cpu, v_cpu, _, _ = self.get_snapshot(layer_idx)
    #     if self.cache[layer_idx] is None:
    #         return None
    #     k_cpu, v_cpu = self.cache[layer_idx]

    #     # 基础校验
    #     assert k_cpu.is_pinned() and v_cpu.is_pinned(), "KV must be pinned for async H2D"
    #     assert k_cpu.ndim == 4 and v_cpu.ndim == 4, "Expect [B, H, T, D]"
    #     B, H, T, D = k_cpu.shape

    #     # —— “非常随机”：无放回均匀采样 —— #
    #     # 使用独立 CPU 生成器，以高熵随机种子避免可预测序列
    #     g = torch.Generator(device='cpu')
    #     seed = int.from_bytes(os.urandom(8), 'little', signed=False) % (2**63 - 1)
    #     g.manual_seed(seed)
    #     k = min(num_select, T)
    #     idx = torch.randperm(T, generator=g)[:k]  # 不排序，保持随机次序

    #     # 预分配/复用 staging 与 GPU 缓冲
    #     k_sel_cpu_buf, v_sel_cpu_buf, k_gpu_buf, v_gpu_buf, cap_T = self._ensure_sel_buffers(
    #         layer_idx, B, H, D, k, device, k_cpu.dtype, v_cpu.dtype
    #     )
    #     # 仅使用前 k 槽位（视图）
    #     k_sel_cpu = k_sel_cpu_buf[:, :, :k, :]
    #     v_sel_cpu = v_sel_cpu_buf[:, :, :k, :]
    #     k_gpu = k_gpu_buf[:, :, :k, :]
    #     v_gpu = v_gpu_buf[:, :, :k, :]

    #     # CPU 侧（pinned）聚合：确保 out= 指向 pinned 缓冲，避免中间非 pinned 张量
    #     torch.index_select(k_cpu, dim=2, index=idx, out=k_sel_cpu)
    #     torch.index_select(v_cpu, dim=2, index=idx, out=v_sel_cpu)

    #     # 异步 H2D：使用与目标 device 对应的专用 stream
    #     h2d_stream = self._get_h2d_stream(device)
    #     with torch.cuda.stream(h2d_stream):
    #         k_gpu.copy_(k_sel_cpu, non_blocking=True)
    #         v_gpu.copy_(v_sel_cpu, non_blocking=True)
    #         k_gpu.record_stream(h2d_stream)
    #         v_gpu.record_stream(h2d_stream)
    #         done = torch.cuda.Event()
    #         done.record(h2d_stream)

    #     return k_gpu, v_gpu

    def mark_layer_buffer_consumed(self, layer_idx: int, stream: torch.cuda.Stream):
        """
        在 compute stream 上记录一个事件，表示该层 GPU buffer 已经被消费完成。
        下一次复用同层 buffer 前，H2D stream 会先等待这个事件，避免覆盖仍在使用的数据。
        """
        if self._gpu_k_buffer[layer_idx] is None or self._gpu_v_buffer[layer_idx] is None:
            return
        evt = torch.cuda.Event(enable_timing=False)
        evt.record(stream)
        self._gpu_buffer_free_events[layer_idx] = evt

    def get_sparse_async_uva(
        self,
        layer_idx: int,
        device: torch.device,
        token_indices: torch.Tensor,
    ):
        """
        使用CUDA UVA直接gather分散的KV到GPU（零拷贝，无需CPU index_select）

        要求：
            - 已编译并导入cuda_uva扩展
            - KV cache必须是pinned memory
            - token_indices在CPU上

        返回：
            (k_selected, v_selected, done_event) 或 None
        """
        # 检查CUDA UVA是否可用
        try:
            import cuda_uva
        except ImportError:
            # 回退到普通get_sparse_async
            return self.get_sparse_async(layer_idx, device, token_indices)

        k_cpu, v_cpu, _seq_len, _ver = self.get_snapshot(layer_idx)
        if k_cpu is None:
            return None

        # 验证条件
        if not k_cpu.is_pinned() or not v_cpu.is_pinned():
            # 非pinned memory，回退到普通方法
            return self.get_sparse_async(layer_idx, device, token_indices)

        assert k_cpu.ndim == 4 and v_cpu.ndim == 4, "Expect [B, H, T, D]"
        assert token_indices.device.type == "cpu", "token_indices must be on CPU"

        B, H, T, D = k_cpu.shape

        torch.cuda.synchronize()

        # 直接使用CUDA UVA gather（GPU直接读取CPU pinned memory）
        k_gpu, v_gpu = cuda_uva.uva_gather(
            k_cpu, v_cpu, token_indices,
            B, H, T, D
        )

        # 创建完成事件
        done_event = torch.cuda.Event(enable_timing=False)
        done_event.record()

        return k_gpu, v_gpu, done_event

    def get_sparse_async(
        self,
        layer_idx: int,
        device: torch.device,
        token_indices: torch.Tensor,
    ):
        """
        根据 token 重要性索引，异步加载选定的 KV cache 到 GPU。

        Args:
            layer_idx: 层索引
            device: 目标 GPU 设备
            token_indices: 重要 token 的索引 [num_selected]，必须在 CPU 上

        Returns:
            (k_selected, v_selected, done_event) 或 None
            - k_selected/v_selected: [B, H, num_selected, D] GPU 张量
        """
        k_cpu, v_cpu, _seq_len, _ver = self.get_snapshot(layer_idx)
        if k_cpu is None:
            return None

        assert k_cpu.is_pinned() and v_cpu.is_pinned(), "KV must be pinned for async H2D"
        assert k_cpu.ndim == 4 and v_cpu.ndim == 4, "Expect [B, H, T, D]"
        assert token_indices.device.type == "cpu", "token_indices must be on CPU"

        dev = torch.device(device)
        B, H, _, D = k_cpu.shape
        num_selected = token_indices.shape[0]

        # 预分配/复用 pinned CPU 缓冲用于 index_select
        if not hasattr(self, "_sparse_cpu_buffers"):
            self._sparse_cpu_buffers = {}

        buf_key = (B, H, D, k_cpu.dtype)
        if buf_key not in self._sparse_cpu_buffers:
            self._sparse_cpu_buffers[buf_key] = {
                "k_buf": torch.empty(B, H, num_selected, D, dtype=k_cpu.dtype, pin_memory=True),
                "v_buf": torch.empty(B, H, num_selected, D, dtype=v_cpu.dtype, pin_memory=True),
                "capacity": num_selected,
            }

        cpu_bufs = self._sparse_cpu_buffers[buf_key]
        if cpu_bufs["capacity"] < num_selected:
            # 扩容
            cpu_bufs["k_buf"] = torch.empty(B, H, num_selected, D, dtype=k_cpu.dtype, pin_memory=True)
            cpu_bufs["v_buf"] = torch.empty(B, H, num_selected, D, dtype=v_cpu.dtype, pin_memory=True)
            cpu_bufs["capacity"] = num_selected

        k_sel_cpu = cpu_bufs["k_buf"][:, :, :num_selected, :]
        v_sel_cpu = cpu_bufs["v_buf"][:, :, :num_selected, :]

        # CPU 侧 index_select（pinned -> pinned）
        torch.index_select(k_cpu, dim=2, index=token_indices, out=k_sel_cpu)
        torch.index_select(v_cpu, dim=2, index=token_indices, out=v_sel_cpu)

        # 预分配/复用 GPU 缓冲
        if not hasattr(self, "_sparse_gpu_buffers"):
            self._sparse_gpu_buffers = {}

        gpu_buf_key = (layer_idx, B, H, D, dev.index)
        if gpu_buf_key not in self._sparse_gpu_buffers:
            self._sparse_gpu_buffers[gpu_buf_key] = {
                "k_buf": torch.empty(B, H, num_selected, D, dtype=k_cpu.dtype, device=dev),
                "v_buf": torch.empty(B, H, num_selected, D, dtype=v_cpu.dtype, device=dev),
                "capacity": num_selected,
            }

        gpu_bufs = self._sparse_gpu_buffers[gpu_buf_key]
        if gpu_bufs["capacity"] < num_selected:
            gpu_bufs["k_buf"] = torch.empty(B, H, num_selected, D, dtype=k_cpu.dtype, device=dev)
            gpu_bufs["v_buf"] = torch.empty(B, H, num_selected, D, dtype=v_cpu.dtype, device=dev)
            gpu_bufs["capacity"] = num_selected

        k_sel_gpu = gpu_bufs["k_buf"][:, :, :num_selected, :]
        v_sel_gpu = gpu_bufs["v_buf"][:, :, :num_selected, :]

        # 异步 H2D
        with torch.cuda.stream(self._h2d_stream):
            k_sel_gpu.copy_(k_sel_cpu, non_blocking=True)
            v_sel_gpu.copy_(v_sel_cpu, non_blocking=True)
            done_event = torch.cuda.Event(enable_timing=False)
            done_event.record(self._h2d_stream)

        return k_sel_gpu, v_sel_gpu, done_event

    @staticmethod
    def get_full_kv(
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
        key_new: torch.Tensor,
        value_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if past_key_value is None:
            return key_new, value_new
        k_old, v_old = past_key_value
        return (
            torch.cat([k_old, key_new], dim=2),
            torch.cat([v_old, value_new], dim=2),
        )

    # @torch.no_grad()
    # def append(self, layer_idx: int, key_new: torch.Tensor, value_new: torch.Tensor):
    #     """
    #     在 attention 计算结束后，把新生成的 KV 添加到 CPU 缓存中（使用 pinned memory）。
    #     """
    #     # ✅ 搬到 CPU 并设为 pinned
    #     k_new = key_new.detach().cpu().pin_memory()
    #     v_new = value_new.detach().cpu().pin_memory()

    #     # k_new = key_new.detach().cpu()
    #     # v_new = value_new.detach().cpu()

    #     if self.cache[layer_idx] is None:
    #         # k_total, v_total = k_new[:,:,:100,:], v_new[:,:,:100,:]
    #         k_total, v_total = k_new, v_new
    #     else:
    #         k_old, v_old = self.cache[layer_idx]
    #         k_total = torch.cat([k_old, k_new], dim=2)
    #         v_total = torch.cat([v_old, v_new], dim=2)

    #     # ✅ 截断（仅保留最近 max_len 个）
    #     if k_total.size(2) > self.max_len:
    #         k_total = k_total[:, :, -self.max_len:, :]
    #         v_total = v_total[:, :, -self.max_len:, :]
    #         self.current_seq_len = self.max_len
    #     else:
    #         self.current_seq_len = k_total.size(2)

    #     self.cache[layer_idx] = (k_total, v_total)
    #     self.cache[layer_idx] = (
    #         k_total.contiguous().pin_memory(),
    #         v_total.contiguous().pin_memory(),
    #     )

        # print(f"append: layer {layer_idx}, new KV shape: {k_new.shape}, total KV shape: {k_total.shape}")
    
    # ---- 提交接口（主线程调用，极快返回）----
    def submit_append(self, layer_idx: int, key_new: torch.Tensor, value_new: torch.Tensor,
                      block: bool = False, timeout_ms: float = 0.0) -> bool:
        """
        提交一次 KV 追加任务。不会创建新线程。
        - block=False：队列满立即返回 False（最小化停顿）
        - block=True：等待直到有空位或超时
        """
        item = (layer_idx, key_new, value_new)
        if not block:
            try:
                self._q.put_nowait(item)
                return True
            except queue.Full:
                return False
        else:
            timeout = None if timeout_ms <= 0 else timeout_ms / 1000.0
            try:
                self._q.put(item, block=True, timeout=timeout)
                return True
            except queue.Full:
                return False
            
    # ---- 常驻线程主循环（顺序消费任务，可选攒批）----
    def _loop(self):
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.1)  # 0.1秒没人投任务就继续下一轮
                layer_idx, k, v = item
                self.append_new(layer_idx, k, v)     # 直接处理一个任务
                self._q.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    # ---- 真正执行 KV 追加（在工作线程里调用）----
    @torch.no_grad()
    def append_new(self, layer_idx: int, key_new: torch.Tensor, value_new: torch.Tensor):
        """
        注意：此方法现在由后台工作线程调用。
        这里不做 torch.cuda.synchronize()，避免全局同步。
        """

        # 1) 搬到 CPU 并 pinned（避免二次 pin）
        #    你之前最后又 .contiguous().pin_memory()，这里简化为只 pin 一次
        k_new_cpu = key_new.detach().cpu().pin_memory() if key_new.is_cuda else key_new.detach().pin_memory()
        v_new_cpu = value_new.detach().cpu().pin_memory() if value_new.is_cuda else value_new.detach().pin_memory()

        # 2) 追加到对应层
        k_total, v_total = None, None
        # print(k_new_cpu.shape)
        if self.cache[layer_idx] is None:
            # k_total, v_total = k_new_cpu[:,:,:4000,:], v_new_cpu[:,:,:4000,:]
            k_total, v_total = k_new_cpu, v_new_cpu
        else:
            k_old, v_old = self.cache[layer_idx]
            # 注意：cat 会产生新张量；若频繁增长，可考虑块式增长策略
            k_total = torch.cat([k_old, k_new_cpu], dim=2)
            v_total = torch.cat([v_old, v_new_cpu], dim=2)

        # 3) 截断为最近 max_len
        if k_total.size(2) > self.max_len:
            k_total = k_total[:, :, -self.max_len:, :]
            v_total = v_total[:, :, -self.max_len:, :]
            self.current_seq_len = self.max_len
        else:
            self.current_seq_len = k_total.size(2)
            

        # 4) 确保连续 + 仅在需要时 pin（避免重复 pin 的开销）
        if not k_total.is_contiguous():
            k_total = k_total.contiguous()
        if not v_total.is_contiguous():
            v_total = v_total.contiguous()
        if not k_total.is_pinned():
            k_total = k_total.pin_memory()
        if not v_total.is_pinned():
            v_total = v_total.pin_memory()
        
        lock = self._locks[layer_idx]
        with lock:
            self.cache[layer_idx]   = (k_total, v_total)   # 原子交换引用
            self._seq_len[layer_idx] = self.current_seq_len
            self._versions[layer_idx] += 1     


    # ---- 可选：优雅关闭 ----
    def shutdown(self, wait: bool = False, drain: bool = True, timeout_s: Optional[float] = None):
        """
        - drain=True：等待队列清空（不阻塞主线提交路径），再退出
        - wait=True：等待后台线程结束
        """
        if drain:
            try:
                self._q.join()
            except Exception:
                pass
        self._stop.set()
        if wait:
            self._worker.join(timeout=timeout_s)

     # -------- 读侧 API：拿一致快照（无锁）--------
    def get_snapshot(self, layer_idx: int) -> Tuple[Optional[torch.Tensor],
                                                    Optional[torch.Tensor],
                                                    int, int]:
        """
        返回 (k, v, seq_len, version)
        - k/v 可能为 None（尚未写入）
        - 读者拿到后，保持这些引用使用即可；Python 引用计数保证对象不会在使用过程中被释放
        """
        # 无锁：原子读取引用；需要强一致可校验版本后重试
        entry = self.cache[layer_idx]
        if entry is None:
            return None, None, 0, self._versions[layer_idx]
        k, v = entry
        return k, v, self._seq_len[layer_idx], self._versions[layer_idx]











