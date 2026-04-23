import torch
from typing import Optional, Tuple, List
import threading
import queue
import time

device = "cuda:0"

class CPUKVCacheOptimized:
    """优化版 CPU KV Cache - 主线程非阻塞 D2H，后台线程只做 CPU 操作"""

    def __init__(self, num_layers: int, max_len: int, queue_maxsize: int = 256):
        self.num_layers = num_layers
        self.max_len = max_len
        self.cache = [None] * num_layers
        self.current_seq_len = 0

        # GPU buffer 池（用于 get_async）
        self._gpu_k_buffer: List[Optional[torch.Tensor]] = [None] * num_layers
        self._gpu_v_buffer: List[Optional[torch.Tensor]] = [None] * num_layers
        self._gpu_buffer_capacity: List[int] = [0] * num_layers

        # D2H 预缓冲池（用于 submit_append）
        # 每层预分配 pinned CPU buffer，复用避免重复分配
        self._d2h_k_buffer: List[Optional[torch.Tensor]] = [None] * num_layers
        self._d2h_v_buffer: List[Optional[torch.Tensor]] = [None] * num_layers
        self._d2h_events: List[Optional[torch.cuda.Event]] = [None] * num_layers

        self._h2d_stream = torch.cuda.Stream(device=device)
        self._d2h_stream = torch.cuda.Stream(device=device)

        self._locks = {i: threading.Lock() for i in range(num_layers)}
        self._seq_len: List[int] = [0] * num_layers
        self._versions: List[int] = [0] * num_layers

        # 任务队列 - 现在存的是 CPU tensor
        self._q: "queue.Queue[Tuple[int, torch.Tensor, torch.Tensor]]" = queue.Queue(maxsize=queue_maxsize)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, name="kv-worker", daemon=True)
        self._worker.start()

    def get_seq_len(self):
        return self.current_seq_len

    def _ensure_d2h_buffer(self, layer_idx: int, shape: Tuple, dtype: torch.dtype):
        """确保该层有预分配的 pinned CPU buffer 用于 D2H"""
        if (self._d2h_k_buffer[layer_idx] is None or
            self._d2h_k_buffer[layer_idx].shape != shape or
            self._d2h_k_buffer[layer_idx].dtype != dtype):

            self._d2h_k_buffer[layer_idx] = torch.empty(shape, dtype=dtype, pin_memory=True)
            self._d2h_v_buffer[layer_idx] = torch.empty(shape, dtype=dtype, pin_memory=True)

    def submit_append(self, layer_idx: int, key_new: torch.Tensor, value_new: torch.Tensor,
                      block: bool = False, timeout_ms: float = 0.0) -> bool:
        """
        优化版提交：主线程用非阻塞 D2H，只提交轻量任务到队列
        """
        if not key_new.is_cuda:
            # 已经是 CPU tensor，直接提交
            item = (layer_idx, key_new, value_new)
        else:
            # === 关键优化：主线程用非阻塞 D2H ===
            self._ensure_d2h_buffer(layer_idx, key_new.shape, key_new.dtype)

            k_buf = self._d2h_k_buffer[layer_idx]
            v_buf = self._d2h_v_buffer[layer_idx]

            # 在专用 D2H stream 上异步拷贝
            with torch.cuda.stream(self._d2h_stream):
                k_buf.copy_(key_new, non_blocking=True)
                v_buf.copy_(value_new, non_blocking=True)

                # 记录完成事件
                evt = torch.cuda.Event()
                evt.record(self._d2h_stream)
                self._d2h_events[layer_idx] = evt

            # 提交的是 CPU buffer 引用和事件
            item = (layer_idx, k_buf, v_buf, evt)

        # 提交到队列（极快，不涉及数据拷贝）
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

    def _loop(self):
        """后台线程：只做 CPU 操作，不碰 CUDA"""
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.05)

                if len(item) == 4:
                    layer_idx, k_cpu, v_cpu, evt = item
                    # 等待 D2H 完成（如果还在传输中）
                    if evt is not None:
                        evt.synchronize()
                else:
                    layer_idx, k_cpu, v_cpu = item

                self._append_cpu(layer_idx, k_cpu, v_cpu)
                self._q.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"KV cache worker error: {e}")

    @torch.no_grad()
    def _append_cpu(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """
        纯 CPU 操作：拼接、截断、更新缓存
        此时 k_new, v_new 已经在 CPU pinned memory 中
        """
        lock = self._locks[layer_idx]
        with lock:
            if self.cache[layer_idx] is None:
                k_total, v_total = k_new, v_new
            else:
                k_old, v_old = self.cache[layer_idx]
                k_total = torch.cat([k_old, k_new], dim=2)
                v_total = torch.cat([v_old, v_new], dim=2)

            # 截断
            if k_total.size(2) > self.max_len:
                k_total = k_total[:, :, -self.max_len:, :]
                v_total = v_total[:, :, -self.max_len:, :]
                self.current_seq_len = self.max_len
            else:
                self.current_seq_len = k_total.size(2)

            # 确保连续（cat 可能产生非连续张量）
            if not k_total.is_contiguous():
                k_total = k_total.contiguous()
            if not v_total.is_contiguous():
                v_total = v_total.contiguous()

            self.cache[layer_idx] = (k_total, v_total)
            self._seq_len[layer_idx] = self.current_seq_len
            self._versions[layer_idx] += 1

    def get_snapshot(self, layer_idx: int):
        """获取缓存快照（用于 get_async）"""
        entry = self.cache[layer_idx]
        if entry is None:
            return None, None, 0, self._versions[layer_idx]
        k, v = entry
        return k, v, self._seq_len[layer_idx], self._versions[layer_idx]

    def get_async(self, layer_idx: int, device: torch.device):
        """异步获取 KV 到 GPU（复用预分配 buffer）"""
        k_cpu, v_cpu, seq_len, ver = self.get_snapshot(layer_idx)
        if k_cpu is None:
            return None

        assert k_cpu.is_pinned() and v_cpu.is_pinned()
        B, H, T_cpu, D = k_cpu.shape

        # 懒分配 GPU buffer
        if (self._gpu_k_buffer[layer_idx] is None or
            self._gpu_buffer_capacity[layer_idx] < T_cpu):
            alloc_len = max(self.max_len, T_cpu)
            self._gpu_k_buffer[layer_idx] = torch.empty((B, H, alloc_len, D),
                                                         dtype=k_cpu.dtype, device=device)
            self._gpu_v_buffer[layer_idx] = torch.empty((B, H, alloc_len, D),
                                                         dtype=v_cpu.dtype, device=device)
            self._gpu_buffer_capacity[layer_idx] = alloc_len

        k_gpu_buf = self._gpu_k_buffer[layer_idx]
        v_gpu_buf = self._gpu_v_buffer[layer_idx]

        with torch.cuda.stream(self._h2d_stream):
            k_gpu_buf[:, :, :T_cpu, :].copy_(k_cpu, non_blocking=True)
            v_gpu_buf[:, :, :T_cpu, :].copy_(v_cpu, non_blocking=True)
            done_event = torch.cuda.Event()
            done_event.record(self._h2d_stream)

        return k_gpu_buf[:, :, :T_cpu, :], v_gpu_buf[:, :, :T_cpu, :], done_event

    def shutdown(self, wait: bool = False, drain: bool = True, timeout_s: Optional[float] = None):
        if drain:
            try:
                self._q.join()
            except Exception:
                pass
        self._stop.set()
        if wait:
            self._worker.join(timeout=timeout_s)
