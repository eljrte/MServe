import threading
import numpy as np

def start_cpu_background_compute(threads=4, mat_n=512, iters=10_000):
    """
    启动若干个 CPU 线程做大矩阵乘/点积，制造强 CPU 占用。
    返回 (stop_event, [threads])
    """
    stop = threading.Event()
    ths = []

    def worker():
        # 用 numpy（或 torch.cpu）都可；numpy 通常会调用 MKL/OpenBLAS，负载更高
        a = np.random.randn(mat_n, mat_n).astype(np.float32)
        b = np.random.randn(mat_n, mat_n).astype(np.float32)
        i = 0
        while not stop.is_set() and i < iters:
            # 矩阵乘运算（CPU），不保留结果，重点是占用 CPU
            _ = a @ b
            i += 1

    for _ in range(threads):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        ths.append(t)

    return stop, ths