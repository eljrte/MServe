import torch
import time
import os
import nvtx
import torch.cuda.nvtx as nvtx2
def vit_simulation():
    # --- 1. 配置参数 ---
    # ViT Giant/Huge 级别的参数
    NUM_LAYERS = 32
    SEQ_LEN = 4000      # token 长度
    HIDDEN_SIZE = 1280   # 嵌入维度
    INTERMEDIATE_SIZE = 3420 # MLP 中间维度
    
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # print(f"Process ID: {os.getpid()}")
    # print(f"Device: {DEVICE}")
    # print(f"Configuration: Layers={NUM_LAYERS}, Seq={SEQ_LEN}, Hidden={HIDDEN_SIZE}, MLP_Inter={INTERMEDIATE_SIZE}")
    # print("-" * 40)

    # --- 2. 初始化 Tensor (占用显存) ---
    # 使用 float32 以产生更大的计算压力
    # 输入 Batch Size = 1
    input_tensor = torch.randn(SEQ_LEN, HIDDEN_SIZE, device=DEVICE)
    
    # 模拟 Attention 的权重 (简化为一次由 Hidden -> Hidden 的投影)
    # 实际上 Attention 包含 Q,K,V,O 四个投影，这里合并负载为一个大矩阵乘法
    w_att = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE)
    
    # MLP 部分的权重
    w_mlp_up = torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE, device=DEVICE)    # 升维
    w_mlp_down = torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, device=DEVICE)  # 降维

    # 预热 GPU
    # for _ in range(5):
    #     torch.matmul(input_tensor, w_att)

    print("开始模拟负载 (Press Ctrl+C to stop)...")
    


    # --- 3. 模拟 32 层 Transformer Block ---
    # 这是一个纯计算循环，不保存梯度，只消耗算力
    
    # 为了避免 pytorch 优化掉未使用的计算，我们在每层传递 x
    # 但为了避免数值爆炸，我们每轮重置 x 为 input_tensor
    x = input_tensor 

    with torch.no_grad():
        with nvtx.annotate(f"encoding", color="green"):
            for i in range(NUM_LAYERS):

                # A. Attention 模块 (简化模拟)
                # [10000, 1280] x [1280, 1280] -> [10000, 1280]
                # 真正的 Attention 还有 Score 计算 (N^2)，但在 Hidden 较大时，投影也是主要负载之一
                nvtx2.range_push("My visual attn Layer")
                x = torch.matmul(x, w_att)
                nvtx2.range_pop()
                
                nvtx2.range_push("My visual MLP Layer")
                # B. MLP 模块
                # 1. 升维 (Up-projection/GeLU part)
                # [10000, 1280] x [1280, 3420] -> [10000, 3420]
                mlp_inter = torch.matmul(x, w_mlp_up)
                
                # (可选) 模拟非线性激活函数的开销，虽然比起矩阵乘法很小
                # mlp_inter = torch.nn.functional.gelu(mlp_inter)

                # 2. 降维 (Down-projection)
                # [10000, 3420] x [3420, 1280] -> [10000, 1280]
                x = torch.matmul(mlp_inter, w_mlp_down)
                nvtx2.range_pop()
            torch.cuda.synchronize()

    print("Encoding结束")



if __name__ == "__main__":
    vit_simulation()