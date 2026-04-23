# 这个文件是用来验证 shadowkv的第二个motivation的
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def analyze_key_cosine_similarity(past_key_values, chunk_size=32, layer_index=16, head_index=2):
    # past_key_values 是 key tensor 的 list
    key = past_key_values[layer_index]  # shape: (batch, num_heads, seq_len, head_dim)

    key = key[0, head_index]  # (seq_len, head_dim)

    # print(key.shape)

    print("key std:", key.std())
    print("key max:", key.max())
    print("key min:", key.min())

    if key.dim() != 2:
        raise ValueError(f"Expected key of shape (seq_len, head_dim), but got {key.shape}")

    seq_len, head_dim = key.shape
    num_chunks = seq_len // chunk_size

    min_similarities = []

    for i in range(num_chunks):
        chunk = key[i * chunk_size: (i + 1) * chunk_size]  # (chunk_size, head_dim)
        mean_vec = chunk.mean(dim=0, keepdim=True)         # (1, head_dim)
        # Normalize for cosine sim
        chunk_norm = F.normalize(chunk, dim=1)
        mean_norm = F.normalize(mean_vec, dim=1)
        sims = torch.matmul(chunk_norm, mean_norm.T).squeeze(1)  # (chunk_size,)
        min_similarities.append(sims.min().item())

    # 可视化
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(min_similarities, marker='o', linestyle='-', alpha=0.7)
    plt.xlabel("Chunk index")
    plt.ylabel("Min Cosine Similarity in Chunk")
    plt.title(f"Layer {layer_index}, Head {head_index} - Min Cosine Similarity per Chunk")
    plt.grid(True)
    plt.savefig("k_chunk_similarity.png")
