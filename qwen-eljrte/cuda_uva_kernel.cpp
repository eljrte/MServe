// CUDA UVA Gather Kernel
// 实现GPU直接通过UVA访问CPU pinned memory

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel: 使用UVA从CPU pinned memory gather数据到GPU
// 注意：k_in_ptr和v_in_ptr是CPU pinned memory的指针，但通过UVA可以被GPU直接访问
template <typename T>
__global__ void uva_gather_kernel(
    T* __restrict__ k_out,
    T* __restrict__ v_out,
    const T* __restrict__ k_in,   // CPU pinned memory, accessed via UVA
    const T* __restrict__ v_in,   // CPU pinned memory, accessed via UVA
    const int64_t* __restrict__ indices,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int num_selected
) {
    // 每个线程处理一个元素
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * num_selected * head_dim;

    if (tid >= total_elements) return;

    // 解码索引
    int d = tid % head_dim;
    int s = (tid / head_dim) % num_selected;
    int h = (tid / (head_dim * num_selected)) % num_heads;
    int b = tid / (head_dim * num_selected * num_heads);

    // 获取要gather的源索引
    int src_idx = indices[s];

    // 计算源和目标偏移
    int src_offset = ((b * num_heads + h) * seq_len + src_idx) * head_dim + d;
    int dst_offset = ((b * num_heads + h) * num_selected + s) * head_dim + d;

    // UVA访问：GPU直接读取CPU pinned memory
    k_out[dst_offset] = k_in[src_offset];
    v_out[dst_offset] = v_in[src_offset];
}

// Host函数：调用CUDA kernel
std::vector<torch::Tensor> uva_gather_cuda(
    torch::Tensor k_in,
    torch::Tensor v_in,
    torch::Tensor indices,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // 检查输入
    TORCH_CHECK(k_in.is_pinned(), "k_in must be pinned memory for UVA");
    TORCH_CHECK(v_in.is_pinned(), "v_in must be pinned memory for UVA");
    TORCH_CHECK(k_in.is_cpu(), "k_in must be on CPU (pinned)");
    TORCH_CHECK(v_in.is_cpu(), "v_in must be on CPU (pinned)");
    TORCH_CHECK(indices.is_cpu(), "indices must be on CPU");

    int num_selected = indices.size(0);

    // 创建输出tensor（GPU上）
    auto options = torch::TensorOptions()
        .dtype(k_in.dtype())
        .device(torch::kCUDA);

    torch::Tensor k_out = torch::empty({batch_size, num_heads, num_selected, head_dim}, options);
    torch::Tensor v_out = torch::empty({batch_size, num_heads, num_selected, head_dim}, options);

    // 计算grid和block大小
    int total_elements = batch_size * num_heads * num_selected * head_dim;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // 启动kernel
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        k_in.scalar_type(),
        "uva_gather_kernel",
        [&] {
            uva_gather_kernel<scalar_t><<<blocks, threads_per_block>>>(
                k_out.data_ptr<scalar_t>(),
                v_out.data_ptr<scalar_t>(),
                k_in.data_ptr<scalar_t>(),
                v_in.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(),
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                num_selected
            );
        }
    );

    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\n", cudaGetErrorString(err));
    }

    return {k_out, v_out};
}

// 绑定到Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uva_gather", &uva_gather_cuda, "UVA Gather (GPU reads CPU pinned memory directly)");
}
