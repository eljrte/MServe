// CUDA UVA Gather Kernel - Simplified Version
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// float32 kernel
__global__ void uva_gather_kernel_f32(
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    const int64_t* __restrict__ indices,
    int batch_size, int num_heads, int seq_len, int head_dim, int num_selected
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * num_selected * head_dim;
    if (tid >= total) return;

    int d = tid % head_dim;
    int s = (tid / head_dim) % num_selected;
    int h = (tid / (head_dim * num_selected)) % num_heads;
    int b = tid / (head_dim * num_selected * num_heads);

    int src_idx = indices[s];
    int src_off = ((b * num_heads + h) * seq_len + src_idx) * head_dim + d;
    int dst_off = ((b * num_heads + h) * num_selected + s) * head_dim + d;

    k_out[dst_off] = k_in[src_off];
    v_out[dst_off] = v_in[src_off];
}

// float16 kernel
__global__ void uva_gather_kernel_f16(
    __half* __restrict__ k_out,
    __half* __restrict__ v_out,
    const __half* __restrict__ k_in,
    const __half* __restrict__ v_in,
    const int64_t* __restrict__ indices,
    int batch_size, int num_heads, int seq_len, int head_dim, int num_selected
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * num_selected * head_dim;
    if (tid >= total) return;

    int d = tid % head_dim;
    int s = (tid / head_dim) % num_selected;
    int h = (tid / (head_dim * num_selected)) % num_heads;
    int b = tid / (head_dim * num_selected * num_heads);

    int src_idx = indices[s];
    int src_off = ((b * num_heads + h) * seq_len + src_idx) * head_dim + d;
    int dst_off = ((b * num_heads + h) * num_selected + s) * head_dim + d;

    k_out[dst_off] = k_in[src_off];
    v_out[dst_off] = v_in[src_off];
}

// bfloat16 kernel
__global__ void uva_gather_kernel_bf16(
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const int64_t* __restrict__ indices,
    int batch_size, int num_heads, int seq_len, int head_dim, int num_selected
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * num_selected * head_dim;
    if (tid >= total) return;

    int d = tid % head_dim;
    int s = (tid / head_dim) % num_selected;
    int h = (tid / (head_dim * num_selected)) % num_heads;
    int b = tid / (head_dim * num_selected * num_heads);

    int src_idx = indices[s];
    int src_off = ((b * num_heads + h) * seq_len + src_idx) * head_dim + d;
    int dst_off = ((b * num_heads + h) * num_selected + s) * head_dim + d;

    k_out[dst_off] = k_in[src_off];
    v_out[dst_off] = v_in[src_off];
}

// Host wrapper
std::vector<torch::Tensor> uva_gather(
    torch::Tensor k_in, torch::Tensor v_in, torch::Tensor indices,
    int batch_size, int num_heads, int seq_len, int head_dim
) {
    TORCH_CHECK(k_in.is_pinned(), "k_in must be pinned memory");
    TORCH_CHECK(v_in.is_pinned(), "v_in must be pinned memory");

    int num_selected = indices.size(0);
    auto options = torch::TensorOptions().dtype(k_in.dtype()).device(torch::kCUDA);
    torch::Tensor k_out = torch::empty({batch_size, num_heads, num_selected, head_dim}, options);
    torch::Tensor v_out = torch::empty({batch_size, num_heads, num_selected, head_dim}, options);

    int total = batch_size * num_heads * num_selected * head_dim;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    if (k_in.scalar_type() == torch::kFloat32) {
        uva_gather_kernel_f32<<<blocks, threads>>>(
            k_out.data_ptr<float>(), v_out.data_ptr<float>(),
            k_in.data_ptr<float>(), v_in.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            batch_size, num_heads, seq_len, head_dim, num_selected
        );
    } else if (k_in.scalar_type() == torch::kFloat16) {
        uva_gather_kernel_f16<<<blocks, threads>>>(
            k_out.data_ptr<at::Half>(), v_out.data_ptr<at::Half>(),
            k_in.data_ptr<at::Half>(), v_in.data_ptr<at::Half>(),
            indices.data_ptr<int64_t>(),
            batch_size, num_heads, seq_len, head_dim, num_selected
        );
    } else if (k_in.scalar_type() == torch::kBFloat16) {
        uva_gather_kernel_bf16<<<blocks, threads>>>(
            k_out.data_ptr<at::BFloat16>(), v_out.data_ptr<at::BFloat16>(),
            k_in.data_ptr<at::BFloat16>(), v_in.data_ptr<at::BFloat16>(),
            indices.data_ptr<int64_t>(),
            batch_size, num_heads, seq_len, head_dim, num_selected
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }

    cudaDeviceSynchronize();
    return {k_out, v_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uva_gather", &uva_gather, "UVA Gather");
}
