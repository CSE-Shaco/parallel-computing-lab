#include "kernels.cuh"
namespace second_moment {
__global__ void baseline(const float* x, float* out, int m, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;
    float sum = 0;
    for (int e = 0; e < m; ++e)
        sum += x[static_cast<std::size_t>(e)*n+row] * x[static_cast<std::size_t>(e)*n+col];
    out[static_cast<std::size_t>(row)*n+col] = sum / m;
}
void launch_baseline(const float* x, float* out, int m, int n) {
    const dim3 block(tile_size, tile_size);
    baseline<<<dim3((n+tile_size-1)/tile_size, (n+tile_size-1)/tile_size), block>>>(x,out,m,n);
    CUDA_CHECK(cudaGetLastError());
}
}
