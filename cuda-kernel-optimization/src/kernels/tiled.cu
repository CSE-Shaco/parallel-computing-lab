#include "kernels.cuh"
namespace second_moment {
__global__ void tiled(const float* x, float* out, int m, int n) {
    __shared__ float left[tile_size][tile_size];
    __shared__ float right[tile_size][tile_size];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * tile_size + ty;
    const int col = blockIdx.x * tile_size + tx;
    float sum = 0;
    for (int base = 0; base < m; base += tile_size) {
        left[ty][tx] = row < n && base+tx < m ? x[static_cast<std::size_t>(base+tx)*n+row] : 0;
        right[ty][tx] = col < n && base+ty < m ? x[static_cast<std::size_t>(base+ty)*n+col] : 0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < tile_size; ++k) sum += left[ty][k] * right[k][tx];
        // All threads reach both barriers, including threads outside the output boundary.
        __syncthreads();
    }
    if (row < n && col < n) out[static_cast<std::size_t>(row)*n+col] = sum / m;
}
void launch_tiled(const float* x, float* out, int m, int n) {
    tiled<<<dim3((n+tile_size-1)/tile_size,(n+tile_size-1)/tile_size),dim3(tile_size,tile_size)>>>(x,out,m,n);
    CUDA_CHECK(cudaGetLastError());
}
}
