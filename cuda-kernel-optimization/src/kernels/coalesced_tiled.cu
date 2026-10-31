#include "kernels.cuh"
#include "tile_layout.hpp"

namespace second_moment {

__global__ void coalesced_tiled(const float* x, float* out, int m, int n) {
    using Layout = CoalescedLayout;
    __shared__ float left[Layout::tile][Layout::tile];
    __shared__ float right[Layout::tile][Layout::tile];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row_base = blockIdx.y * Layout::tile;
    const int col_base = blockIdx.x * Layout::tile;
    const int row = row_base + ty, col = col_base + tx;
    float sum = 0.0f;

    for (int base = 0; base < m; base += Layout::tile) {
        const int sample = base + Layout::row_offset(ty, 0);
        // Adjacent x lanes read adjacent features from the same input row.
        // Keep sample-by-feature storage; transpose through the read indices.
        left[ty][tx] = input_in_bounds(sample, row_base + tx, m, n)
            ? x[input_index(sample, row_base + tx, n)] : 0.0f;
        right[ty][tx] = input_in_bounds(sample, col, m, n)
            ? x[input_index(sample, col, n)] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < Layout::tile; ++k)
            sum += left[k][ty] * right[k][tx];
        // Out-of-range output lanes still participate in loading and barriers.
        __syncthreads();
    }
    if (row < n && col < n) out[input_index(row, col, n)] = sum / m;
}

void launch_coalesced_tiled(const float* x, float* out, int m, int n) {
    using Layout = CoalescedLayout;
    coalesced_tiled<<<dim3((n + Layout::tile - 1) / Layout::tile,
                          (n + Layout::tile - 1) / Layout::tile),
                     dim3(Layout::thread_columns, Layout::thread_rows)>>>(x, out, m, n);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace second_moment
