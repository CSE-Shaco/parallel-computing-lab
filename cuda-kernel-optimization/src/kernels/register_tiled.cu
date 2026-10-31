#include "kernels.cuh"
#include "tile_layout.hpp"

namespace second_moment {

__global__ void register_tiled(const float* x, float* out, int m, int n) {
    using Layout = RegisterLayout;
    __shared__ float left[Layout::tile][Layout::tile];
    __shared__ float right[Layout::tile][Layout::tile];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row_base = blockIdx.y * Layout::tile;
    const int col_base = blockIdx.x * Layout::tile;
    const int col = col_base + tx;
    float sums[Layout::outputs_per_thread] = {};

    for (int base = 0; base < m; base += Layout::tile) {
        #pragma unroll
        for (int slot = 0; slot < Layout::outputs_per_thread; ++slot) {
            const int local_row = Layout::row_offset(ty, slot);
            const int sample = base + local_row;
            left[local_row][tx] = input_in_bounds(sample, row_base + tx, m, n)
                ? x[input_index(sample, row_base + tx, n)] : 0.0f;
            right[local_row][tx] = input_in_bounds(sample, col, m, n)
                ? x[input_index(sample, col, n)] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < Layout::tile; ++k) {
            // One shared right value contributes to four register accumulators.
            const float right_value = right[k][tx];
            #pragma unroll
            for (int slot = 0; slot < Layout::outputs_per_thread; ++slot)
                sums[slot] += left[k][Layout::row_offset(ty, slot)] * right_value;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int slot = 0; slot < Layout::outputs_per_thread; ++slot) {
        const int row = row_base + Layout::row_offset(ty, slot);
        if (row < n && col < n) out[input_index(row, col, n)] = sums[slot] / m;
    }
}

void launch_register_tiled(const float* x, float* out, int m, int n) {
    using Layout = RegisterLayout;
    register_tiled<<<dim3((n + Layout::tile - 1) / Layout::tile,
                         (n + Layout::tile - 1) / Layout::tile),
                    dim3(Layout::thread_columns, Layout::thread_rows)>>>(x, out, m, n);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace second_moment
