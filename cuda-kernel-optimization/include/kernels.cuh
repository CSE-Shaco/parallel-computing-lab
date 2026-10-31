#pragma once
#include <cuda_runtime.h>
#include "tile_layout.hpp"
#include <array>
#include <stdexcept>
#include <string>
inline void cuda_check(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(expression) cuda_check((expression), #expression)
namespace second_moment {
constexpr int tile_size = 16;
void launch_baseline(const float* x, float* out, int m, int n);
void launch_tiled(const float* x, float* out, int m, int n);
void launch_coalesced_tiled(const float* x, float* out, int m, int n);
void launch_register_tiled(const float* x, float* out, int m, int n);
using Launcher = void (*)(const float*, float*, int, int);
struct KernelVariant {
    const char* name;
    Launcher launch;
    int block_x, block_y, output_tile, reduction_tile, outputs_per_thread;
};
inline const std::array<KernelVariant, 4> kernel_variants{{
    {"baseline", launch_baseline, 16, 16, 16, 1, 1},
    {"tiled", launch_tiled, 16, 16, 16, 16, 1},
    {"coalesced_tiled", launch_coalesced_tiled,
     CoalescedLayout::thread_columns, CoalescedLayout::thread_rows,
     CoalescedLayout::tile, CoalescedLayout::tile, CoalescedLayout::outputs_per_thread},
    {"register_tiled", launch_register_tiled,
     RegisterLayout::thread_columns, RegisterLayout::thread_rows,
     RegisterLayout::tile, RegisterLayout::tile, RegisterLayout::outputs_per_thread}
}};
struct DeviceBuffer {
    float* pointer = nullptr;
    explicit DeviceBuffer(std::size_t bytes) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), bytes)); }
    ~DeviceBuffer() { if (pointer) cudaFree(pointer); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct Event {
    cudaEvent_t value{};
    Event() { CUDA_CHECK(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
}
