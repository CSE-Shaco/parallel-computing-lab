#pragma once
#include "kernels.cuh"
#include "reference.hpp"
namespace second_moment {
inline double check_output(Launcher launch, const std::vector<float>& x,
                           const std::vector<double>& expected, int m, int n,
                           DeviceBuffer& device_x, DeviceBuffer& device_out) {
    CUDA_CHECK(cudaMemset(device_out.pointer, 0xff, expected.size()*sizeof(float)));
    launch(device_x.pointer, device_out.pointer, m, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(expected.size());
    CUDA_CHECK(cudaMemcpy(got.data(), device_out.pointer, got.size()*sizeof(float), cudaMemcpyDeviceToHost));
    return validate(got, expected);
}
}
