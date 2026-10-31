#pragma once

#include <vector>

namespace second_moment {

std::vector<float> input(int samples, int features);
std::vector<double> reference(const std::vector<float>& x, int samples, int features);
double validate(const std::vector<float>& got, const std::vector<double>& expected);

}  // namespace second_moment
