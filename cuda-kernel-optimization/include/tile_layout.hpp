#pragma once

#include <cstddef>

#if defined(__CUDACC__)
#define SECOND_MOMENT_HD __host__ __device__
#else
#define SECOND_MOMENT_HD
#endif

namespace second_moment {

// A block cooperatively loads two sample-by-feature tiles. Each thread owns
// one feature column and Tile / ThreadRows sample/output rows.
template <int Tile, int ThreadRows>
struct TileLayout {
    static_assert(Tile > 0 && ThreadRows > 0 && Tile % ThreadRows == 0,
                  "thread rows must divide the tile");
    static constexpr int tile = Tile;
    static constexpr int thread_columns = Tile;
    static constexpr int thread_rows = ThreadRows;
    static constexpr int outputs_per_thread = Tile / ThreadRows;

    SECOND_MOMENT_HD static constexpr int row_offset(int ty, int slot) {
        return ty + slot * ThreadRows;
    }
};

using CoalescedLayout = TileLayout<16, 16>;
using RegisterLayout = TileLayout<32, 8>;

SECOND_MOMENT_HD inline constexpr bool input_in_bounds(
    int sample, int feature, int samples, int features) {
    return sample < samples && feature < features;
}

SECOND_MOMENT_HD inline constexpr std::size_t input_index(
    int sample, int feature, int features) {
    return static_cast<std::size_t>(sample) * features + feature;
}

}  // namespace second_moment

#undef SECOND_MOMENT_HD
