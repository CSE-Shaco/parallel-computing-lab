# CUDA kernel variants and optimization rationale

All variants compute the same uncentered second-moment matrix for row-major input `X` with `m` samples and `n` features:

```text
S[i,j] = sum(e = 0 .. m-1, X[e,i] * X[e,j]) / m
```

This is `XᵀX/m`; it does not subtract feature means. CUDA arithmetic is single precision, and the supplied bounded-input checks use `2e-4 + 2e-4*abs(reference)` absolute-plus-relative tolerance.

## Four-way ablation

The variants keep the operation and data identical while changing the staging/reuse strategy:

| Kernel | Output tile | Threads per block | Outputs per thread | Main change |
| --- | ---: | ---: | ---: | --- |
| `baseline` | 16×16 | 16×16 | 1 | Direct global-memory reads and one output per thread; scalar sample reduction |
| `tiled` | 16×16 | 16×16 | 1 | Original 16-sample shared-memory tiles for left and right operands |
| `coalesced_tiled` | 16×16 | 16×16 | 1 | Cooperatively load the left tile with adjacent lanes reading adjacent features; retains 16-sample reductions |
| `register_tiled` | 32×32 | 32×8 | 4 | Load 32-sample tiles and reuse the right operand across four row accumulators held per thread |

`tiled` remains the original measured implementation. `coalesced_tiled` changes the global-load orientation of the left operand to expose contiguous feature reads across adjacent lanes, then reads the transposed shared-memory tile during the reduction. `register_tiled` maps each 32-lane row of threads to adjacent output columns, gives each thread four output rows, and reuses a right-tile value across those accumulators. Both alternatives use zero padding for partial tiles, keep barriers unconditional across each block, and bounds-check final stores.

The host-side `tile_layout.hpp` helpers define index calculations shared by those new kernels. `tests/tile_layout_test.cpp` checks that cooperative loads cover their tiles, output indices have one owner, and boundary handling agrees with a CPU reference for aligned and partial dimensions. This can catch host-visible indexing mistakes; it cannot verify CUDA compilation, barriers, race behavior, register allocation, or device performance.

## What the first measurement did and did not show

The [frozen RTX 4060 WSL2 run](../results/rtx4060-wsl-run-02/README.md) measured only `baseline` and the original `tiled` kernel at source snapshot `d5a7070`. Tiling was about 11.93% slower at 257×65 and used about 1.79% and 3.60% less elapsed kernel time at 1024×128 and 4096×256. The recording has 30 launches per kernel and shape in one session; clocks, thermal state and background activity were uncontrolled. It has no profiler counters.

Those timings do not tell us which instruction or memory behavior limited either kernel, and they say nothing about `coalesced_tiled` or `register_tiled`. Strided input loads in the original tile and barrier work are concrete code properties; attributing the observed timings to them remains a hypothesis until counters and controlled repeats support it. The original shared-memory reads broadcast a repeated word within a warp, so bank conflicts are not established as the cause.

## Validation before making a speedup claim

Build and validate the current four-kernel source as one unit on the same RTX 4060 session. Run all 12 correctness shapes for all four variants, memcheck and racecheck, then measure all kernels on identical dimensions with the same compiler flags, warmups and repetition policy. The fresh run includes the two original kernels again so comparisons do not mix sessions or compiler configurations. Keep the historical two-kernel directory unchanged.

After a successful sweep, use Nsight Compute (when supported in the chosen WSL setup) to inspect global-load efficiency, cache behavior, barrier stalls, register usage and occupancy. Compare each relevant counter for the same input and build. These measurements can explain a timing difference; code structure alone cannot establish that the new variants are faster.

Potential later work includes tile-size sweeps, symmetry-aware work reduction, and cuBLAS comparison. None is represented as implemented or measured here.
