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

## Four-kernel measurement: gains and negative ablation

The [RTX 4060 WSL2 run 03](../results/rtx4060-wsl-run-03/README.md) compiled source snapshot `33c5536` and passed 12 GPU correctness shapes for every variant, then memcheck and racecheck. It measured all four implementations together, using ten warmups, rotating order, and 40 samples per kernel/shape. Ratios below are **same-run baseline median ÷ candidate median**; values below 1 mean that candidate was slower.

| Input shape | Original tiled | Coalesced tiled | Register tiled |
| --- | ---: | ---: | ---: |
| 257 × 65 | 0.900× | 0.690× | 0.601× |
| 1024 × 128 | 1.006× | 0.845× | 1.067× |
| 4096 × 256 | 1.037× | 0.863× | 1.305× |
| 1024 × 512 | 1.041× | 0.872× | 1.405× |

The coalesced-load variant lost to baseline on every tested shape. That negative result is part of the ablation: contiguous cooperative input loads do not ensure a lower kernel time after shared-memory accesses, indexing, synchronization and compiler choices are included. No counter evidence identifies the limiting factor here, so bank conflicts, cache effects and barriers are hypotheses to profile, not established explanations.

Register tiling reduced median time by about 23.40% at 4096×256 and 28.83% at 1024×512, but increased it by about 66.32% at 257×65. It combines a 32×32 tile, 32×8 block, four accumulators per thread and reuse across outputs. This is an implementation-level comparison, not a controlled single-variable proof that one of those changes caused the improvement. The source expresses register-reuse intent; no compiler register report, occupancy measurement or profiler counter was captured to establish actual placement or pressure.

## Repeated-sweep outcome

The [five-sweep paired analysis](../results/rtx4060-repeat-20261031T095734Z-analysis/README.md) repeated the same four-way experiment on the same host/day. It summarizes a candidate's baseline-relative median-time ratio within each sweep, then reports the median/range of those five ratios:

| Shape | Original tile | Coalesced tile | Register tile | Register ratio range |
| --- | ---: | ---: | ---: | ---: |
| 257 × 65 | 0.906× | 0.692× | 0.600× | 0.596–0.600× |
| 1024 × 128 | 1.007× | 0.845× | 1.067× | 1.066–1.067× |
| 4096 × 256 | 1.035× | 0.862× | 1.304× | 1.302–1.306× |
| 1024 × 512 | 1.042× | 0.872× | 1.406× | 1.390–1.406× |

Coalesced tiling remains below baseline on every shape in every sweep; register tiling remains below baseline on the smallest case and above it in all five sweeps on each larger case. This strengthens the descriptive repeatability evidence for those workloads, while leaving the negative ablation and causal uncertainty intact. The best five-sweep median ratio 1.406× corresponds to about 28.85% median paired runtime reduction, not a 40.6% reduction. No launch pooling, cross-session denominator or best-run selection is used.

## Follow-up validation and profiling

Keep all recorded inputs immutable. The five consecutive sweeps are still one same-host/day sequence without controlled thermal/clock/background-load states. Further controlled experiments and profiling are needed before generalizing beyond this GPU, workload and environment. Forty launches within a sweep do not create forty independent sessions, and historical sessions must not supply denominators for new sweeps.

When supported in the chosen WSL setup, use Nsight Compute to inspect global-load efficiency, cache behavior, barrier stalls, actual register usage and occupancy for the same inputs and build. Potential later work includes tile-size sweeps, symmetry-aware work reduction and cuBLAS comparison. None of those follow-ups is represented as implemented or measured here.
