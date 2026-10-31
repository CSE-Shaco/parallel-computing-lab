# Tiling design and hypotheses

For `m×n` row-major input, direct computation performs `m` products for each of `n²` outputs, repeatedly accessing feature columns. A 16×16 output tile stages a 16-sample slice for its row and column features in shared memory, then reuses these values across threads. The reduction maintains the same sample order as the direct kernel.

The tiled kernel loads `left[ty][tx] = X[base+tx,row]` and `right[ty][tx] = X[base+ty,col]`. Each thread computes the sum of `left[ty][k]*right[k][tx]`. Out-of-range loads become zero, and all threads participate in both barriers. The final store alone is output-bounds guarded.

This reduces redundant global accesses within a block, but it introduces shared-memory traffic and two barriers per tile. Left-side column-oriented loads are strided and may limit memory efficiency; no coalescing benefit is claimed for them. The recorded RTX 4060 run provides a first shape-dependent timing comparison, rather than a guarantee that these tradeoffs always favor tiling. No claim is made that this kernel competes with optimized BLAS implementations.

Potential follow-ups: transpose input or cooperative load layout, test tile dimensions, examine occupancy/register pressure, exploit matrix symmetry, and compare with `cublasSsyrk` under identical mathematical and timing conditions. None of these follow-ups are implemented or benchmarked.

## Recorded outcome

In [one RTX 4060 WSL2 session](../results/rtx4060-wsl-run-02/README.md), the tiled kernel was about 11.93% slower at 257×65, used about 1.79% less elapsed kernel time at 1024×128 and about 3.60% less at 4096×256. Each estimate compares median kernel times from 30 launches. Small workloads may not amortize tiling and synchronization costs, but no profiler data isolates the cause here. No occupancy, bandwidth or cache measurement was captured, and the fixed clocks/thermal/background-load conditions required for stronger causal claims were not established. Repeat independent sweeps before interpreting the small differences as robust optimization gains.
