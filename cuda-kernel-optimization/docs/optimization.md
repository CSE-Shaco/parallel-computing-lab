# Tiling design and hypotheses

For `m×n` row-major input, direct computation performs `m` products for each of `n²` outputs, repeatedly accessing feature columns. A 16×16 output tile stages a 16-sample slice for its row and column features in shared memory, then reuses these values across threads. The reduction maintains the same sample order as the direct kernel.

The tiled kernel loads `left[ty][tx] = X[base+tx,row]` and `right[ty][tx] = X[base+ty,col]`. Each thread computes the sum of `left[ty][k]*right[k][tx]`. Out-of-range loads become zero, and all threads participate in both barriers. The final store alone is output-bounds guarded.

This reduces redundant global accesses within a block, but it introduces shared-memory traffic and two barriers per tile. Left-side column-oriented loads are strided and may limit memory efficiency; no coalescing benefit is claimed for them. A faster runtime is a hypothesis until measured. No claim is made that this kernel competes with optimized BLAS implementations.

Potential follow-ups: transpose input or cooperative load layout, test tile dimensions, examine occupancy/register pressure, exploit matrix symmetry, and compare with `cublasSsyrk` under identical mathematical and timing conditions. None of these follow-ups are implemented or benchmarked.
