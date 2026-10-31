# OpenMP Workload Scheduling

**What I Built:** An independent scheduling experiment with static, dynamic, guided, and auto policies, configurable chunks, and uniform/tail-heavy integer tasks.

**How It Works:** Each task runs a deterministic sequence of unsigned shifts/XORs/multiplies, producing a result that is checked against serial execution. `schedule(runtime)` uses the selected policy. Each worker records task count, total iterations assigned, and loop time before the final barrier.

**What I Verified:** Independent Python oracles and distribution checks passed 126 cases, covering every policy, 1/4/8 workers, chunks larger than workloads, and uniform/tail-heavy input. Six invalid configurations were rejected. All 60 OpenMP benchmark records also passed the built-in serial check.

## Technical Highlights

- Keep task results observable and validate them, avoiding an optimizable discarded-work benchmark.
- Distinguish number of tasks from iteration cost when examining load balance.
- Exclude final barrier waiting from per-worker active-loop measurements.
- Treat auto as an implementation choice, with no user chunk-size claim.

## Build & Run

From the repository root, with a compiler/runtime supporting OpenMP:

```sh
mkdir -p build
clang++ -O3 -std=c++11 -Wall -Wextra -Wpedantic -fopenmp openmp-scheduling/src/main.cpp -o build/scheduling
./build/scheduling static 4 10 1000 100 10000 uniform
./build/scheduling dynamic 4 10 1000 100 10000 tail
./build/scheduling guided 4 1 1000 100 10000 tail
./build/scheduling auto 4 0 1000 100 10000 tail
python3 openmp-scheduling/tests/test_cli.py build/scheduling
```

Arguments: `policy threads chunk items light_iterations heavy_iterations uniform|tail`. Auto requires `chunk=0`, which denotes no user chunk specification. Other policies require a positive chunk. Uniform uses `light_iterations` throughout; tail assigns `heavy_iterations` to the last `max(1, items/10)` tasks. Every numeric field is capped at 10,000,000 and worker count at 256; choose workloads appropriate for your machine. Actual runtime team size is recorded.

## Policy Decisions

Static has little per-task coordination, but large chunks can concentrate heavy tail tasks. Dynamic distributes chunks as workers become available and adds acquisition overhead. Guided reduces chunk sizes as work remains; heavy tasks can still cluster inside a large chunk. Auto behavior depends on the runtime. The committed small experiment does not establish a universal winner. [Load balancing](../docs/load-balancing.md) explains these tradeoffs.

## Experimental Results

[Actual raw records](../benchmarks/results/macos-arm64-smoke-2026-10-31-v3/README.md) cover uniform/tail layouts at 4 requested workers and chunks 1/10/100 (auto once per layout). In the tail layout, dynamic chunk=1 had a 0.121 ms median and chunk=100 had 0.304 ms. The two layouts are distinct workloads; these exploratory measurements are affected by small-workload and coordination costs and do not rank policies generally. See the [CPU Benchmark Guide](../benchmarks/README.md) for separate workload charts and artifact provenance.

## Limitations and Future Work

Per-worker loop durations include scheduler overhead and OS interruptions; they are not pure CPU compute time. Heterogeneous CPU placement was uncontrolled. Add larger workloads, explicit affinity records, multiple independent runs, and a serial throughput comparison before making performance claims. The course-supplied `explore_region` and initialization code were not copied.
