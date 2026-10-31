# Work Distribution and Load Balancing

The independent workload writes one output per task, checks every output after timing, and accumulates statistics in worker-local variables and writes each worker's final statistics once, avoiding hot adjacent shared counters. Uniform tasks use the same iteration count. Tail-heavy tasks increase the last tenth's iteration count (at least one task). This synthetic workload reveals scheduling decisions without copying the course-supplied numerical routine.

`static,chunk` assigns chunks cyclically ahead of time. `dynamic,chunk` hands out chunks as workers request more. `guided,chunk` decreases chunk sizes toward a lower bound. `auto` delegates policy to the compiler/runtime and has no requested chunk; the CLI requires zero to make that distinction explicit. The code uses `omp_set_schedule` and `schedule(runtime)`.

Task counts alone can hide imbalance: ten heavy tasks may cost more than many light tasks. Output includes total loop iterations per worker as a cost proxy. Active-loop time is recorded after a `nowait` worksharing loop and before the parallel-region barrier, so it does not include waiting at that final barrier. It still includes task acquisition, instructions updating statistics, and preemption; it is not a hardware utilization measurement.

The original notebook recorded durations after the implicit worksharing barrier, so workers could appear to have similar times despite different amounts of work. Its auto output was repeated under chunk labels even though auto did not use those chunks. Those outputs are not imported as evidence here.

Expect static to suit uniform work when assignment cost is low, and dynamic to help irregular work when improved distribution exceeds its scheduling cost. These are hypotheses to test, not conclusions for every workload. Large chunks can defeat dynamic balancing; a clustered tail is not representative of all irregular workloads. The committed run provides raw per-worker data for inspection.
