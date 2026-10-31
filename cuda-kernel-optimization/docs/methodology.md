# CUDA correctness and benchmark methodology

This protocol compares four implementations of `XᵀX/m` on identical deterministic input. It checks CUDA correctness and sanitizer results before it accepts any timing output.

## Validation sequence

1. Capture the GPU, Windows driver, WSL kernel, CUDA compiler/runtime, architecture, compiler commands, and source hashes.
2. Compile and run the independent host reference and CPU tile-layout oracle.
3. Compile the four-kernel GPU correctness program for `sm_89` on an RTX 4060. Check all 12 shapes against the FP64 CPU reference using `abs_error <= 2e-4 + 2e-4*abs(reference)`.
4. Run Compute Sanitizer memcheck and racecheck against that correctness executable. A missing tool is **Not Verified**; a reported error or warning stops the benchmark stage.
5. Compile the benchmark with `-std=c++17 -O3 -lineinfo -arch=sm_89`, with no fast-math flag. Allocation, device transfers and CPU reference work remain outside the timed region.
6. Run 10 warmups per kernel. Across repetitions, rotate kernel order so each implementation starts from each position. Record the per-kernel block/output/reduction tile and outputs-per-thread metadata.
7. For each requested shape, validate all four outputs before timing and again after timing. Use CUDA events around one launch at a time, synchronizing the stop event before reading elapsed time.
8. Retain every positive finite sample. Report median, minimum, maximum and count per kernel/shape. The candidate speedup ratio is **baseline median ÷ candidate median**; values above 1 mean that candidate had a lower median time in that run.

The recorded Windows WSL run 03 uses 40 repetitions across the four variants. That sweep compares the original shapes plus 1024×512, which increases the feature/output dimension while keeping the CPU reference workload comparable to the existing largest case. Choose a new results directory for every run. GPU clock, thermal state, power limit and background load remain uncontrolled unless recorded and fixed separately.

## Results and provenance

Schema version 1 is the historical two-kernel format and remains readable. The four-kernel runner emits schema version 2 with explicit metadata for every kernel. The analyzer rejects mixed schemas, missing variants, duplicate shapes, invalid values, unvalidated samples and inconsistent repetitions. It keeps raw samples, generates summaries and a four-series SVG, and records the speedup definition in the output.

The immutable [RTX 4060 run 02](../results/rtx4060-wsl-run-02/README.md) used schema version 1 and compared only `baseline` and the original `tiled` kernel. Its source-hash match was audited against the pre-optimization source snapshot at commit `d5a7070` after accounting for Windows CRLF checkout bytes. Later kernels and changes to the benchmark executable are outside that evidence. The [schema-2 run 03](../results/rtx4060-wsl-run-03/README.md) times all four variants together at source snapshot `33c5536`; all 20 recorded source hashes match its LF bytes exactly. It passed 12 GPU shapes, host reference/layout checks and memcheck/racecheck before timing. Its four shapes × four variants × 40 samples produce 640 retained timings. Do not combine old medians with new measurements. The separately replayed analysis matches recorded summary values and SVG bytes; this validates record consistency, not an independent GPU execution.

Preserve execution JSON, per-shape raw data, logs, combined samples, summary and SVG with the result. Generated executables remain local evidence and are ignored by Git. A successful analyzer run cannot turn synthetic test fixtures into performance data, nor can host-only tests validate GPU execution.

The measured coalesced-load variant is slower than the same-run baseline on every shape, while the register-tiled variant is slower on the smallest shape and faster on the three larger ones. Report both outcomes rather than selecting only the best ratio. The best observed baseline/candidate ratio, 1.405× at 1024×512, corresponds to about 28.83% less median kernel time, not a 40.5% time reduction. No profiling evidence isolates why an implementation wins or loses; no CPU/GPU or end-to-end comparison is measured.
