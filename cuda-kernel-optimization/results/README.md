# Recorded GPU results

- [Five repeated four-kernel RTX 4060 sweeps](rtx4060-repeat-20261031T095734Z-analysis/README.md): 3,200 samples across five consecutive validated sweeps on the same host/day. Analysis summarizes five within-sweep paired ratios, not pooled launches. Register tiling at 1024×512 has median ratio 1.406×, range 1.390–1.406×; coalesced tiling loses on every measured shape/sweep.

- [Run 03: four-kernel RTX 4060 / WSL2 / CUDA 13.3](rtx4060-wsl-run-03/README.md): source snapshot `33c5536`, 12 GPU correctness shapes for all four variants, host reference/layout checks, memcheck/racecheck, and 640 timing samples (4 shapes × 4 kernels × 40 launches). The register-tiled variant reached 1.405× baseline/candidate median at 1024×512; the coalesced-load variant was slower than baseline on every shape.
- [Historical run 02: two-kernel RTX 4060 / WSL2 / CUDA 13.3](rtx4060-wsl-run-02/README.md): successful eight-shape baseline/tiled correctness, memcheck/racecheck and 180 timing samples (3 shapes × 2 kernels × 30 launches). Its source snapshot and measurement policy are separate from run 03.

The retained artifacts include raw samples, metadata, summaries, logs and comparison figures. Generated `build/` executables stay local; their hashes remain in `execution.json`. Do not compare a new candidate against a baseline from another run. Same-session launch repetitions do not establish repeatability across independent sessions, and kernel-only ratios are not end-to-end or CPU/GPU speedups.

CPU-only CMake/CTest has passed separately; CUDA-enabled CMake/GPU CTest is not verified by these direct-`nvcc` sessions. No GPU execution was repeated by the macOS record reviewer. For an independent follow-up, use [the execution workflow](../docs/gpu-execution.md), choose a new output directory, and review artifacts before adding them to Git.
