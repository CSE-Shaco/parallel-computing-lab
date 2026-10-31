# Running the four-kernel GPU validation and benchmark

Run this workflow in Ubuntu on WSL2 with the RTX 4060. The repository includes a direct `nvcc` path, so CMake is optional for this run. Use the Windows NVIDIA driver inside WSL; do not install a Linux NVIDIA driver in the distribution. NVIDIA's [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/) explains the supported driver/toolkit split.

## Prepare the updated checkout

From Ubuntu, change to the existing `parallel-computing-lab` checkout and update it:

```sh
cd ~/parallel-computing-lab
git pull --ff-only origin main
```

If that is a new WSL distribution, install/configure GitHub CLI authentication there so it can read the private repository. Open the CUDA subproject:

```sh
cd cuda-kernel-optimization
```

## Check the target and tools

```sh
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
nvcc --version
compute-sanitizer --version
python3 benchmarks/run_gpu_validation.py --preflight-only
```

Confirm the device is a GeForce RTX 4060, compute capability 8.9, and that the toolkit exposes `nvcc` and Compute Sanitizer. Use `sm_89` for this card. If any prerequisite fails, resolve that first; a preflight pass only checks tool/device inventory, not kernel correctness.

## Run a fresh four-way sweep

[Run 03](../results/rtx4060-wsl-run-03/README.md) is now a retained successful four-kernel session. The command below uses `run-04` so it does not overwrite that evidence.

From `cuda-kernel-optimization/`, run:

```sh
python3 benchmarks/run_gpu_validation.py \
  --arch sm_89 \
  --out results/rtx4060-wsl-run-04 \
  --repeats 40 \
  --shape 257,65 \
  --shape 1024,128 \
  --shape 4096,256 \
  --shape 1024,512
```

This rebuilds and validates the independent CPU reference and the host tile-layout oracle, compiles four GPU kernels, checks all 12 correctness shapes on each, then runs memcheck and racecheck. Only when those gates pass does it collect 10 warmups plus 40 measured launches per kernel/shape. The order rotates across repetitions. The original shapes are measured again alongside the additional 1024×512 shape, so the four candidates share one fresh build and session.

The CPU reference cost scales as `O(samples × features²)`. The suggested shapes keep the reference work moderate while testing partial tiles, both existing cases and a larger output matrix. If you change shapes, record the command and keep all four kernels on the same set. The runner refuses to reuse an existing output directory; choose another unused run number for each subsequent sweep.

If compilation, correctness, Compute Sanitizer or timing fails, the run is recorded as failed and retains its logs. Do not use a failed result as a benchmark. In WSL, a “Failed to initialize WDDM debugger interface” message means the Windows GPU debug interface needs to be enabled for Compute Sanitizer; it does not by itself indicate a kernel memory bug. NVIDIA lists the required Windows registry interface under [Compute Sanitizer's Windows behavior](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#windows-specific-behavior). Do not count sanitizer coverage as passed until both tools complete successfully.

## Review and publish the run

Successful output contains:

- `execution.json`: environment, source/binary hashes, compile and run commands, validation status.
- `benchmark.json` and `raw/`: per-kernel timings and per-shape raw samples.
- `summary.json` and `kernel-time.svg`: medians, min–max spread, counts, and baseline-relative ratios.
- `logs/`: build, correctness, sanitizer, benchmark and analyzer output.
- `build/`: local executables used for the run; this directory is ignored by Git.

Inspect that `execution.json` has `status: complete`, all four kernel names appear for every requested shape, the schema is version 2, and sanitizer reports no errors or warnings. Then review exactly what Git will stage:

```sh
git status --short
git add -- results/rtx4060-wsl-run-04/execution.json \
  results/rtx4060-wsl-run-04/benchmark.json \
  results/rtx4060-wsl-run-04/summary.json \
  results/rtx4060-wsl-run-04/kernel-time.svg \
  results/rtx4060-wsl-run-04/raw \
  results/rtx4060-wsl-run-04/logs
git diff --cached --stat
```

Verify that only this successful run's evidence is staged and that `build/` executables are absent. Then commit and push the run. Keep the historical [run 02](../results/rtx4060-wsl-run-02/README.md) unchanged: it measures only the original baseline/tiled pair at source snapshot `d5a7070`.

## Measurement scope

The kernels compute the uncentered second-moment matrix `XᵀX/m`; CPU reference work and host/device transfers are outside the timed region. The four-kernel analyzer reports baseline median divided by each candidate median, so a ratio above 1 means the candidate's median was lower in that sweep. These are kernel-only measurements, not end-to-end application speedups.

The runner does not lock clocks, thermal state, power policy or background load. Forty repeated launches reduce dependence on a single sample but remain one device session. Use independent sweeps and a profiler to support broader conclusions. The previous run 02 is historical evidence for two kernels only. Run 03 validates all four variants and records both positive and negative results; it does not establish that either new implementation will win on another session, input or GPU. Its source hashes match the exact LF source snapshot `33c5536`; run 02 used Windows CRLF checkout bytes at its older snapshot.
