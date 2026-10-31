# Recorded GPU results

- [RTX 4060 / WSL2 / CUDA 13.3](rtx4060-wsl-run-02/README.md): successful correctness, memcheck/racecheck and three-shape timing run, 30 samples per kernel per shape.

The retained artifacts include raw samples, metadata, summaries, logs and a regenerated comparison figure. Generated `build/` executables stay local; their hashes remain in `execution.json`. The separately audited records establish what this session measured, not a general GPU speedup. CMake/CTest and an independent repeated GPU session remain unverified.

For a new run, use [the GPU execution workflow](../docs/gpu-execution.md). Choose a new output directory and review artifacts before explicitly adding them to Git.
