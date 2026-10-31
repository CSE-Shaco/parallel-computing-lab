#!/usr/bin/env python3
"""Build and run CPU-side checks without CMake or CUDA; never benchmark."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default=os.environ.get("CXX", "c++"),
                        help="C++ compiler executable (use LLVM clang++ for OpenMP on macOS)")
    parser.add_argument("--build-dir", type=Path,
                        help="new directory for generated files; default: a fresh temporary directory")
    parser.add_argument("--only", choices=["all", "threads", "openmp", "cuda-host", "benchmarks"], default="all")
    parser.add_argument("--require-openmp", action="store_true",
                        help="fail instead of skipping when the compiler lacks OpenMP")
    args = parser.parse_args()
    compiler = shutil.which(args.compiler)
    if compiler is None and args.only != "benchmarks":
        parser.error("compiler executable not found")
    root = Path(__file__).resolve().parents[1]
    if args.build_dir is None:
        build = Path(tempfile.mkdtemp(prefix="parallel-computing-lab-check-"))
    else:
        build = args.build_dir.resolve()
        if build.exists():
            parser.error("build directory already exists; choose a new path to preserve prior outputs")
        build.mkdir(parents=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    results = {}
    commands = []

    def run(command):
        command = [str(value) for value in command]
        commands.append(command)
        print("RUN " + " ".join(command), flush=True)
        subprocess.run(command, cwd=root, env=env, check=True)

    warnings = ["-O3", "-Wall", "-Wextra", "-Wpedantic"]
    if args.only in ["all", "benchmarks"]:
        run([sys.executable, "-m", "unittest", "discover", "-s", root / "benchmarks/tests", "-p", "test_*.py"])
        results["benchmark_tools"] = "PASS: schema, incomplete/corrupt records, preservation and runner failure regressions"
    if args.only in ["all", "threads"]:
        executable = build / "pi_mc"
        run([compiler, "-std=c++11", *warnings, "-pthread",
             root / "cpp-multithreading/src/main.cpp", "-o", executable])
        run([sys.executable, root / "cpp-multithreading/tests/test_cli.py", executable])
        results["threads"] = "PASS: 52 exact comparisons and 8 invalid inputs"
    if args.only in ["all", "openmp"]:
        probe_source = build / "openmp_probe.cpp"
        probe_source.write_text("#include <omp.h>\nint main() { return omp_get_max_threads() < 1; }\n")
        probe = build / "openmp_probe"
        probe_command = [compiler, "-std=c++11", "-fopenmp", str(probe_source), "-o", str(probe)]
        commands.append(probe_command)
        compiled = subprocess.run(probe_command, cwd=root, env=env, capture_output=True, text=True)
        if compiled.returncode != 0:
            results["openmp"] = "SKIPPED: compiler/runtime probe did not compile with -fopenmp"
            print(results["openmp"], flush=True)
            if compiled.stderr:
                print(compiled.stderr, file=sys.stderr)
            if args.require_openmp:
                raise RuntimeError("OpenMP is required but unavailable")
        else:
            run([probe])
            executable = build / "scheduling"
            run([compiler, "-std=c++11", *warnings, "-fopenmp",
                 root / "openmp-scheduling/src/main.cpp", "-o", executable])
            run([sys.executable, root / "openmp-scheduling/tests/test_cli.py", executable])
            results["openmp"] = "PASS: 126 reference/distribution comparisons and 6 invalid inputs"
    if args.only in ["all", "cuda-host"]:
        cuda = root / "cuda-kernel-optimization"
        executable = build / "reference_test"
        run([compiler, "-std=c++17", *warnings, "-Werror", "-I", cuda / "include",
             cuda / "src/reference/reference.cpp", cuda / "tests/reference_test.cpp", "-o", executable])
        run([executable])
        run([sys.executable, "-m", "unittest", "discover", "-s", cuda / "tests", "-p", "test_*.py"])
        results["cuda_host"] = "PASS: CPU reference checks and 6 analysis tests"
    print(json.dumps({"compiler": compiler, "build_directory": str(build),
                      "results": results, "commands": commands,
                      "gpu": "NOT RUN: this direct runner covers CPU-side checks only"}, indent=2))


if __name__ == "__main__":
    main()
