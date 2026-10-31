#!/usr/bin/env python3
"""Validate and measure on a real NVIDIA CUDA host; never emulate GPU results."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAPES = ("257,65", "1024,128", "4096,256")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths = [ROOT / "CMakeLists.txt"]
    for folder in ("include", "src", "tests", "benchmarks"):
        paths.extend(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and p.suffix in (".hpp", ".cuh", ".cpp", ".cu", ".py"))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}


def shape(value):
    parts = value.split(",")
    if len(parts) != 2 or any(not x.isdecimal() or not 1 <= int(x) <= 1000000 for x in parts):
        raise argparse.ArgumentTypeError("shape must be samples,features in [1,1000000]")
    return ",".join(str(int(x)) for x in parts)


def preflight(nvcc="nvcc"):
    paths = {"nvcc": shutil.which(nvcc), "nvidia_smi": shutil.which("nvidia-smi"),
             "compute_sanitizer": shutil.which("compute-sanitizer")}
    missing = [name for name in ("nvcc", "nvidia_smi") if not paths[name]]
    if missing:
        raise RuntimeError("NVIDIA CUDA execution unavailable: missing " + ", ".join(missing))
    probes = {}
    for name, command in (
        ("nvcc_version", [paths["nvcc"], "--version"]),
        ("gpu_inventory", [paths["nvidia_smi"],
                           "--query-gpu=index,name,driver_version,memory.total",
                           "--format=csv,noheader,nounits"]),
    ):
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
        if not result.stdout.strip():
            raise RuntimeError(name + " returned no device/toolchain information")
        probes[name] = {"command": command, "stdout": result.stdout.strip()}
    return paths, probes


def run_validation(args, tools, probes):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    build = out / "build"
    logs = out / "logs"
    build.mkdir()
    logs.mkdir()
    metadata = {"schema_version": 1, "status": "running", "started_utc": now(),
                "host": platform.platform(), "architecture": args.arch,
                "requested_shapes": args.shape or list(DEFAULT_SHAPES), "repeats": args.repeats,
                "source_sha256": sources(), "preflight": probes, "commands": [],
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "timing_scope": "Kernel-only CUDA events; CPU reference and transfers excluded.",
                "measurement_limits": "GPU clocks, thermal state, power policy and background load are not fixed by this runner.",
                "operation": "uncentered_second_moment", "sanitizer": "Not Verified: tool unavailable"}
    record = out / "execution.json"

    def save():
        record.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    def execute(label, command):
        command = [str(p) for p in command]
        entry = {"label": label, "command": command, "started_utc": now()}
        metadata["commands"].append(entry)
        save()
        try:
            result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True,
                                    text=True, timeout=args.timeout,
                                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            entry["returncode"] = result.returncode
            stdout, stderr = result.stdout, result.stderr
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            entry["returncode"] = getattr(error, "returncode", None)
            entry["error"] = str(error)
            stdout, stderr = getattr(error, "stdout", ""), getattr(error, "stderr", "")
            stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout or ""
            stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr or ""
            (logs / (label + ".stdout.log")).write_text(stdout)
            (logs / (label + ".stderr.log")).write_text(stderr)
            save()
            raise
        (logs / (label + ".stdout.log")).write_text(stdout)
        (logs / (label + ".stderr.log")).write_text(stderr)
        save()
        return stdout

    save()
    try:
        flags = ["-std=c++17", "-O3", "-lineinfo", "-arch=" + args.arch, "-I", ROOT / "include"]
        common = [ROOT / "src/reference/reference.cpp", ROOT / "src/kernels/baseline.cu",
                  ROOT / "src/kernels/tiled.cu"]
        host = build / "reference_test"
        correct = build / "correctness_test"
        benchmark = build / "kernel_benchmark"
        execute("build-host-reference", [tools["nvcc"], *flags, ROOT / "src/reference/reference.cpp",
                                         ROOT / "tests/reference_test.cpp", "-o", host])
        execute("host-reference", [host])
        execute("build-correctness", [tools["nvcc"], *flags, *common,
                                       ROOT / "tests/correctness_test.cu", "-o", correct])
        execute("gpu-correctness", [correct])
        if tools["compute_sanitizer"]:
            for mode in ("memcheck", "racecheck"):
                execute(mode, [tools["compute_sanitizer"], "--tool", mode,
                               "--error-exitcode", "99", correct])
            metadata["sanitizer"] = "PASS: memcheck and racecheck"
        execute("build-benchmark", [tools["nvcc"], *flags, *common,
                                    ROOT / "src/main.cu", "-o", benchmark])
        metadata["binary_sha256"] = {p.name: digest(p) for p in (host, correct, benchmark)}
        build_options = json.dumps([x["command"] for x in metadata["commands"] if x["label"].startswith("build-")])
        raw = out / "raw"
        raw.mkdir()
        measured = []
        metadata["completed_shapes"] = []
        for index, dimensions in enumerate(metadata["requested_shapes"]):
            destination = raw / ("shape-" + dimensions.replace(",", "x") + ".json")
            execute("benchmark-shape-" + str(index),
                    [sys.executable, ROOT / "benchmarks/run_benchmark.py", "--binary", benchmark,
                     "--repeats", str(args.repeats), "--build-options", build_options,
                     "--shape", dimensions, "--output", destination])
            piece = json.loads(destination.read_text())
            if len(piece.get("runs", [])) != 1 or piece.get("binary_sha256") != metadata["binary_sha256"][benchmark.name]:
                raise ValueError("per-shape output or executable identity is invalid")
            measured.extend(piece["runs"])
            metadata["completed_shapes"].append(dimensions)
            save()
        document = {"captured_at_utc": now(), "host": platform.platform(),
                    "build_options": build_options,
                    "binary_sha256": metadata["binary_sha256"][benchmark.name], "runs": measured}
        with (out / "benchmark.json").open("x") as stream:
            stream.write(json.dumps(document, indent=2, allow_nan=False) + "\n")
        actual_shapes = [f"{r['samples']},{r['features']}" for r in document["runs"]]
        if actual_shapes != metadata["requested_shapes"]:
            raise ValueError("benchmark output dimensions disagree with requested shapes")
        if document["binary_sha256"] != metadata["binary_sha256"][benchmark.name]:
            raise ValueError("benchmark executable changed before measurement")
        for gpu_run in document["runs"]:
            if gpu_run.get("validated") is not True or any(
                    len(k["milliseconds"]) != args.repeats for k in gpu_run["results"]):
                raise ValueError("correctness gate or requested repetition count was not satisfied")
        analyzed = execute("analysis", [sys.executable, ROOT / "benchmarks/analyze_results.py",
                                        out / "benchmark.json", "--figure", out / "kernel-time.svg"])
        summary = [json.loads(line) for line in analyzed.splitlines() if line.strip()]
        (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        if sources() != metadata["source_sha256"]:
            raise ValueError("source files changed during execution")
        if any(digest(build / name) != value for name, value in metadata["binary_sha256"].items()):
            raise ValueError("an executable changed during execution")
        metadata.update(status="complete", completed_utc=now(),
                        gpu_validation="PASS: host reference, 8 GPU correctness shapes, per-run CPU checks",
                        result_sha256={name: digest(out / name) for name in
                                       ("benchmark.json", "summary.json", "kernel-time.svg")})
        save()
        return 0
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(status="failed", failed_utc=now(),
                        failure={"type": type(error).__name__, "message": str(error)})
        save()
        print(f"GPU validation failed; evidence retained at {out}: {error}", file=sys.stderr)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="new output directory; existing destinations are refused")
    parser.add_argument("--arch", help="explicit target architecture, e.g. sm_75 only for a compatible GPU")
    parser.add_argument("--nvcc", default="nvcc", help="CUDA compiler executable")
    parser.add_argument("--shape", type=shape, action="append", default=[])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=300, help="maximum seconds per command")
    parser.add_argument("--preflight-only", action="store_true", help="probe actual NVIDIA tools/device; create no run")
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 1000000 or args.timeout < 1:
        parser.error("repeats must be in [1,1000000]; timeout must be positive")
    if args.shape and len(set(args.shape)) != len(args.shape):
        parser.error("duplicate shapes are not allowed")
    if not args.preflight_only:
        if args.out is None or not args.arch or not re.fullmatch(r"sm_[0-9]{2,3}", args.arch):
            parser.error("execution requires --out and an explicit --arch sm_NN")
        if args.out.exists() or args.out.is_symlink():
            parser.error("output already exists; choose a new directory")
    try:
        tools, probes = preflight(args.nvcc)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    if args.preflight_only:
        print(json.dumps({"status": "NVIDIA tools responded", "tools": tools, "probes": probes}, indent=2))
        return 0
    return run_validation(args, tools, probes)


if __name__ == "__main__":
    raise SystemExit(main())
