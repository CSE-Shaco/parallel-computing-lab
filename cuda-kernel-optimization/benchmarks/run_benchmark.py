#!/usr/bin/env python3
"""Run real GPU binaries; fail closed when execution or correctness fails."""
import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_results import KERNELS, load_document, summarize, validate_run


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/kernel_benchmark"))
    parser.add_argument("--shape", action="append", default=[], help="samples,features; repeatable")
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--build-options", required=True, help="Record exact configure/build command and options")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 1000000:
        parser.error("repeats must be in [1,1000000]")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new path to preserve evidence")
    dimensions = []
    for shape in args.shape or ["257,65", "1024,128", "4096,256"]:
        values = shape.split(",")
        if len(values) != 2 or any(not v.isdecimal() or not 1 <= int(v) <= 1000000 for v in values):
            parser.error("shape must contain two integers in [1,1000000]")
        pair = tuple(int(value) for value in values)
        if pair in dimensions:
            parser.error("duplicate shapes are not allowed")
        dimensions.append(pair)
    binary_digest = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    runs = []
    for samples, features in dimensions:
        command = [str(args.binary.resolve()), str(samples), str(features), str(args.repeats)]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        run = load_document(completed.stdout)
        validate_run(run, repeats=args.repeats)
        if (run["samples"], run["features"]) != (samples, features):
            raise ValueError("binary returned different workload dimensions")
        run["command"] = command
        runs.append(run)
    result = {"captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "host": platform.platform(), "build_options": args.build_options,
              "binary_sha256": binary_digest, "runs": runs}
    summarize(result)  # Reject mixed schemas, environments or policies before recording success.
    if hashlib.sha256(args.binary.read_bytes()).hexdigest() != binary_digest:
        raise ValueError("benchmark executable changed during measurement")
    if runs[0].get("schema_version", 1) == 2:
        result["benchmark_schema_version"] = 2
        result["kernel_names"] = list(KERNELS)
        result["measurement_policy"] = {"warmup": 10, "kernel_order": "rotating"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
