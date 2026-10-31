#!/usr/bin/env python3
"""Run real GPU binaries; fail closed when execution or correctness fails."""
import argparse
import datetime
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/kernel_benchmark"))
    parser.add_argument("--shape", action="append", default=[], help="samples,features; repeatable")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--build-options", required=True, help="Record exact configure/build command and options")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new path to preserve evidence")
    binary_digest = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    runs = []
    for shape in args.shape or ["257,65", "1024,128", "4096,256"]:
        values = shape.split(",")
        if len(values) != 2 or any(not v.isdecimal() or int(v) < 1 for v in values):
            parser.error("shape must contain two positive integers")
        command = [str(args.binary.resolve()), *values, str(args.repeats)]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        run = json.loads(completed.stdout)
        if run.get("validated") is not True:
            raise ValueError("binary did not confirm correctness")
        run["command"] = command
        runs.append(run)
    result = {"captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "host": platform.platform(), "build_options": args.build_options,
              "binary_sha256": binary_digest, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
