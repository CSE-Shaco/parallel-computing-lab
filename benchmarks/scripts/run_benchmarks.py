"""Collect CPU timing evidence in a new directory; preserve failed runs too."""
import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from benchmark_data import command_key, jobs, load_run, validate_row


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_model():
    """Capture the current host only; never backfill an older measurement."""
    try:
        if sys.platform == "darwin":
            result = subprocess.run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                    check=True, capture_output=True, text=True, timeout=5)
            return result.stdout.strip() or None
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", type=Path, default=Path("build"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=1000000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--items", type=int, default=1000)
    ap.add_argument("--light", type=int, default=100)
    ap.add_argument("--heavy", type=int, default=10000)
    ap.add_argument("--compiler", default="unspecified", help="operator-declared build compiler/version")
    ap.add_argument("--flags", default="unspecified", help="operator-declared build flags")
    ap.add_argument("--compiler-executable", help="optional executable to probe now; not proof of binary build provenance")
    ap.add_argument("--timeout", type=float, default=30, help="seconds allowed for each executable invocation")
    args = ap.parse_args()
    if not (0 < args.samples <= 10**12) or min(args.repeats, args.items, args.light, args.heavy) <= 0:
        ap.error("sizes/repeats must be positive; samples must not exceed 10^12")
    if max(args.items, args.light, args.heavy) > 10**7:
        ap.error("items/light/heavy must not exceed 10000000")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        ap.error("timeout must be finite and positive")
    if args.out.exists():
        ap.error("output directory exists; choose a new path to preserve previous runs")
    binaries = {name: (args.build / name).resolve() for name in ("pi_mc", "scheduling")}
    if not binaries["pi_mc"].is_file() or not os.access(binaries["pi_mc"], os.X_OK):
        ap.error("pi_mc executable missing or not executable")
    if binaries["scheduling"].exists() and (not binaries["scheduling"].is_file() or not os.access(binaries["scheduling"], os.X_OK)):
        ap.error("scheduling exists but is not an executable file")
    compiler_probe = None
    if args.compiler_executable:
        executable = shutil.which(args.compiler_executable)
        if executable is None:
            ap.error("compiler-executable not found")
        try:
            completed = subprocess.run([executable, "--version"], check=True, capture_output=True,
                                       text=True, timeout=args.timeout)
            compiler_probe = {"executable": executable, "version_output": completed.stdout.strip(),
                              "scope": "Operator-selected executable probed now; not evidence that it built these binaries."}
        except (OSError, subprocess.SubprocessError) as error:
            ap.error(f"compiler probe failed: {error}")
    missing = [name for name, path in binaries.items() if not path.is_file()]
    hashes = {name: digest(path) for name, path in binaries.items() if path.is_file()}
    parameters = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    parameters["seed"] = 42
    matrix = jobs(parameters, "scheduling" not in missing)
    metadata = {
        "schema_version": 2, "status": "running",
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "logical_cpus": os.cpu_count(), "processor": platform.processor(),
        "cpu_model_captured_now": cpu_model(), "python": platform.python_version(),
        "compiler": args.compiler, "flags": args.flags, "compiler_probe": compiler_probe,
        "build_provenance_scope": "compiler/flags are operator declarations; executable hashes are measured, not source/build attestation.",
        "argv": parameters, "binary_sha256": hashes,
        "omp_environment": {key: value for key, value in os.environ.items() if key.startswith("OMP_")},
        "missing": missing, "configuration_count": len(matrix), "shuffle_seed": 2026,
        "warmup_processes_per_configuration": 1,
        "scope": "Exploratory local run; no CPU affinity, clock, thermal, or background-load control.",
        "timing": "Internal steady_clock wall time; process startup and correctness reference excluded. Thread/team creation included.",
    }
    # Preflight is complete. This directory belongs to this invocation exclusively.
    args.out.mkdir(parents=True, exist_ok=False)
    metadata_path = args.out / "metadata.json"

    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    save_metadata()
    raw = args.out / "raw.jsonl"
    raw.touch(exist_ok=False)
    reference_count = None
    expected_checksums = {}
    records = 0
    try:
        def execute(name, command, repeat):
            nonlocal reference_count
            completed = subprocess.run([str(binaries[name]), *command], check=True,
                                       capture_output=True, text=True, timeout=args.timeout)
            from benchmark_data import parse_json
            row = parse_json(completed.stdout)
            row.update(experiment=name, repeat=repeat)
            key = validate_row(row)
            if key != command_key(name, command):
                raise ValueError("executable output configuration differs from its command")
            if name == "pi_mc":
                if reference_count is None:
                    reference_count = row["count"]
                if row["count"] != reference_count:
                    raise ValueError("Monte Carlo count differs from serial warmup")
            else:
                workload = (row["layout"], row["items"], row["light"], row["heavy"])
                expected_checksums.setdefault(workload, row["checksum"])
                if expected_checksums[workload] != row["checksum"]:
                    raise ValueError("OpenMP checksum changed for an equivalent workload")
            return row

        for name, command in matrix:
            execute(name, command, 0)
        expanded = [(name, command, repeat) for repeat in range(args.repeats) for name, command in matrix]
        random.Random(2026).shuffle(expanded)
        with raw.open("a") as output:
            for name, command, repeat in expanded:
                row = execute(name, command, repeat)
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
                records += 1
        if any(digest(binaries[name]) != value for name, value in hashes.items()):
            raise ValueError("executable bytes changed while measuring")
        # Check the completed raw matrix using the same reader used for analysis.
        metadata["status"] = "complete"
        metadata["completed_records"] = records
        save_metadata()
        load_run(args.out)
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(status="failed", completed_records=records,
                        failure={"type": type(error).__name__, "message": str(error)})
        details = [f"{type(error).__name__}: {error}"]
        for stream in ("stdout", "stderr"):
            value = getattr(error, stream, None)
            if value:
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                details.append(f"{stream} (last 8192 characters):\n{value[-8192:]}")
        (args.out / "failure.log").write_text("\n".join(details) + "\n")
        save_metadata()
        print(f"Benchmark failed; partial evidence retained in {args.out}: {error}", file=sys.stderr)
        return 1
    print(f"Saved {records} measured records to {args.out}; status=complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
