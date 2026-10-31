"""Shared record contract for the CPU benchmark runner and analyst.

Validation establishes internal consistency, not timing authenticity or a hardware
performance claim. The legacy 2026-10-31 run is supported without modifying it.
"""
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

THREADS = (1, 2, 4, 8, 16, 32)
POLICIES = ("static", "dynamic", "guided", "auto")
CHUNKS = (1, 10, 100)


def integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is not an integer in the permitted range")
    return value


def number(value, name, positive=True):
    if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _pairs(values):
    out = {}
    for key, value in values:
        if key in out:
            raise ValueError(f"duplicate JSON field: {key}")
        out[key] = value
    return out


def parse_json(text):
    def invalid(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid)


def jobs(args, include_openmp=True):
    seed = args.get("seed", 42)
    result = [("pi_mc", ["serial", "1", str(args["samples"]), str(seed)])]
    result += [("pi_mc", [mode, str(t), str(args["samples"]), str(seed)])
               for mode in ("local", "mutex") for t in THREADS]
    if include_openmp:
        result += [("scheduling", [policy, "4", str(chunk), str(args["items"]),
                                    str(args["light"]), str(args["heavy"]), layout])
                   for layout in ("uniform", "tail") for policy in POLICIES
                   for chunk in ((0,) if policy == "auto" else CHUNKS)]
    return result


def command_key(experiment, command):
    # Keys include every input dimension, even though legacy display names do not.
    # MC: experiment, mode, workers, samples, seed.
    # OMP: experiment, layout, policy, workers, chunk, items, light, heavy.
    if experiment == "pi_mc":
        mode, threads, samples, seed = command
        return (experiment, mode, int(threads), int(samples), int(seed))
    policy, threads, chunk, items, light, heavy, layout = command
    return (experiment, layout, policy, int(threads), int(chunk), int(items), int(light), int(heavy))


def row_key(row):
    experiment = row.get("experiment")
    if experiment == "pi_mc":
        mode = row["mode"]
        if mode not in ("serial", "local", "mutex"):
            raise ValueError("unknown Monte Carlo mode")
        threads = integer(row["threads"], "threads", 1, 256)
        if mode == "serial" and threads != 1:
            raise ValueError("serial requires one worker")
        return (experiment, mode, threads, integer(row["samples"], "samples", 1, 10**12),
                integer(row["seed"], "seed", 0, 2**64 - 1))
    if experiment == "scheduling":
        policy, layout = row["policy"], row["layout"]
        if policy not in POLICIES or layout not in ("uniform", "tail"):
            raise ValueError("unknown scheduling policy/layout")
        threads = integer(row["threads"], "threads", 1, 256)
        chunk = integer(row["chunk"], "chunk", 0, 10**7)
        if (policy == "auto" and chunk != 0) or (policy != "auto" and chunk == 0):
            raise ValueError("auto uses chunk=0; other policies require a positive chunk")
        return (experiment, layout, policy, threads, chunk,
                integer(row["items"], "items", 1, 10**7),
                integer(row["light"], "light", 1, 10**7),
                integer(row["heavy"], "heavy", 1, 10**7))
    raise ValueError("unknown experiment (only CPU Monte Carlo and OpenMP are accepted)")


def validate_row(row):
    if not isinstance(row, dict):
        raise ValueError("record must be an object")
    key = row_key(row)
    number(row["ms"], "ms")
    integer(row["repeat"], "repeat")
    if key[0] == "pi_mc":
        count = integer(row["count"], "count", 0, row["samples"])
        pi = number(row["pi"], "pi", positive=False)
        if not math.isclose(pi, 4.0 * count / row["samples"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("pi does not agree with count/samples")
    else:
        if row.get("verified") is not True:
            raise ValueError("OpenMP record did not confirm result validation")
        integer(row["checksum"], "checksum", 0, 2**64 - 1)
        actual = integer(row["actual_threads"], "actual_threads", 1, row["threads"])
        workers = row["workers"]
        if not isinstance(workers, list) or len(workers) != actual:
            raise ValueError("worker records do not agree with actual team size")
        assigned, iterations = 0, 0
        for worker in workers:
            count = integer(worker["items"], "worker.items", 0, row["items"])
            work = integer(worker["iterations"], "worker.iterations")
            number(worker["active_ms"], "worker.active_ms", positive=False)
            if row["layout"] == "uniform" or row["heavy"] == row["light"]:
                if work != count * row["light"]:
                    raise ValueError("worker iteration count does not match uniform cost")
            else:
                difference = row["heavy"] - row["light"]
                extra = work - count * row["light"]
                if extra % difference or not 0 <= extra // difference <= count:
                    raise ValueError("worker iteration count cannot represent assigned task costs")
            assigned += count
            iterations += work
        tail = max(1, row["items"] // 10) if row["layout"] == "tail" else 0
        expected = (row["items"] - tail) * row["light"] + tail * row["heavy"]
        if assigned != row["items"] or iterations != expected:
            raise ValueError("worker totals do not cover the declared workload exactly")
    return key


def group_name(key):
    if key[0] == "pi_mc":
        return f"pi/{key[1]}/{key[2]} threads"
    return f"omp/{key[1]}/{key[2]}/chunk {key[4]}"


def summarize(groups):
    result = {}
    for key, rows in sorted(groups.items(), key=lambda item: group_name(item[0])):
        times = [row["ms"] for row in rows]
        result[group_name(key)] = {"runs": len(times), "median_ms": statistics.median(times),
                                   "min_ms": min(times), "max_ms": max(times)}
    return result


def load_run(path):
    path = Path(path)
    meta = parse_json((path / "metadata.json").read_text())
    if not isinstance(meta, dict) or type(meta.get("schema_version", 1)) is not int or meta.get("schema_version", 1) not in (1, 2):
        raise ValueError("unsupported metadata schema")
    if meta.get("schema_version") == 2 and any(name not in meta for name in ("status", "configuration_count", "completed_records")):
        raise ValueError("schema 2 requires completion status and record/configuration counts")
    if meta.get("status", "complete") != "complete":
        raise ValueError("run is incomplete or failed; analysis requires a complete run")
    args = meta["argv"]
    integer(args["samples"], "metadata.samples", 1, 10**12)
    integer(args["repeats"], "metadata.repeats", 1)
    integer(args.get("seed", 42), "metadata.seed", 0, 2**64 - 1)
    for name in ("items", "light", "heavy"):
        integer(args[name], "metadata." + name, 1, 10**7)
    missing = meta.get("missing", [])
    if not isinstance(missing, list) or any(item != "scheduling" for item in missing) or len(set(missing)) != len(missing):
        raise ValueError("only the optional OpenMP executable may be missing")
    expected = {command_key(experiment, command) for experiment, command in jobs(args, "scheduling" not in missing)}
    for binary in ("pi_mc",) if "scheduling" in missing else ("pi_mc", "scheduling"):
        if not re.fullmatch(r"[a-fA-F0-9]{64}", meta["binary_sha256"][binary]):
            raise ValueError("missing/invalid recorded executable SHA-256")
    groups, seen, counts, checksums, teams = {}, set(), set(), {}, {}
    rows = []
    for line_number, line in enumerate((path / "raw.jsonl").read_text().splitlines(), 1):
        if not line:
            raise ValueError(f"blank raw record at line {line_number}")
        try:
            row = parse_json(line)
            key = validate_row(row)
            if key not in expected:
                raise ValueError("configuration disagrees with metadata/declared matrix")
            if row["repeat"] >= args["repeats"] or (key, row["repeat"]) in seen:
                raise ValueError("duplicate or out-of-range configuration/repeat")
            seen.add((key, row["repeat"]))
            groups.setdefault(key, []).append(row)
            rows.append(row)
            if key[0] == "pi_mc":
                counts.add(row["count"])
            else:
                workload = (row["layout"], row["items"], row["light"], row["heavy"])
                checksums.setdefault(workload, set()).add(row["checksum"])
                teams.setdefault(key, set()).add(row["actual_threads"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"raw line {line_number}: {error}") from error
    if set(groups) != expected or len(rows) != len(expected) * args["repeats"]:
        raise ValueError("incomplete configuration/repetition matrix")
    if len(counts) != 1 or any(len(values) != 1 for values in checksums.values()):
        raise ValueError("results differ between equivalent workloads")
    if any(len(values) != 1 for values in teams.values()):
        raise ValueError("actual OpenMP team size changed within a configuration")
    if len({row["actual_threads"] for row in rows if row["experiment"] == "scheduling"}) > 1:
        raise ValueError("actual team size differs across scheduling configurations; results are not comparable")
    if "configuration_count" in meta:
        integer(meta["configuration_count"], "configuration_count", 1)
        if meta["configuration_count"] != len(expected):
            raise ValueError("configuration count disagrees with metadata")
    if "completed_records" in meta:
        integer(meta["completed_records"], "completed_records")
        if meta["completed_records"] != len(rows):
            raise ValueError("completed record count disagrees with raw data")
    summary = summarize(groups)
    legacy = path / "summary.json"
    if legacy.exists() and parse_json(legacy.read_text()) != summary:
        raise ValueError("recorded summary does not agree with raw records")
    return meta, rows, groups, summary


def source_hashes(path):
    result = {}
    for name in ("metadata.json", "raw.jsonl", "summary.json", "medians.svg"):
        p = Path(path) / name
        if p.is_file():
            result[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result
