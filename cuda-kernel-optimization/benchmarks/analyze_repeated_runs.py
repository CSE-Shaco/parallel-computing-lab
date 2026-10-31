#!/usr/bin/env python3
"""Audit recorded schema-2 sweeps and summarize paired per-run median ratios.

This is post-hoc analysis, not GPU execution or an authenticity attestation.
"""
import argparse
import csv
import hashlib
import html
import json
import math
import re
import statistics
from pathlib import Path

from analyze_results import KERNELS, VARIANTS, load_document, summarize, validate_run

RESULT_FILES = ("benchmark.json", "summary.json", "kernel-time.svg")
VALIDATION = "PASS: host reference and layout oracle, 12 GPU correctness shapes for four kernels, per-run CPU checks"
HASH = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_file(directory, relative):
    path = directory / relative
    require(path.is_file() and not path.is_symlink(), "missing or symlinked input: " + relative)
    require(directory in path.resolve().parents, "input escapes run directory: " + relative)
    return path


def read_json(directory, relative):
    return load_document(checked_file(directory, relative).read_text())


def hash_map(value, label):
    require(isinstance(value, dict) and bool(value), "missing " + label)
    require(all(isinstance(k, str) and isinstance(v, str) and HASH.fullmatch(v)
                for k, v in value.items()), "invalid " + label)
    return value


def normalized_builds(execution):
    """Compare build inputs/options while allowing each fresh output directory."""
    commands = []
    for entry in execution["commands"]:
        if not entry["label"].startswith("build-"):
            continue
        args = entry["command"]
        require(isinstance(args, list) and all(isinstance(x, str) for x in args), "invalid build argv")
        require(args and Path(args[0]).name == "nvcc", "expected direct nvcc build")
        required = ("-std=c++17", "-O3", "-lineinfo", "-arch=" + execution["architecture"])
        require(all(option in args for option in required), "missing required compiler flag")
        require(args.count("-I") == args.count("-o") == 1, "unexpected build include/output arguments")
        require(args.index("-I") + 1 < len(args) and args.index("-o") + 1 < len(args), "missing build argument value")
        include = args[args.index("-I") + 1]
        require(include.endswith("/include"), "unexpected include directory")
        module = include[:-len("/include")]
        normalized = []
        index = 0
        while index < len(args):
            token = args[index]
            if token == "-o":
                require(index + 1 < len(args), "missing output path")
                normalized.extend(("-o", "<BUILD>/" + Path(args[index + 1]).name))
                index += 2
                continue
            if index == 0:
                token = "nvcc"
            elif token.startswith(module + "/"):
                token = "<MODULE>/" + token[len(module) + 1:]
            normalized.append(token)
            index += 1
        commands.append({"label": entry["label"], "argv": normalized})
    return commands


def load_session(directory):
    directory = Path(directory).resolve()
    require(directory.is_dir(), "run directory missing")
    execution = read_json(directory, "execution.json")
    require(type(execution.get("schema_version")) is int and execution["schema_version"] == 2
            and type(execution.get("benchmark_schema_version")) is int
            and execution["benchmark_schema_version"] == 2, "expected schema-2 execution")
    require(execution.get("status") == "complete", "run is not complete")
    require(execution.get("sanitizer") == "PASS: memcheck and racecheck"
            and execution.get("gpu_validation") == VALIDATION, "validation gates not passed")
    require(type(execution.get("correctness_shapes")) is int and execution["correctness_shapes"] == 12,
            "unexpected correctness coverage")
    repeats = execution.get("repeats")
    require(type(repeats) is int and repeats > 0, "invalid repeat count")
    configuration = execution.get("variant_configuration")
    require(isinstance(configuration, dict), "invalid variant configuration")
    for name, fields in VARIANTS.items():
        require(isinstance(configuration.get(name), dict), "missing variant configuration")
        for key, expected in fields.items():
            actual = configuration[name].get(key)
            if isinstance(expected, list):
                require(isinstance(actual, list) and all(type(x) is int for x in actual), "invalid integer configuration")
            else:
                require(type(actual) is int, "invalid integer configuration")
    require(execution.get("kernel_names") == list(KERNELS)
            and execution.get("variant_configuration") == VARIANTS
            and execution.get("measurement_policy") == {"warmup": 10, "kernel_order": "rotating"},
            "unexpected kernel configuration/policy")
    source_hashes = hash_map(execution.get("source_sha256"), "source hashes")
    binary_hashes = hash_map(execution.get("binary_sha256"), "binary hashes")
    require(set(binary_hashes) == {"reference_test", "tile_layout_test", "correctness_test", "kernel_benchmark"},
            "unexpected binary identity set")
    result_hashes = hash_map(execution.get("result_sha256"), "result hashes")
    require(set(result_hashes) == set(RESULT_FILES), "unexpected result hash set")
    for name in RESULT_FILES:
        require(digest(checked_file(directory, name)) == result_hashes[name], "result hash mismatch: " + name)
    document = read_json(directory, "benchmark.json")
    rows = summarize(document)
    require(all(row.get("schema_version") == 2 for row in rows), "schema-1 samples not accepted")
    require(rows == read_json(directory, "summary.json"), "summary disagrees with raw samples")
    require(document.get("binary_sha256") == binary_hashes["kernel_benchmark"], "binary identity mismatch")
    shapes = []
    for run in document["runs"]:
        validate_run(run, required_schema=2, repeats=repeats)
        shape = f"{run['samples']},{run['features']}"
        shapes.append(shape)
        raw = read_json(directory, f"raw/shape-{run['samples']}x{run['features']}.json")
        require(raw.get("runs") == [run] and raw.get("binary_sha256") == document["binary_sha256"],
                "per-shape raw record disagrees with combined document")
    require(shapes == execution.get("requested_shapes") == execution.get("completed_shapes"),
            "requested/completed shape matrix differs")
    labels = ["build-host-reference", "host-reference", "build-host-layout", "host-layout",
              "build-correctness", "gpu-correctness", "memcheck", "racecheck", "build-benchmark"]
    labels += ["benchmark-shape-" + str(i) for i in range(len(shapes))] + ["analysis"]
    commands = execution.get("commands")
    require(isinstance(commands, list) and all(isinstance(item, dict) for item in commands)
            and [item.get("label") for item in commands] == labels,
            "command validation sequence differs")
    require(all(type(item.get("returncode")) is int and item["returncode"] == 0 for item in commands),
            "nonzero or absent command return code")
    inputs = ["execution.json", *RESULT_FILES]
    inputs += [f"raw/shape-{run['samples']}x{run['features']}.json" for run in document["runs"]]
    for label in labels:
        for stream in ("stdout", "stderr"):
            inputs.append(f"logs/{label}.{stream}.log")
            checked_file(directory, inputs[-1])
    checks = {
        "host-reference": "CPU reference, validation rejection and input checks passed",
        "host-layout": "50 host layout/oracle cases passed (not GPU execution)",
        "gpu-correctness": "12 shapes passed for baseline, tiled, coalesced_tiled and register_tiled kernels",
        "memcheck": "ERROR SUMMARY: 0 errors",
        "racecheck": "0 hazards displayed (0 errors, 0 warnings)",
    }
    for label, needle in checks.items():
        stdout = checked_file(directory, f"logs/{label}.stdout.log").read_text()
        require(needle in stdout, "validation log does not confirm success: " + label)
        require(not checked_file(directory, f"logs/{label}.stderr.log").read_text().strip(),
                "validation stderr is not empty: " + label)
        require(not re.search(r"(?:Error:|Warning:|Internal Sanitizer Error)", stdout, re.IGNORECASE),
                "validation log contains diagnostic: " + label)
    probe = execution.get("preflight", {})
    require(set(probe) == {"nvcc_version", "gpu_inventory"}, "missing tool/device inventory")
    inventory = {name: probe[name].get("stdout") for name in probe}
    require(all(isinstance(v, str) and v.strip() for v in inventory.values()), "empty tool/device inventory")
    first = document["runs"][0]
    environment = {key: first[key] for key in ("gpu", "compute_capability", "runtime_version",
                                               "driver_version", "cuda_compiler_version")}
    common = {key: execution.get(key) for key in ("host", "architecture", "CUDA_VISIBLE_DEVICES")}
    require(isinstance(common["host"], str) and common["host"]
            and isinstance(common["architecture"], str) and re.fullmatch(r"sm_[0-9]+", common["architecture"]),
            "invalid host/architecture metadata")
    common.update(environment=environment, inventory=inventory, source_sha256=source_hashes,
                  repeats=repeats, shapes=shapes, kernel_names=list(KERNELS),
                  variant_configuration=VARIANTS, measurement_policy={"warmup": 10, "kernel_order": "rotating"},
                  normalized_build_commands=normalized_builds(execution))
    return {"directory": directory, "label": directory.name, "execution": execution,
            "execution_sha256": digest(directory / "execution.json"), "common": common, "rows": rows,
            "input_sha256": {name: digest(directory / name) for name in inputs}}


def aggregate(directories):
    paths = [Path(path).resolve() for path in directories]
    require(len(paths) >= 2, "at least two recorded runs are required")
    require(len(set(paths)) == len(paths), "duplicate run directory")
    require(len({path.name for path in paths}) == len(paths), "duplicate run labels")
    sessions = [load_session(path) for path in paths]
    require(len({item["execution_sha256"] for item in sessions}) == len(sessions), "duplicate execution record")
    require(all(item["common"] == sessions[0]["common"] for item in sessions),
            "incompatible source, environment, build or measurement policy")
    per_run, rows = [], []
    for index, first in enumerate(sessions[0]["rows"]):
        row = {"shape": first["shape"], "kernels": {}}
        for name in KERNELS:
            ratios, reductions = [], []
            for session in sessions:
                measured = session["rows"][index]["kernels"]
                ratio = measured[name]["baseline_speedup"]
                reduction = 100 * (1 - measured[name]["median_ms"] / measured["baseline"]["median_ms"])
                require(math.isfinite(ratio) and ratio > 0 and math.isfinite(reduction), "non-finite computed ratio/time change")
                ratios.append(ratio); reductions.append(reduction)
                per_run.append({"run": session["label"], "shape": row["shape"], "kernel": name,
                                "sample_count": measured[name]["count"],
                                "baseline_median_ms": measured["baseline"]["median_ms"],
                                "candidate_median_ms": measured[name]["median_ms"],
                                "baseline_median_div_candidate_median": ratio,
                                "runtime_reduction_percent": reduction})
            row["kernels"][name] = {
                "session_count": len(sessions), "paired_ratios": ratios,
                "paired_ratio_median": statistics.median(ratios), "paired_ratio_min": min(ratios),
                "paired_ratio_max": max(ratios), "sessions_lower_time_than_baseline": sum(r > 1 for r in ratios),
                "runtime_reduction_percent_median": statistics.median(reductions),
                "runtime_reduction_percent_min": min(reductions), "runtime_reduction_percent_max": max(reductions)}
        rows.append(row)
    summary = {"schema_version": 1, "analysis_unit": "per-run baseline median / candidate median",
               "aggregation": "median and min-max of paired ratios; no pooled launch samples",
               "session_count": len(sessions), "run_labels": [s["label"] for s in sessions],
               "total_launch_samples": sum(item["sample_count"] for item in per_run),
               "common": sessions[0]["common"], "shapes": rows,
               "limits": ["Same-host repeated sweeps are not independent machines or thermal states",
                          "No statistical significance, confidence intervals or end-to-end speedup inferred",
                          "Recorded source/result/log consistency is not an authenticity attestation",
                          "Target binaries absent; their recorded hashes are not independently rehashed"]}
    provenance = {"analyzer_sha256": digest(Path(__file__)), "runs": [
        {"label": s["label"], "started_utc": s["execution"].get("started_utc"),
         "completed_utc": s["execution"].get("completed_utc"),
         "input_sha256": s["input_sha256"], "binary_sha256_recorded": s["execution"]["binary_sha256"]}
        for s in sessions]}
    return summary, per_run, provenance


def draw(summary, destination):
    """Show every session ratio, its range and median; baseline ratio is one."""
    shapes = summary["shapes"]
    panels = len(shapes); columns = 2
    height = 105 + math.ceil(panels / columns) * 270 + 35
    values = [r for shape in shapes for name in KERNELS[1:] for r in shape["kernels"][name]["paired_ratios"]]
    low = max(0, math.floor(min(min(values), 1.0) * .9 * 10) / 10)
    high = math.ceil(max(max(values), 1.0) * 1.1 * 10) / 10
    span = high - low
    text = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" viewBox="0 0 1120 {height}">',
            '<rect width="100%" height="100%" fill="#f5f7fa"/>',
            '<g font-family="system-ui,sans-serif" font-size="13" fill="#172b42">',
            '<text x="28" y="30" font-size="21" font-weight="700">Repeated GPU sweeps: paired median-time ratios</text>',
            '<text x="28" y="55">Baseline median / candidate median. Above 1 = lower candidate kernel time.</text>',
            f'<text x="28" y="76">{summary["session_count"]} points per row; line = range, thick mark = median. No launch-sample pooling.</text>']
    colors = ("#bb641a", "#007f71", "#7550ad")
    for i, shape in enumerate(shapes):
        px, py = 20 + (i % columns) * 550, 98 + (i // columns) * 270
        axis_start, axis_end = px + 160, px + 505
        x = lambda ratio: axis_start + (ratio - low) / span * (axis_end - axis_start)
        text.extend((f'<rect x="{px}" y="{py}" width="535" height="252" rx="10" fill="white" stroke="#dce3eb"/>',
                     f'<text x="{px+16}" y="{py+27}" font-size="17" font-weight="650">{html.escape(shape["shape"])}</text>',
                     f'<path d="M{x(1):.3f},{py+40} V{py+182}" stroke="#687e96" stroke-dasharray="4 4"/>'))
        for j, (name, color) in enumerate(zip(KERNELS[1:], colors)):
            kernel = shape["kernels"][name]; y = py + 61 + j * 49
            label = {"tiled": "Original tile", "coalesced_tiled": "Coalesced tile", "register_tiled": "Register tile"}[name]
            text.extend((f'<text x="{px+16}" y="{y+4}">{label}</text>',
                         f'<path d="M{x(kernel["paired_ratio_min"]):.3f},{y} H{x(kernel["paired_ratio_max"]):.3f}" stroke="{color}" stroke-width="2"/>'))
            for k, ratio in enumerate(kernel["paired_ratios"]):
                jitter = (k - (summary["session_count"]-1)/2) * 2.0
                title = html.escape(summary["run_labels"][k] + f": {ratio:.9f}×")
                text.append(f'<circle cx="{x(ratio):.3f}" cy="{y+jitter:.3f}" r="3.5" fill="{color}" fill-opacity=".65"><title>{title}</title></circle>')
            median = x(kernel["paired_ratio_median"])
            text.extend((f'<path d="M{median:.3f},{y-9} V{y+9}" stroke="{color}" stroke-width="3"/>',
                         f'<text x="{px+160}" y="{y+22}" font-size="11" fill="#53677e">{kernel["paired_ratio_median"]:.3f}× [{kernel["paired_ratio_min"]:.3f}, {kernel["paired_ratio_max"]:.3f}]</text>'))
        for tick in range(5):
            value = low + span * tick / 4
            text.append(f'<text x="{x(value):.3f}" y="{py+220}" text-anchor="middle" font-size="11" fill="#53677e">{value:.2f}</text>')
    text.append(f'<text x="28" y="{height-15}" font-size="12">Kernel-only observations from repeated same-host sweeps; not confidence intervals or independent-device evidence.</text>')
    text.append('</g></svg>')
    with destination.open("x") as stream:
        stream.write("\n".join(text) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("output already exists; choose a new directory")
    output = args.out.resolve()
    if any(Path(path).resolve() in output.parents for path in args.runs):
        parser.error("output must not be inside an input run; preserve recorded evidence")
    try:
        summary, per_run, provenance = aggregate(args.runs)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=False)
    for name, value in (("summary.json", summary), ("provenance.json", provenance)):
        with (args.out / name).open("x") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with (args.out / "per-run.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_run[0]))
        writer.writeheader(); writer.writerows(per_run)
    draw(summary, args.out / "stability.svg")
    print(f"Validated {summary['session_count']} runs, {summary['total_launch_samples']} launch samples; wrote {args.out}")
    return 0


if __name__ == "__main__":
    main()
