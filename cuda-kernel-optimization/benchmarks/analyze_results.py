#!/usr/bin/env python3
"""Validate and summarize supplied GPU samples; synthetic tests are not evidence."""
import argparse
import html
import json
import math
import statistics
from pathlib import Path


LEGACY_KERNELS = ("baseline", "tiled")
KERNELS = ("baseline", "tiled", "coalesced_tiled", "register_tiled")
VARIANTS = {
    name: {"block": [32, 8] if name == "register_tiled" else [16, 16],
           "output_tile": [32, 32] if name == "register_tiled" else [16, 16],
           "reduction_tile": 1 if name == "baseline" else 32 if name == "register_tiled" else 16,
           "outputs_per_thread": 4 if name == "register_tiled" else 1}
    for name in KERNELS
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def load_document(text):
    return json.loads(text, object_pairs_hook=_unique_object)


def validate_run(run, *, required_schema=None, repeats=None):
    """Return results by kernel after checking the recorded execution contract."""
    if not isinstance(run, dict):
        raise ValueError("run must be an object")
    schema = run.get("schema_version", 1)
    if type(schema) is not int or schema not in (1, 2):
        raise ValueError("unsupported benchmark schema")
    if required_schema is not None and schema != required_schema:
        raise ValueError("benchmark schema does not match this runner")
    if run.get("validated") is not True or run.get("operation") != "uncentered_second_moment":
        raise ValueError("unvalidated or unexpected operation")
    if any(type(run.get(key)) is not int or run[key] < 1 for key in ("samples", "features")):
        raise ValueError("invalid dimensions")
    for key in ("gpu", "compute_capability", "cuda_compiler_version"):
        if not isinstance(run.get(key), str) or not run[key].strip():
            raise ValueError("incomplete GPU/environment metadata")
    for key in ("runtime_version", "driver_version"):
        if type(run.get(key)) is not int or run[key] < 0:
            raise ValueError("incomplete GPU/environment metadata")
    expected = LEGACY_KERNELS if schema == 1 else KERNELS
    results = run.get("results")
    if not isinstance(results, list) or len(results) != len(expected):
        raise ValueError("unexpected kernel result count")
    kernels = {}
    for entry in results:
        if not isinstance(entry, dict) or entry.get("kernel") not in expected:
            raise ValueError("unexpected kernel result")
        name = entry["kernel"]
        if name in kernels:
            raise ValueError("duplicate kernel result")
        values = entry.get("milliseconds")
        if not isinstance(values, list) or not values or any(
                type(value) not in (float, int) or not math.isfinite(value) or value <= 0
                for value in values):
            raise ValueError("timings must be positive finite samples")
        if repeats is not None and len(values) != repeats:
            raise ValueError("incorrect repetition count")
        if schema == 2:
            error = entry.get("max_abs_error")
            if type(error) not in (float, int) or not math.isfinite(error) or error < 0:
                raise ValueError("invalid correctness error metadata")
            for key, expected_value in VARIANTS[name].items():
                value = entry.get(key)
                if isinstance(expected_value, list):
                    valid = (isinstance(value, list) and len(value) == len(expected_value)
                             and all(type(item) is int for item in value)
                             and value == expected_value)
                else:
                    valid = type(value) is int and value == expected_value
                if not valid:
                    raise ValueError("unexpected variant configuration: " + name + "." + key)
        kernels[name] = entry
    if set(kernels) != set(expected):
        raise ValueError("missing kernel result")
    if len({len(entry["milliseconds"]) for entry in kernels.values()}) != 1:
        raise ValueError("unmatched sample counts")
    if schema == 2:
        if (type(run.get("warmup")) is not int or run["warmup"] != 10
                or run.get("kernel_order") != "rotating"
                or run.get("kernel_names") != list(KERNELS) or "block" in run):
            raise ValueError("unexpected four-kernel measurement policy")
    return kernels


def summarize(document):
    rows = []
    if not isinstance(document, dict) or not isinstance(document.get("runs"), list) or not document["runs"]:
        raise ValueError("no runs supplied")
    environments, shapes, schemas, repetitions = set(), set(), set(), set()
    for run in document["runs"]:
        kernels = validate_run(run)
        schema = run.get("schema_version", 1)
        schemas.add(schema)
        if len(schemas) != 1:
            raise ValueError("mixed benchmark schemas must be analyzed separately")
        environment = tuple(run[key] for key in ("gpu", "compute_capability", "runtime_version", "driver_version", "cuda_compiler_version"))
        environments.add(environment)
        if len(environments) != 1:
            raise ValueError("mixed GPU/environment runs must be analyzed separately")
        shape = (run["samples"], run["features"])
        if shape in shapes:
            raise ValueError("duplicate dimensions")
        shapes.add(shape)
        repetitions.add(len(kernels["baseline"]["milliseconds"]))
        if len(repetitions) != 1:
            raise ValueError("mixed repetition counts must be analyzed separately")
        row = {"shape": f"{run['samples']}x{run['features']}", "gpu": run["gpu"]}
        if schema == 1:
            # Preserve legacy summary keys and SVG bytes for retained measurements.
            for name, entry in kernels.items():
                values = entry["milliseconds"]
                row[name] = statistics.median(values)
                row[name + "_min"] = min(values)
                row[name + "_max"] = max(values)
                row[name + "_count"] = len(values)
            row["speedup"] = row["baseline"] / row["tiled"]
        else:
            baseline = statistics.median(kernels["baseline"]["milliseconds"])
            row["schema_version"] = 2
            row["speedup_definition"] = "baseline median / candidate median"
            row["kernels"] = {}
            for name in KERNELS:
                values = kernels[name]["milliseconds"]
                median = statistics.median(values)
                row["kernels"][name] = {
                    "median_ms": median, "min_ms": min(values), "max_ms": max(values),
                    "count": len(values), "baseline_speedup": baseline / median,
                    **VARIANTS[name],
                }
        rows.append(row)
    return rows


def draw_legacy(rows, destination):
    width, height = 900, 90 + 90 * len(rows)
    maximum = max(max(r["baseline_max"], r["tiled_max"]) for r in rows)
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<g font-family="sans-serif" font-size="13" fill="#18212a">',
              '<text x="20" y="25">Kernel time (ms): median bars; whiskers show min–max</text>',
              '<text x="20" y="47">Blue: baseline. Orange: shared-memory tiled. Transfers excluded.</text>']
    for index, row in enumerate(rows):
        y = 80 + index * 90
        pieces.append(f'<text x="20" y="{y}">{html.escape(row["shape"])} ({row["speedup"]:.3f}×)</text>')
        for offset, name, color in [(8, "baseline", "#2767a3"), (32, "tiled", "#c76620")]:
            scale = 510 / maximum
            length = row[name] * scale
            low, high = 200 + row[name+"_min"] * scale, 200 + row[name+"_max"] * scale
            pieces.append(f'<rect x="200" y="{y+offset}" width="{length}" height="16" fill="{color}"/>')
            pieces.append(f'<path d="M{low},{y+offset+8} H{high} M{low},{y+offset+3} v10 M{high},{y+offset+3} v10" stroke="#111"/>')
            pieces.append(f'<text x="735" y="{y+offset+13}">{row[name]:.5f} ms (n={row[name+"_count"]})</text>')
    pieces.append('</g></svg>')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write("\n".join(pieces) + "\n")



def draw(rows, destination):
    if not rows:
        raise ValueError("no summary rows supplied")
    if rows[0].get("schema_version", 1) == 1:
        return draw_legacy(rows, destination)
    width, row_height = 1140, 154
    height = 90 + row_height * len(rows)
    maximum = max(value["max_ms"] for row in rows for value in row["kernels"].values())
    colors = ("#2767a3", "#c76620", "#26836b", "#8050a2")
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<g font-family="sans-serif" font-size="13" fill="#18212a">',
              '<text x="20" y="25">Kernel time (ms): median bars; whiskers show min–max</text>',
              '<text x="20" y="47">Ratios: baseline median / candidate median. Transfers excluded.</text>']
    scale = 510 / maximum
    for index, row in enumerate(rows):
        top = 80 + index * row_height
        pieces.append(f'<text x="20" y="{top}">{html.escape(row["shape"])}</text>')
        for order, (name, color) in enumerate(zip(KERNELS, colors)):
            value = row["kernels"][name]
            y = top + 10 + 26 * order
            low, high = 260 + value["min_ms"] * scale, 260 + value["max_ms"] * scale
            pieces.append(f'<text x="20" y="{y+13}">{html.escape(name)}</text>')
            pieces.append(f'<rect x="260" y="{y}" width="{value["median_ms"]*scale}" height="16" fill="{color}"/>')
            pieces.append(f'<path d="M{low},{y+8} H{high} M{low},{y+3} v10 M{high},{y+3} v10" stroke="#111"/>')
            pieces.append(f'<text x="795" y="{y+13}">{value["median_ms"]:.6f} ms | n={value["count"]} | {value["baseline_speedup"]:.3f}×</text>')
    pieces.append('</g></svg>')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write("\n".join(pieces) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    rows = summarize(load_document(args.input.read_text()))
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    if args.figure:
        draw(rows, args.figure)


if __name__ == "__main__":
    main()
