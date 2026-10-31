#!/usr/bin/env python3
"""Summarize validated real samples and optionally draw a dependency-free SVG."""
import argparse
import html
import json
import math
import statistics
from pathlib import Path


def summarize(document):
    rows = []
    if not document.get("runs"):
        raise ValueError("no runs supplied")
    environments, shapes = set(), set()
    for run in document["runs"]:
        environment = tuple(run.get(key) for key in ["gpu", "compute_capability", "runtime_version", "driver_version", "cuda_compiler_version"])
        if any(value is None for value in environment):
            raise ValueError("incomplete GPU/environment metadata")
        environments.add(environment)
        if len(environments) != 1:
            raise ValueError("mixed GPU/environment runs must be analyzed separately")
        shape = (run.get("samples"), run.get("features"))
        if any(type(value) is not int or value < 1 for value in shape) or shape in shapes:
            raise ValueError("invalid or duplicate dimensions")
        shapes.add(shape)
        if run.get("validated") is not True or run.get("operation") != "uncentered_second_moment":
            raise ValueError("unvalidated or unexpected operation")
        kernels = {entry["kernel"]: entry for entry in run["results"]}
        if set(kernels) != {"baseline", "tiled"} or len(run["results"]) != 2:
            raise ValueError("expected baseline and tiled results")
        row = {"shape": f"{run['samples']}x{run['features']}", "gpu": run["gpu"]}
        for name, entry in kernels.items():
            values = entry["milliseconds"]
            if not values or any(not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("timings must be positive finite samples")
            row[name] = statistics.median(values)
            row[name + "_min"] = min(values)
            row[name + "_max"] = max(values)
            row[name + "_count"] = len(values)
        if row["baseline_count"] != row["tiled_count"]:
            raise ValueError("unmatched sample counts")
        row["speedup"] = row["baseline"] / row["tiled"]
        rows.append(row)
    return rows


def draw(rows, destination):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    rows = summarize(json.loads(args.input.read_text()))
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    if args.figure:
        draw(rows, args.figure)


if __name__ == "__main__":
    main()
