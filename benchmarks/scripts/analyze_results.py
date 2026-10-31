"""Validate a complete CPU run and write separated, non-destructive analysis."""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
from benchmark_data import CHUNKS, POLICIES, THREADS, group_name, load_run, source_hashes

NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)


def element(parent, tag, text=None, **attrs):
    node = ET.SubElement(parent, "{" + NS + "}" + tag,
                         {key.replace("_", "-"): str(value) for key, value in attrs.items()})
    if text is not None:
        node.text = str(text)
    return node


def canvas(title, subtitle, width=1100, height=500):
    root = ET.Element("{" + NS + "}svg", {"width": str(width), "height": str(height),
                      "viewBox": f"0 0 {width} {height}", "role": "img"})
    element(root, "title", title)
    element(root, "desc", subtitle)
    element(root, "rect", x=0, y=0, width=width, height=height, fill="white")
    element(root, "text", title, x=25, y=30, font_family="sans-serif", font_size=20, fill="#14273b")
    element(root, "text", subtitle, x=25, y=55, font_family="sans-serif", font_size=12, fill="#46566a")
    return root


def save_svg(root, path):
    ET.indent(root, space="  ")
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")


def monte_carlo_chart(groups, summary, out):
    selected = {key: rows for key, rows in groups.items() if key[0] == "pi_mc"}
    if not selected:
        return
    values = [row["ms"] for rows in selected.values() for row in rows]
    lower, upper = math.floor(math.log10(min(values))), math.ceil(math.log10(max(values)))
    if upper == lower:
        upper += 1
    left, right, top, bottom = 100, 1040, 115, 420
    y = lambda value: bottom - (math.log10(value) - lower) / (upper - lower) * (bottom - top)
    x = lambda thread: left + THREADS.index(thread) / (len(THREADS) - 1) * (right - left)
    repeats = next(iter(summary.values()))["runs"]
    root = canvas("Monte Carlo: counter synchronization and thread scaling",
                  f"Same indexed samples and seed; median with min-max whiskers; n={repeats}; exploratory measurements")
    for power in range(lower, upper + 1):
        value = 10 ** power
        element(root, "line", x1=left, y1=y(value), x2=right, y2=y(value), stroke="#dce3eb")
        element(root, "text", f"{value:g}", x=left - 12, y=y(value) + 4,
                text_anchor="end", font_family="sans-serif", font_size=12)
    for thread in THREADS:
        element(root, "text", thread, x=x(thread), y=bottom + 23, text_anchor="middle",
                font_family="sans-serif", font_size=12)
    element(root, "line", x1=left, y1=top, x2=left, y2=bottom, stroke="#566476")
    element(root, "line", x1=left, y1=bottom, x2=right, y2=bottom, stroke="#566476")
    element(root, "text", "Worker count (doubling steps)", x=(left + right) / 2, y=bottom + 52,
            text_anchor="middle", font_family="sans-serif", font_size=13)
    element(root, "text", "Wall time (ms, log10)", x=left, y=top - 13,
            font_family="sans-serif", font_size=12)
    serial = next(summary[group_name(key)]["median_ms"] for key in selected if key[1] == "serial")
    element(root, "line", x1=left, y1=y(serial), x2=right, y2=y(serial),
            stroke="#657183", stroke_dasharray="6 4", stroke_width=2)
    for mode, color, legend_x in [("local", "#176ca4", 290), ("mutex", "#c45a23", 500)]:
        configs = sorted((key for key in selected if key[1] == mode), key=lambda key: key[2])
        points = []
        for key in configs:
            stat = summary[group_name(key)]
            px, py = x(key[2]), y(stat["median_ms"])
            points.append(f"{px:.3f},{py:.3f}")
            element(root, "line", x1=px, y1=y(stat["min_ms"]), x2=px, y2=y(stat["max_ms"]), stroke=color)
            for value in (stat["min_ms"], stat["max_ms"]):
                element(root, "line", x1=px - 5, y1=y(value), x2=px + 5, y2=y(value), stroke=color)
            circle = element(root, "circle", cx=px, cy=py, r=4, fill=color)
            element(circle, "title", f"{mode}, {key[2]} workers: median {stat['median_ms']:.6f} ms; min {stat['min_ms']:.6f}, max {stat['max_ms']:.6f}")
        element(root, "polyline", points=" ".join(points), fill="none", stroke=color, stroke_width=2)
        element(root, "line", x1=legend_x, y1=83, x2=legend_x + 25, y2=83, stroke=color, stroke_width=3)
        element(root, "text", f"{mode} counters", x=legend_x + 33, y=87, font_family="sans-serif", font_size=12)
    element(root, "line", x1=720, y1=83, x2=745, y2=83, stroke="#657183", stroke_dasharray="6 4")
    element(root, "text", "serial median (one worker)", x=753, y=87, font_family="sans-serif", font_size=12)
    save_svg(root, out / "monte-carlo-scaling.svg")


def scheduling_chart(layout, groups, summary, out):
    selected = sorted((key for key in groups if key[0] == "scheduling" and key[1] == layout),
                      key=lambda key: (POLICIES.index(key[2]), key[4]))
    if not selected:
        return
    first = selected[0]
    repeats = len(groups[first])
    team = groups[first][0]["actual_threads"]
    subtitle = (f"{first[5]} items, light={first[6]}, heavy={first[7]}; requested {first[3]}, actual {team} workers; "
                f"n={repeats}; median and min-max; independent x-axis per workload")
    root = canvas(f"OpenMP scheduling: {layout} workload", subtitle, height=500)
    left, right, top, bottom = 240, 800, 100, 415
    limit = max(summary[group_name(key)]["max_ms"] for key in selected) * 1.12
    x = lambda value: left + value / limit * (right - left)
    colors = {"static": "#176ca4", "dynamic": "#267b61", "guided": "#7754a0", "auto": "#b96420"}
    for tick in range(6):
        value = limit * tick / 5
        element(root, "line", x1=x(value), y1=top - 12, x2=x(value), y2=bottom, stroke="#dce3eb")
        element(root, "text", f"{value:.3f}", x=x(value), y=bottom + 22, text_anchor="middle", font_family="sans-serif", font_size=11)
    element(root, "text", "Measured wall time (ms)", x=(left + right) / 2, y=bottom + 48,
            text_anchor="middle", font_family="sans-serif", font_size=13)
    for i, key in enumerate(selected):
        stat = summary[group_name(key)]
        py = top + i * 30
        label = "auto (runtime choice)" if key[2] == "auto" else f"{key[2]} / chunk {key[4]}"
        element(root, "text", label, x=25, y=py + 13, font_family="sans-serif", font_size=12)
        element(root, "rect", x=left, y=py, width=x(stat["median_ms"]) - left, height=18, fill=colors[key[2]])
        element(root, "line", x1=x(stat["min_ms"]), y1=py + 9, x2=x(stat["max_ms"]), y2=py + 9, stroke="#152438", stroke_width=2)
        for value in (stat["min_ms"], stat["max_ms"]):
            element(root, "line", x1=x(value), y1=py + 3, x2=x(value), y2=py + 15, stroke="#152438")
        element(root, "text", f"{stat['median_ms']:.6f} [{stat['min_ms']:.6f}, {stat['max_ms']:.6f}]",
                x=830, y=py + 13, font_family="monospace", font_size=10)
    element(root, "text", "Small, uncontrolled run: ranges are not confidence intervals; compare policies within this workload.",
            x=25, y=490, font_family="sans-serif", font_size=11, fill="#46566a")
    save_svg(root, out / f"openmp-{layout}.svg")


def records_for_table(groups, summary):
    serial = next(stat["median_ms"] for name, stat in summary.items() if name == "pi/serial/1 threads")
    result = []
    ordered = sorted(groups, key=lambda key: (0, ("serial", "local", "mutex").index(key[1]), key[2])
                     if key[0] == "pi_mc" else (1, ("uniform", "tail").index(key[1]), POLICIES.index(key[2]), key[4]))
    for key in ordered:
        rows = groups[key]
        stat = summary[group_name(key)]
        item = {"configuration": group_name(key), "experiment": key[0],
                "layout": key[1] if key[0] == "scheduling" else "",
                "mode_or_policy": key[1] if key[0] == "pi_mc" else key[2],
                "requested_workers": key[2] if key[0] == "pi_mc" else key[3],
                "actual_workers": key[2] if key[0] == "pi_mc" else rows[0]["actual_threads"],
                "chunk": "" if key[0] == "pi_mc" else key[4], **stat,
                "relative_time_to_serial": stat["median_ms"] / serial if key[0] == "pi_mc" else "",
                "assigned_work_max_over_mean_median": ""}
        if key[0] == "scheduling":
            ratios = []
            for row in rows:
                work = [worker["iterations"] for worker in row["workers"]]
                ratios.append(max(work) / statistics.mean(work))
            item["assigned_work_max_over_mean_median"] = statistics.median(ratios)
        result.append(item)
    return result


def write_analysis(run, out):
    run, out = Path(run), Path(out)
    if out.exists():
        raise ValueError("analysis output directory already exists; choose a fresh --out")
    # Complete validation precedes all output creation.
    hashes = source_hashes(run)
    meta, rows, groups, summary = load_run(run)
    if hashes != source_hashes(run):
        raise ValueError("source evidence changed while being validated")
    table = records_for_table(groups, summary)
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (out / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    monte_carlo_chart(groups, summary, out)
    for layout in ("uniform", "tail"):
        scheduling_chart(layout, groups, summary, out)
    relative = Path(os.path.relpath(run.resolve(), out.resolve())).as_posix()
    provenance = {
        "analysis_schema_version": 2,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_run_relative_to_analysis": relative, "source_sha256": hashes,
        "source_recorded_utc": meta.get("recorded_utc"), "source_environment": {
            key: meta.get(key) for key in ("platform", "machine", "logical_cpus", "processor", "cpu_model_captured_now", "compiler", "flags")},
        "records": len(rows), "configurations": len(groups), "repeats_per_configuration": meta["argv"]["repeats"],
        "validation": "Complete declared configuration/repeat matrix; field/type/finite checks; output consistency; worker workload conservation; recorded summary agreement when supplied.",
        "scope": "Derived statistics and figures only. No new measurements; no GPU data. Metadata is preserved evidence, not build/hardware attestation.",
        "graphics": "Deterministic dependency-free SVG rendered with the Python standard-library XML API.",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    model_note = ("CPU model is present only as captured source metadata; it is not inferred from this analysis host."
                  if meta.get("cpu_model_captured_now") else
                  "CPU model was not recorded in the source; it has not been inferred from a later host probe.")
    lines = ["# CPU Benchmark Analysis", "", "Derived from the retained exploratory measurements; **no new timings or GPU results** were created.", "",
             f"Source: [{run.name}]({relative}/). {len(rows)} records, {len(groups)} configurations, {meta['argv']['repeats']} repeats per configuration.", "",
             f"The source run had no affinity, thermal/frequency or background-load control. Min-max whiskers show the observed repetitions, not confidence intervals. {model_note}", "",
             "## Monte Carlo synchronization and scaling", "", "![Monte Carlo scaling](monte-carlo-scaling.svg)", "",
             "The y-axis is logarithmic. The dashed serial median is a one-worker baseline, not another multithreaded series. Relative time below is median / serial median for identical samples; it is a run-specific ratio, not a general speedup claim.", "",
             "| Mode | Workers | Median ms | Min-max ms | Relative time to serial |", "|---|---:|---:|---:|---:|"]
    for item in table:
        if item["experiment"] == "pi_mc":
            lines.append(f"| {item['mode_or_policy']} | {item['requested_workers']} | {item['median_ms']:.6f} | {item['min_ms']:.6f}-{item['max_ms']:.6f} | {item['relative_time_to_serial']:.3f} |")
    for layout in ("uniform", "tail"):
        subset = [item for item in table if item["layout"] == layout]
        if not subset:
            continue
        lines += ["", f"## OpenMP: {layout}", "", f"![OpenMP {layout}](openmp-{layout}.svg)", "",
                  "Panels have independent time-axis ranges. Compare policy/chunk settings within the same workload. Assigned-work imbalance is max worker iterations / mean worker iterations; 1.0 is equal assignment, not measured CPU utilization.", "",
                  "| Policy | Chunk | Actual workers | Median ms | Min-max ms | Assigned work max/mean |", "|---|---:|---:|---:|---:|---:|"]
        for item in subset:
            chunk = "runtime choice" if item["mode_or_policy"] == "auto" else item["chunk"]
            lines.append(f"| {item['mode_or_policy']} | {chunk} | {item['actual_workers']} | {item['median_ms']:.6f} | {item['min_ms']:.6f}-{item['max_ms']:.6f} | {item['assigned_work_max_over_mean_median']:.3f} |")
    lines += ["", "## Evidence and limits", "", "[Statistics CSV](summary.csv) contains all configurations; [summary JSON](summary.json) preserves the legacy statistics schema. [Provenance](provenance.json) hashes the immutable source files and records validation scope.", "",
              "Thread/team creation and synchronization are included in measured algorithm wall time; process startup and correctness references are excluded. These very small workloads expose startup/coordination effects. They do not establish a universally best thread count or scheduling policy. Compiler/flags in old metadata are operator declarations, not independently verified build commands. The analyzer checks internal consistency, not whether a timing record was genuinely executed.", ""]
    (out / "README.md").write_text("\n".join(lines))
    if hashes != source_hashes(run):
        raise RuntimeError("source evidence changed while being analyzed; outputs must not be trusted")
    return provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="new directory for derived artifacts; source files are never overwritten")
    args = ap.parse_args()
    try:
        result = write_analysis(args.run, args.out)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        ap.error(str(error))
    print(f"Validated {result['records']} records / {result['configurations']} configurations; wrote analysis to {args.out}")


if __name__ == "__main__":
    main()
