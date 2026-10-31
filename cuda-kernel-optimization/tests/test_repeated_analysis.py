"""Synthetic protocol tests only; no timings here are GPU performance evidence."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))
import analyze_repeated_runs as repeated
from analyze_results import KERNELS, VARIANTS, summarize


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=True) + "\n")


def synthetic_session(parent, name, *, baseline=1.0, register=.5, gpu="synthetic-unit-only"):
    """Create a temporary synthetic record with the real validation schema."""
    root = parent / name
    (root / "raw").mkdir(parents=True)
    (root / "logs").mkdir()
    run = {"schema_version": 2, "operation": "uncentered_second_moment", "validated": True,
           "samples": 1, "features": 1, "gpu": gpu, "compute_capability": "8.9",
           "runtime_version": 13030, "driver_version": 13030, "cuda_compiler_version": "synthetic",
           "warmup": 10, "kernel_order": "rotating", "kernel_names": list(KERNELS), "results": []}
    for kernel in KERNELS:
        value = register if kernel == "register_tiled" else baseline if kernel == "baseline" else baseline * 2
        run["results"].append({"kernel": kernel, "milliseconds": [value] * 3,
                               "max_abs_error": 0.0, **copy.deepcopy(VARIANTS[kernel])})
    binary_hash = hashlib.sha256(("synthetic binary " + name).encode()).hexdigest()
    document = {"binary_sha256": binary_hash, "runs": [run]}
    write_json(root / "benchmark.json", document)
    write_json(root / "raw/shape-1x1.json", document)
    write_json(root / "summary.json", summarize(document))
    (root / "kernel-time.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>\n')
    labels = ["build-host-reference", "host-reference", "build-host-layout", "host-layout",
              "build-correctness", "gpu-correctness", "memcheck", "racecheck", "build-benchmark",
              "benchmark-shape-0", "analysis"]
    stdout = {
        "host-reference": "CPU reference, validation rejection and input checks passed",
        "host-layout": "50 host layout/oracle cases passed (not GPU execution)",
        "gpu-correctness": "12 shapes passed for baseline, tiled, coalesced_tiled and register_tiled kernels",
        "memcheck": "========= ERROR SUMMARY: 0 errors",
        "racecheck": "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)",
    }
    commands = []
    for label in labels:
        args = ["/synthetic/bin/nvcc", "-std=c++17", "-O3", "-lineinfo", "-arch=sm_89",
                "-I", "/synthetic/module/include", "/synthetic/module/src/unit.cpp",
                "-o", f"/synthetic/{name}/build/{label}"] if label.startswith("build-") else ["synthetic-unit-only"]
        commands.append({"label": label, "command": args, "returncode": 0})
        (root / f"logs/{label}.stdout.log").write_text(stdout.get(label, "") + "\n")
        (root / f"logs/{label}.stderr.log").write_text("")
    execution = {"schema_version": 2, "benchmark_schema_version": 2, "status": "complete",
                 "host": "synthetic-unit-host", "architecture": "sm_89", "CUDA_VISIBLE_DEVICES": None,
                 "started_utc": "synthetic-" + name, "completed_utc": "synthetic-" + name,
                 "sanitizer": "PASS: memcheck and racecheck", "gpu_validation": repeated.VALIDATION,
                 "correctness_shapes": 12, "repeats": 3, "kernel_names": list(KERNELS),
                 "variant_configuration": copy.deepcopy(VARIANTS),
                 "measurement_policy": {"warmup": 10, "kernel_order": "rotating"},
                 "requested_shapes": ["1,1"], "completed_shapes": ["1,1"],
                 "source_sha256": {"unit.cpp": "a" * 64},
                 "binary_sha256": {name: binary_hash for name in ["reference_test", "tile_layout_test", "correctness_test", "kernel_benchmark"]},
                 "preflight": {"nvcc_version": {"stdout": "synthetic compiler"},
                               "gpu_inventory": {"stdout": "synthetic GPU inventory"}},
                 "commands": commands,
                 "result_sha256": {file: repeated.digest(root / file) for file in repeated.RESULT_FILES}}
    write_json(root / "execution.json", execution)
    return root


def edit_execution(root, edit):
    execution = json.loads((root / "execution.json").read_text())
    edit(execution)
    write_json(root / "execution.json", execution)


def rehash_results(root):
    edit_execution(root, lambda e: e.update(result_sha256={file: repeated.digest(root / file) for file in repeated.RESULT_FILES}))


class RepeatedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.first = synthetic_session(self.root, "synthetic-first", baseline=1.0, register=.5)
        self.second = synthetic_session(self.root, "synthetic-second", baseline=100.0, register=100.0)

    def tearDown(self):
        self.temp.cleanup()

    def test_aggregates_paired_ratios_not_pooled_launch_medians(self):
        report, rows, provenance = repeated.aggregate([self.first, self.second])
        register = report["shapes"][0]["kernels"]["register_tiled"]
        self.assertEqual(register["paired_ratios"], [2.0, 1.0])
        self.assertEqual(register["paired_ratio_median"], 1.5)
        self.assertEqual(register["runtime_reduction_percent_median"], 25.0)
        self.assertEqual(register["sessions_lower_time_than_baseline"], 1)
        self.assertNotEqual(register["paired_ratio_median"], 50.5 / 50.25)
        self.assertEqual(len(rows), 8)
        self.assertEqual(report["total_launch_samples"], 24)
        self.assertNotEqual(provenance["runs"][0]["binary_sha256_recorded"], provenance["runs"][1]["binary_sha256_recorded"])

    def test_requires_two_unique_runs(self):
        for paths in ([self.first], [self.first, self.first], [self.first, self.first / "."]):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                repeated.aggregate(paths)

    def test_rejects_duplicate_execution_content(self):
        third = self.root / "synthetic-copy"
        shutil.copytree(self.first, third)
        with self.assertRaisesRegex(ValueError, "duplicate execution"):
            repeated.aggregate([self.first, third])

    def test_rejects_failed_or_unverified_sanitizer(self):
        edit_execution(self.second, lambda e: e.update(status="failed"))
        with self.assertRaisesRegex(ValueError, "not complete"):
            repeated.aggregate([self.first, self.second])
        edit_execution(self.second, lambda e: e.update(status="complete", sanitizer="Not Verified"))
        with self.assertRaisesRegex(ValueError, "validation gates"):
            repeated.aggregate([self.first, self.second])

    def test_rejects_result_hash_tampering(self):
        with (self.second / "benchmark.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            repeated.aggregate([self.first, self.second])

    def test_rejects_rehashed_summary_disagreement(self):
        write_json(self.second / "summary.json", [])
        rehash_results(self.second)
        with self.assertRaisesRegex(ValueError, "summary disagrees"):
            repeated.aggregate([self.first, self.second])

    def test_rejects_per_shape_raw_disagreement(self):
        write_json(self.second / "raw/shape-1x1.json", {"runs": []})
        with self.assertRaisesRegex(ValueError, "raw record disagrees"):
            repeated.aggregate([self.first, self.second])

    def test_rejects_nonfinite_data_even_when_rehashed(self):
        data = json.loads((self.second / "benchmark.json").read_text())
        data["runs"][0]["results"][0]["milliseconds"][0] = float("nan")
        write_json(self.second / "benchmark.json", data)
        rehash_results(self.second)
        with self.assertRaisesRegex(ValueError, "finite"):
            repeated.aggregate([self.first, self.second])

    def test_rejects_different_device_or_source(self):
        other = synthetic_session(self.root, "synthetic-other", gpu="different-synthetic-unit-device")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            repeated.aggregate([self.first, other])
        edit_execution(self.second, lambda e: e.update(source_sha256={"unit.cpp": "b" * 64}))
        with self.assertRaisesRegex(ValueError, "incompatible"):
            repeated.aggregate([self.first, self.second])

    def test_rejects_bad_command_or_sanitizer_diagnostic(self):
        edit_execution(self.second, lambda e: e["commands"][0].update(returncode=1))
        with self.assertRaisesRegex(ValueError, "return code"):
            repeated.aggregate([self.first, self.second])
        edit_execution(self.second, lambda e: e["commands"][0].update(returncode=0))
        with (self.second / "logs/racecheck.stdout.log").open("a") as stream:
            stream.write("========= Error: synthetic diagnostic\n")
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            repeated.aggregate([self.first, self.second])

    def test_duplicate_json_keys_and_symlinked_input_rejected(self):
        (self.second / "execution.json").write_text('{"status":"complete","status":"failed"}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            repeated.aggregate([self.first, self.second])
        (self.second / "execution.json").unlink()
        (self.second / "execution.json").symlink_to(self.first / "execution.json")
        with self.assertRaisesRegex(ValueError, "symlinked"):
            repeated.aggregate([self.first, self.second])

    def test_existing_output_preserved_and_failed_validation_creates_none(self):
        out = self.root / "existing"
        out.mkdir(); (out / "marker").write_text("keep")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            repeated.main([str(self.first), str(self.second), "--out", str(out)])
        self.assertEqual((out / "marker").read_text(), "keep")
        edit_execution(self.second, lambda e: e.update(status="failed"))
        out = self.root / "new"
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            repeated.main([str(self.first), str(self.second), "--out", str(out)])
        self.assertFalse(out.exists())

    def test_nested_output_refused_before_mutating_record(self):
        before = {str(p.relative_to(self.first)): p.read_bytes() for p in self.first.rglob("*") if p.is_file()}
        out = self.first / "new-analysis"
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            repeated.main([str(self.first), str(self.second), "--out", str(out)])
        self.assertFalse(out.exists())
        self.assertEqual(before, {str(p.relative_to(self.first)): p.read_bytes() for p in self.first.rglob("*") if p.is_file()})

    def test_finite_input_overflow_ratio_is_rejected(self):
        extreme = synthetic_session(self.root, "synthetic-extreme", baseline=1e200, register=1e-200)
        with self.assertRaisesRegex(ValueError, "non-finite computed"):
            repeated.aggregate([extreme, self.second])

    def test_mismatched_build_options_refused(self):
        edit_execution(self.second, lambda e: e["commands"][0]["command"].insert(2, "--use_fast_math"))
        with self.assertRaisesRegex(ValueError, "incompatible"):
            repeated.aggregate([self.first, self.second])

    def test_bool_execution_configuration_refused(self):
        edit_execution(self.second, lambda e: e["variant_configuration"]["baseline"].update(outputs_per_thread=True))
        with self.assertRaisesRegex(ValueError, "integer configuration"):
            repeated.aggregate([self.first, self.second])

    def test_successful_cli_outputs_and_svg_points_are_reproducible(self):
        out = self.root / "synthetic-analysis-only"
        before = {str(p.relative_to(self.root)): p.read_bytes() for folder in (self.first, self.second) for p in folder.rglob("*") if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(repeated.main([str(self.first), str(self.second), "--out", str(out)]), 0)
        self.assertEqual(set(p.name for p in out.iterdir()), {"summary.json", "provenance.json", "per-run.csv", "stability.svg"})
        svg = ET.parse(out / "stability.svg")
        self.assertEqual(len(svg.findall(".//{http://www.w3.org/2000/svg}circle")), 6)
        for name, content in before.items():
            self.assertEqual((self.root / name).read_bytes(), content)
        second_out = self.root / "second-synthetic-analysis-only"
        with contextlib.redirect_stdout(io.StringIO()):
            repeated.main([str(self.first), str(self.second), "--out", str(second_out)])
        for name in ("summary.json", "provenance.json", "per-run.csv", "stability.svg"):
            self.assertEqual((out / name).read_bytes(), (second_out / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
