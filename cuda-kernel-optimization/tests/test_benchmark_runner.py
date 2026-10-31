"""Synthetic subprocess protocol tests; no GPU performance data is generated."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "benchmark_runner", Path(__file__).parents[1] / "benchmarks/run_benchmark.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def run_fixture(m=1, n=1, repeats=3):
    names = ["baseline", "tiled", "coalesced_tiled", "register_tiled"]
    result = {"schema_version": 2, "operation": "uncentered_second_moment", "validated": True,
              "samples": m, "features": n, "warmup": 10, "kernel_order": "rotating", "kernel_names": names,
              "gpu": "synthetic-unit-only", "compute_capability": "0.0", "runtime_version": 0,
              "driver_version": 0, "cuda_compiler_version": "synthetic", "results": []}
    for name in names:
        register = name == "register_tiled"
        result["results"].append({"kernel": name, "max_abs_error": 0, "milliseconds": [1.0] * repeats,
                                  "block": [32,8] if register else [16,16],
                                  "output_tile": [32,32] if register else [16,16],
                                  "reduction_tile": 1 if name == "baseline" else 32 if register else 16,
                                  "outputs_per_thread": 4 if register else 1})
    return result


class BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(benchmark.platform, "platform", return_value="synthetic-unit-host")
        patch.start()
        self.addCleanup(patch.stop)

    def arguments(self, binary, output):
        return ["--binary", str(binary), "--output", str(output), "--shape", "1,1",
                "--repeats", "3", "--build-options", "synthetic-unit-fixture"]

    def test_validates_and_retains_four_variant_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary, output = root / "fixture-binary", root / "fixture.json"
            binary.write_bytes(b"synthetic fixture, not executable")
            reply = subprocess.CompletedProcess([], 0, json.dumps(run_fixture()), "")
            with mock.patch.object(benchmark.subprocess, "run", return_value=reply), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(benchmark.main(self.arguments(binary, output)), 0)
            document = json.loads(output.read_text())
            self.assertEqual(document["benchmark_schema_version"], 2)
            self.assertEqual(document["measurement_policy"], {"warmup": 10, "kernel_order": "rotating"})
            self.assertEqual(document["runs"][0]["kernel_names"], ["baseline", "tiled", "coalesced_tiled", "register_tiled"])

    def test_refuses_wrong_dimensions_repetitions_or_kernel_set_before_output(self):
        for mutation in ("dimensions", "repetitions", "kernel_set"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                binary, output = root / "fixture-binary", root / "fixture.json"
                binary.write_bytes(b"synthetic fixture")
                run = run_fixture()
                if mutation == "dimensions": run["samples"] = 2
                elif mutation == "repetitions":
                    for entry in run["results"]: entry["milliseconds"].pop()
                else: run["results"].pop()
                reply = subprocess.CompletedProcess([], 0, json.dumps(run), "")
                with mock.patch.object(benchmark.subprocess, "run", return_value=reply), self.assertRaises(ValueError):
                    benchmark.main(self.arguments(binary, output))
                self.assertFalse(output.exists())

    def test_all_shapes_are_checked_before_binary_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary, output = root / "fixture-binary", root / "fixture.json"
            for shape in ("0,1", "1,1"):
                args = self.arguments(binary, output) + ["--shape", shape]
                with self.subTest(shape=shape), mock.patch.object(benchmark.subprocess, "run") as execute, \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    benchmark.main(args)
                execute.assert_not_called()
                self.assertFalse(output.exists())

    def test_changed_binary_is_not_recorded_as_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary, output = root / "fixture-binary", root / "fixture.json"
            binary.write_bytes(b"first")
            def changed(command, **kwargs):
                binary.write_bytes(b"changed")
                return subprocess.CompletedProcess(command, 0, json.dumps(run_fixture()), "")
            with mock.patch.object(benchmark.subprocess, "run", side_effect=changed), self.assertRaises(ValueError):
                benchmark.main(self.arguments(binary, output))
            self.assertFalse(output.exists())

    def test_refuses_existing_output_and_dangling_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "keep.json"
            output.write_text("keep")
            for destination in (output, root / "dangling.json"):
                if destination != output: destination.symlink_to(root / "missing")
                with mock.patch.object(benchmark.subprocess, "run") as execute, \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    benchmark.main(self.arguments(root / "absent", destination))
                execute.assert_not_called()
            self.assertEqual(output.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
