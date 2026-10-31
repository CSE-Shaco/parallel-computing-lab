"""Host-only runner safeguards; these fixtures never create GPU measurements."""
import argparse
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
    "gpu_validation_runner", Path(__file__).parents[1] / "benchmarks/run_gpu_validation.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def synthetic_run(dimensions="1,1", repeats=3):
    """Clearly synthetic protocol fixture, never device execution evidence."""
    m, n = map(int, dimensions.split(","))
    names = ["baseline", "tiled", "coalesced_tiled", "register_tiled"]
    results = []
    for name in names:
        register = name == "register_tiled"
        results.append({"kernel": name, "max_abs_error": 0.0,
                        "milliseconds": [1.0] * repeats,
                        "block": [32,8] if register else [16,16],
                        "output_tile": [32,32] if register else [16,16],
                        "reduction_tile": 1 if name == "baseline" else 32 if register else 16,
                        "outputs_per_thread": 4 if register else 1})
    return {"schema_version": 2, "validated": True, "operation": "uncentered_second_moment",
            "samples": m, "features": n, "gpu": "synthetic-unit-fixture",
            "compute_capability": "0.0", "runtime_version": 0, "driver_version": 0,
            "cuda_compiler_version": "synthetic", "warmup": 10, "kernel_order": "rotating",
            "kernel_names": names, "results": results}


class GpuRunnerTests(unittest.TestCase):
    def test_shape_validation_and_normalization(self):
        self.assertEqual(runner.shape("0017,19"), "17,19")
        for bad in ("0,1", "1,-1", "1", "1,2,3", "1000001,1"):
            with self.subTest(bad=bad), self.assertRaises(argparse.ArgumentTypeError):
                runner.shape(bad)

    def test_missing_tools_fail_without_measurement(self):
        with mock.patch.object(runner.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "missing nvcc, nvidia_smi"):
                runner.preflight()

    def test_device_probe_failure_propagates(self):
        def fail(command, **kwargs):
            if command[0] == "/tools/nvidia-smi":
                raise subprocess.CalledProcessError(9, command, stderr="no device")
            return subprocess.CompletedProcess(command, 0, "synthetic tool response", "")
        with mock.patch.object(runner.shutil, "which", side_effect=lambda name: "/tools/" + name), \
                mock.patch.object(runner.subprocess, "run", side_effect=fail):
            with self.assertRaises(subprocess.CalledProcessError):
                runner.preflight()

    def test_existing_output_refused_before_tools(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            marker = path / "keep.txt"
            marker.write_text("retained")
            with mock.patch.object(runner, "preflight") as probe, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                runner.main(["--out", str(path), "--arch", "sm_75"])
            self.assertEqual(error.exception.code, 2)
            probe.assert_not_called()
            self.assertEqual(marker.read_text(), "retained")

    def test_invalid_arch_refused_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "new-run"
            with mock.patch.object(runner, "preflight") as probe, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(["--out", str(path), "--arch", "native"])
            probe.assert_not_called()
            self.assertFalse(path.exists())

    def test_missing_tool_preflight_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "new-run"
            with mock.patch.object(runner, "preflight", side_effect=RuntimeError("no NVIDIA device")), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(["--out", str(path), "--arch", "sm_75"])
            self.assertFalse(path.exists())

    def test_compile_failure_retains_logs_and_no_timings(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "failed-run"
            args = argparse.Namespace(out=path, arch="sm_75", shape=[], repeats=30, timeout=1)
            tools = {"nvcc": "/synthetic/nvcc", "compute_sanitizer": None}
            failure = subprocess.CalledProcessError(1, ["synthetic compile"], output="", stderr="compile failed")
            with mock.patch.object(runner.subprocess, "run", side_effect=failure), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.run_validation(args, tools, {}), 1)
            record = json.loads((path / "execution.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["commands"][0]["returncode"], 1)
            self.assertIn("compile failed", (path / "logs/build-host-reference.stderr.log").read_text())
            self.assertFalse((path / "benchmark.json").exists())
            self.assertFalse((path / "kernel-time.svg").exists())

    def test_later_shape_failure_retains_prior_raw_document(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "partial-run"
            args = argparse.Namespace(out=path, arch="sm_75", shape=["1,1", "2,2"], repeats=3, timeout=1)
            tools = {"nvcc": "/synthetic/nvcc", "compute_sanitizer": None}

            def fake_command(command, **kwargs):
                if command[0] == tools["nvcc"]:
                    Path(command[-1]).write_bytes(b"synthetic compile fixture")
                elif command[0] == runner.sys.executable:
                    dimensions = command[command.index("--shape") + 1]
                    if dimensions == "2,2":
                        raise subprocess.CalledProcessError(1, command, stderr="synthetic second-shape failure")
                    destination = Path(command[command.index("--output") + 1])
                    destination.write_text(json.dumps({
                        "binary_sha256": runner.digest(path / "build/kernel_benchmark"),
                        "runs": [synthetic_run(dimensions)],
                    }))
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(runner.subprocess, "run", side_effect=fake_command), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.run_validation(args, tools, {}), 1)
            record = json.loads((path / "execution.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["completed_shapes"], ["1,1"])
            self.assertTrue((path / "raw/shape-1x1.json").is_file())
            self.assertFalse((path / "benchmark.json").exists())
            self.assertFalse((path / "kernel-time.svg").exists())


    def test_mocked_four_variant_pipeline_records_policy_and_compiles_all_sources(self):
        from analyze_results import draw
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic-protocol-unit-only"
            args = argparse.Namespace(out=path, arch="sm_89", shape=["1,1"], repeats=3, timeout=1)
            tools = {"nvcc": "/synthetic/nvcc", "compute_sanitizer": "/synthetic/compute-sanitizer"}
            def command_fixture(command, **kwargs):
                stdout = "synthetic unit response, not a GPU result"
                if command[0] == tools["nvcc"]:
                    Path(command[-1]).write_bytes(b"synthetic unit binary")
                elif command[0] == runner.sys.executable:
                    if "--shape" in command:
                        destination = Path(command[command.index("--output") + 1])
                        destination.write_text(json.dumps({
                            "binary_sha256": runner.digest(path / "build/kernel_benchmark"),
                            "runs": [synthetic_run()],
                        }))
                    else:
                        document = json.loads((path / "benchmark.json").read_text())
                        rows = runner.summarize(document)
                        draw(rows, Path(command[command.index("--figure") + 1]))
                        stdout = "\n".join(json.dumps(row) for row in rows)
                return subprocess.CompletedProcess(command, 0, stdout, "")
            with mock.patch.object(runner.subprocess, "run", side_effect=command_fixture):
                self.assertEqual(runner.run_validation(args, tools, {}), 0)
            record = json.loads((path / "execution.json").read_text())
            self.assertEqual(record["status"], "complete")
            self.assertEqual(record["kernel_names"], ["baseline", "tiled", "coalesced_tiled", "register_tiled"])
            self.assertEqual(record["measurement_policy"], {"warmup": 10, "kernel_order": "rotating"})
            self.assertEqual(record["correctness_shapes"], 12)
            self.assertEqual(record["variant_configuration"]["baseline"]["reduction_tile"], 1)
            self.assertEqual(set(record["binary_sha256"]), {"reference_test", "tile_layout_test", "correctness_test", "kernel_benchmark"})
            for label in ("build-correctness", "build-benchmark"):
                command = next(entry["command"] for entry in record["commands"] if entry["label"] == label)
                files = {Path(value).name for value in command}
                self.assertTrue({"baseline.cu", "tiled.cu", "coalesced_tiled.cu", "register_tiled.cu"}.issubset(files))
            labels = [entry["label"] for entry in record["commands"]]
            self.assertLess(labels.index("host-layout"), labels.index("gpu-correctness"))
            self.assertLess(labels.index("racecheck"), labels.index("build-benchmark"))

    def test_sanitizer_error_stops_before_any_benchmark(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "failed-sanitizer-unit-only"
            args = argparse.Namespace(out=path, arch="sm_89", shape=["1,1"], repeats=3, timeout=1)
            tools = {"nvcc": "/synthetic/nvcc", "compute_sanitizer": "/synthetic/compute-sanitizer"}
            def command_fixture(command, **kwargs):
                if command[0] == tools["nvcc"]:
                    Path(command[-1]).write_bytes(b"synthetic unit binary")
                if command[0] == tools["compute_sanitizer"]:
                    raise subprocess.CalledProcessError(99, command, stderr="synthetic sanitizer failure")
                return subprocess.CompletedProcess(command, 0, "synthetic response", "")
            with mock.patch.object(runner.subprocess, "run", side_effect=command_fixture), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.run_validation(args, tools, {}), 1)
            record = json.loads((path / "execution.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["commands"][-1]["label"], "memcheck")
            self.assertEqual(record["commands"][-1]["returncode"], 99)
            self.assertFalse((path / "benchmark.json").exists())
            self.assertFalse(any(entry["label"] == "build-benchmark" for entry in record["commands"]))

    def test_current_runner_refuses_old_two_kernel_results(self):
        fixture = synthetic_run()
        fixture["results"] = fixture["results"][:2]
        with self.assertRaises(ValueError):
            runner.validate_run(fixture, required_schema=2, repeats=3)
        fixture = synthetic_run()
        fixture["schema_version"] = 1
        with self.assertRaises(ValueError):
            runner.validate_run(fixture, required_schema=2, repeats=3)


if __name__ == "__main__":
    unittest.main()
