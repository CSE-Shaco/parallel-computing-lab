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
                        "runs": [{"synthetic_unit_fixture": True}],
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


if __name__ == "__main__":
    unittest.main()
