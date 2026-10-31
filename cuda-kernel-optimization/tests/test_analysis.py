"""Synthetic parser unit tests, never benchmark evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

spec = importlib.util.spec_from_file_location("analysis", Path(__file__).parents[1]/"benchmarks/analyze_results.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class AnalysisTest(unittest.TestCase):
    def fixture(self, schema=1):
        run = {"validated": True, "operation": "uncentered_second_moment", "samples": 2, "features": 3,
               "gpu": "synthetic-test-only", "compute_capability": "0.0", "runtime_version": 0,
               "driver_version": 0, "cuda_compiler_version": "synthetic", "results": [
                   {"kernel": "baseline", "milliseconds": [2,4,6]},
                   {"kernel": "tiled", "milliseconds": [1,2,3]}]}
        if schema == 2:
            run.update(schema_version=2, warmup=10, kernel_order="rotating",
                       kernel_names=["baseline", "tiled", "coalesced_tiled", "register_tiled"])
            run["results"] += [{"kernel": "coalesced_tiled", "milliseconds": [1, 1.5, 2]},
                               {"kernel": "register_tiled", "milliseconds": [0.5, 1, 1.5]}]
            for entry in run["results"]:
                register = entry["kernel"] == "register_tiled"
                entry.update(max_abs_error=0.0, block=[32, 8] if register else [16, 16],
                             output_tile=[32, 32] if register else [16, 16],
                             reduction_tile=1 if entry["kernel"] == "baseline" else 32 if register else 16,
                             outputs_per_thread=4 if register else 1)
        return {"runs": [run]}

    def test_median_ratio(self):
        self.assertEqual(analysis.summarize(self.fixture())[0]["speedup"], 2)

    def test_reject_unverified(self):
        fixture=self.fixture(); fixture["runs"][0]["validated"]=False
        with self.assertRaises(ValueError): analysis.summarize(fixture)

    def test_reject_invalid_times(self):
        for invalid in [0,-1,float("nan"),float("inf"),True,"1"]:
            fixture=self.fixture();fixture["runs"][0]["results"][0]["milliseconds"]=[invalid]
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): analysis.summarize(fixture)

    def test_reject_mixed_environment(self):
        fixture=self.fixture(); other=copy.deepcopy(fixture["runs"][0]);other["samples"]=3;other["gpu"]="other-synthetic"
        fixture["runs"].append(other)
        with self.assertRaises(ValueError): analysis.summarize(fixture)

    def test_reject_duplicate_dimensions(self):
        fixture=self.fixture();fixture["runs"].append(copy.deepcopy(fixture["runs"][0]))
        with self.assertRaises(ValueError): analysis.summarize(fixture)

    def test_reject_missing_kernel(self):
        fixture=self.fixture();fixture["runs"][0]["results"].pop()
        with self.assertRaises(ValueError): analysis.summarize(fixture)

    def test_four_variant_medians_and_baseline_ratios(self):
        row = analysis.summarize(self.fixture(2))[0]
        self.assertEqual(row["schema_version"], 2)
        self.assertEqual(list(row["kernels"]), ["baseline", "tiled", "coalesced_tiled", "register_tiled"])
        self.assertEqual(row["kernels"]["register_tiled"]["median_ms"], 1)
        self.assertEqual(row["kernels"]["register_tiled"]["baseline_speedup"], 4)
        self.assertEqual(row["kernels"]["coalesced_tiled"]["baseline_speedup"], 4 / 1.5)
        self.assertEqual(row["kernels"]["baseline"]["baseline_speedup"], 1)
        self.assertEqual(row["kernels"]["baseline"]["reduction_tile"], 1)

    def test_schema2_rejects_missing_unexpected_and_duplicate_names(self):
        for name in ("other", "baseline", None):
            fixture = self.fixture(2)
            fixture["runs"][0]["results"][-1]["kernel"] = name
            with self.subTest(name=name), self.assertRaises(ValueError): analysis.summarize(fixture)
        fixture = self.fixture(2)
        fixture["runs"][0]["results"].pop()
        with self.assertRaises(ValueError): analysis.summarize(fixture)

    def test_schema2_rejects_wrong_or_bool_launch_metadata(self):
        for key, value in (("block", [16,16]), ("block", [32,True]),
                           ("output_tile", [16,32]), ("reduction_tile", 16),
                           ("outputs_per_thread", True)):
            fixture = self.fixture(2)
            fixture["runs"][0]["results"][-1][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): analysis.summarize(fixture)

    def test_schema2_rejects_wrong_measurement_policy(self):
        for key, value in (("warmup", 3), ("warmup", True), ("kernel_order", "alternating"),
                           ("kernel_names", ["baseline", "tiled"]), ("block", [16,16])):
            fixture = self.fixture(2)
            fixture["runs"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): analysis.summarize(fixture)

    def test_schema2_requires_finite_nonnegative_error(self):
        for invalid in (None, True, -1, float("inf"), float("nan")):
            fixture = self.fixture(2)
            fixture["runs"][0]["results"][-1]["max_abs_error"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): analysis.summarize(fixture)

    def test_requires_matched_counts_and_requested_repetitions(self):
        fixture = self.fixture(2)
        fixture["runs"][0]["results"][-1]["milliseconds"].pop()
        with self.assertRaises(ValueError): analysis.summarize(fixture)
        with self.assertRaises(ValueError): analysis.validate_run(self.fixture(2)["runs"][0], repeats=40)

    def test_rejects_mixed_schemas(self):
        fixture = self.fixture()
        other = self.fixture(2)["runs"][0]
        other["samples"] = 3
        fixture["runs"].append(other)
        with self.assertRaises(ValueError): analysis.summarize(fixture)

    def test_rejects_mixed_repetitions(self):
        fixture = self.fixture(2)
        other = copy.deepcopy(fixture["runs"][0])
        other["samples"] = 3
        for entry in other["results"]:
            entry["milliseconds"].pop()
        fixture["runs"].append(other)
        with self.assertRaises(ValueError): analysis.summarize(fixture)

    def test_rejects_invalid_dimensions_schema_and_metadata(self):
        for key, value in (("samples", True), ("features", 0), ("schema_version", 3),
                           ("schema_version", True), ("gpu", []), ("runtime_version", True)):
            fixture = self.fixture(2)
            fixture["runs"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): analysis.summarize(fixture)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError): analysis.load_document('{"runs":[],"runs":[]}')

    def test_four_variant_svg_labels_and_geometry(self):
        rows = analysis.summarize(self.fixture(2))
        with tempfile.TemporaryDirectory() as temp:
            figure = Path(temp) / "synthetic-unit-only.svg"
            analysis.draw(rows, figure)
            tree = ET.parse(figure)
            ns = {"svg": "http://www.w3.org/2000/svg"}
            bars = tree.findall(".//svg:rect", ns)[1:]
            self.assertEqual(len(bars), 4)
            self.assertTrue(all(float(bar.attrib["width"]) > 0 for bar in bars))
            self.assertIn("register_tiled", figure.read_text())
            self.assertIn("baseline median / candidate median", figure.read_text())
            with self.assertRaises(FileExistsError): analysis.draw(rows, figure)

    def test_retained_schema1_summary_and_svg_are_identical(self):
        retained = Path(__file__).parents[1] / "results/rtx4060-wsl-run-02"
        rows = analysis.summarize(analysis.load_document((retained / "benchmark.json").read_text()))
        self.assertEqual(rows, json.loads((retained / "summary.json").read_text()))
        with tempfile.TemporaryDirectory() as temp:
            figure = Path(temp) / "replayed.svg"
            analysis.draw(rows, figure)
            self.assertEqual(figure.read_bytes(), (retained / "kernel-time.svg").read_bytes())


if __name__ == "__main__": unittest.main()
