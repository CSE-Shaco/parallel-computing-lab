"""Synthetic parser unit tests, never benchmark evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location("analysis", Path(__file__).parents[1]/"benchmarks/analyze_results.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

class AnalysisTest(unittest.TestCase):
    def fixture(self):
        return {"runs": [{"validated": True, "operation": "uncentered_second_moment", "samples": 2, "features": 3,
                         "gpu": "synthetic-test-only", "compute_capability": "0.0", "runtime_version": 0,
                         "driver_version": 0, "cuda_compiler_version": "synthetic", "results": [
                             {"kernel": "baseline", "milliseconds": [2,4,6]},
                             {"kernel": "tiled", "milliseconds": [1,2,3]}]}]}
    def test_median_ratio(self):
        self.assertEqual(analysis.summarize(self.fixture())[0]["speedup"], 2)
    def test_reject_unverified(self):
        fixture=self.fixture(); fixture["runs"][0]["validated"]=False
        with self.assertRaises(ValueError): analysis.summarize(fixture)
    def test_reject_invalid_times(self):
        for invalid in [0,-1,float("nan"),float("inf")]:
            fixture=self.fixture();fixture["runs"][0]["results"][0]["milliseconds"]=[invalid]
            with self.assertRaises(ValueError): analysis.summarize(fixture)
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

if __name__ == "__main__": unittest.main()
