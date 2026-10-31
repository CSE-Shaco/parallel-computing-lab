"""Integrity and preservation tests. Synthetic times are never benchmark evidence."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from benchmark_data import jobs, load_run, source_hashes
from analyze_results import write_analysis


def fixture():
    meta = {"argv": {"samples": 10, "repeats": 2, "items": 10, "light": 2, "heavy": 4},
            "missing": [], "binary_sha256": {"pi_mc": "a" * 64, "scheduling": "b" * 64}}
    rows = []
    for repeat in range(2):
        for name, command in jobs(meta["argv"]):
            if name == "pi_mc":
                mode, t, n, seed = command
                row = dict(mode=mode, threads=int(t), samples=int(n), seed=int(seed), count=5, pi=2.0)
            else:
                policy, t, chunk, n, light, heavy, layout = command
                workers = [{"items": c, "iterations": c * 2, "active_ms": 0.1} for c in [3, 3, 2, 2]]
                if layout == "tail":
                    workers[-1]["iterations"] += 2
                row = dict(policy=policy, threads=int(t), chunk=int(chunk), items=int(n), light=int(light),
                           heavy=int(heavy), layout=layout, actual_threads=4, workers=workers,
                           verified=True, checksum=123 if layout == "uniform" else 456)
            row.update(experiment=name, repeat=repeat, ms=1.0 + repeat * 2)
            rows.append(row)
    return meta, rows


def write_run(path, meta, rows):
    path.mkdir()
    (path / "metadata.json").write_text(json.dumps(meta) + "\n")
    (path / "raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="benchmark-contract-test-")
        self.base = Path(self.temp.name)
        self.run = self.base / "input"
        self.meta, self.rows = fixture()

    def tearDown(self):
        self.temp.cleanup()

    def load(self, rows=None, meta=None):
        write_run(self.run, self.meta if meta is None else meta, self.rows if rows is None else rows)
        return load_run(self.run)

    def test_legacy_complete_matrix_and_statistics(self):
        meta, rows, groups, summary = self.load()
        self.assertEqual((len(rows), len(groups)), (66, 33))
        self.assertEqual(summary["pi/serial/1 threads"], {"runs": 2, "median_ms": 2.0, "min_ms": 1.0, "max_ms": 3.0})

    def test_schema2_complete(self):
        self.meta.update(schema_version=2, status="complete", completed_records=66, configuration_count=33)
        self.assertEqual(len(self.load()[1]), 66)

    def test_schema2_completion_fields_required_and_typed(self):
        for field, value in [("status", None), ("configuration_count", 33.0), ("completed_records", 66.0)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                meta = copy.deepcopy(self.meta)
                meta.update(schema_version=2, status="complete", completed_records=66, configuration_count=33)
                if value is None: meta.pop(field)
                else: meta[field] = value
                path = Path(tmp) / "run"; write_run(path, meta, self.rows)
                with self.assertRaises(ValueError): load_run(path)

    def test_optional_openmp_missing_is_explicit(self):
        self.meta["missing"] = ["scheduling"]
        self.meta["binary_sha256"].pop("scheduling")
        self.rows = [row for row in self.rows if row["experiment"] == "pi_mc"]
        self.load(); out = self.base / "analysis"
        result = write_analysis(self.run, out)
        self.assertEqual((result["records"], result["configurations"]), (26, 13))
        self.assertFalse((out / "openmp-uniform.svg").exists())

    def test_invalid_times(self):
        for value in [0, -1, float("nan"), float("inf"), -float("inf"), True, "1"]:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                rows = copy.deepcopy(self.rows); rows[0]["ms"] = value
                path = Path(tmp) / "run"; write_run(path, self.meta, rows)
                with self.assertRaises(ValueError): load_run(path)

    def test_missing_repeat(self):
        with self.assertRaises(ValueError): self.load(self.rows[:-1])

    def test_duplicate_repeat(self):
        with self.assertRaises(ValueError): self.load(self.rows + [self.rows[0]])

    def test_mixed_input_sizes(self):
        self.rows[0]["samples"] = 20; self.rows[0]["pi"] = 1.0
        with self.assertRaises(ValueError): self.load()

    def test_unknown_experiment(self):
        self.rows[0]["experiment"] = "gpu"
        with self.assertRaises(ValueError): self.load()

    def test_unverified_openmp(self):
        self.rows[13]["verified"] = False
        with self.assertRaises(ValueError): self.load()

    def test_worker_totals(self):
        self.rows[13]["workers"][0]["items"] = 2
        with self.assertRaises(ValueError): self.load()

    def test_wrong_worker_costs(self):
        self.rows[23]["workers"][0]["iterations"] += 1
        with self.assertRaises(ValueError): self.load()

    def test_inconsistent_monte_carlo_results(self):
        self.rows[0].update(count=6, pi=2.4)
        with self.assertRaises(ValueError): self.load()

    def test_inconsistent_openmp_checksum(self):
        self.rows[13]["checksum"] += 1
        with self.assertRaises(ValueError): self.load()

    def test_bool_is_not_integer(self):
        self.rows[0]["threads"] = True
        with self.assertRaises(ValueError): self.load()

    def test_incomplete_status(self):
        self.meta.update(schema_version=2, status="failed", completed_records=66, configuration_count=33)
        with self.assertRaises(ValueError): self.load()

    def test_bad_record_count(self):
        self.meta.update(schema_version=2, status="complete", completed_records=65, configuration_count=33)
        with self.assertRaises(ValueError): self.load()

    def test_duplicate_json_field(self):
        write_run(self.run, self.meta, self.rows)
        raw = (self.run / "raw.jsonl").read_text().replace('"ms": 1.0', '"ms": 1.0, "ms": 2.0', 1)
        (self.run / "raw.jsonl").write_text(raw)
        with self.assertRaises(ValueError): load_run(self.run)

    def test_stale_recorded_summary(self):
        self.load()
        (self.run / "summary.json").write_text('{"incorrect": {}}')
        with self.assertRaises(ValueError): load_run(self.run)

    def test_existing_output_preserved(self):
        self.load(); out = self.base / "analysis"; out.mkdir(); (out / "sentinel").write_text("retain")
        with self.assertRaises(ValueError): write_analysis(self.run, out)
        self.assertEqual((out / "sentinel").read_text(), "retain")
        self.assertEqual(len(list(out.iterdir())), 1)

    def test_invalid_input_creates_no_analysis(self):
        self.rows.pop(); write_run(self.run, self.meta, self.rows); out = self.base / "analysis"
        with self.assertRaises(ValueError): write_analysis(self.run, out)
        self.assertFalse(out.exists())

    def test_output_statistics_svg_and_source_immutability(self):
        self.load(); hashes = source_hashes(self.run); out = self.base / "analysis"
        result = write_analysis(self.run, out)
        self.assertEqual((result["records"], result["configurations"]), (66, 33))
        self.assertEqual(hashes, source_hashes(self.run))
        for name in ["monte-carlo-scaling.svg", "openmp-uniform.svg", "openmp-tail.svg"]:
            svg = ET.parse(out / name).getroot()
            self.assertTrue(svg.tag.endswith("svg"))
            self.assertEqual(svg.attrib["viewBox"], "0 0 1100 500")
        with (out / "summary.csv").open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 33)
        self.assertEqual(float(rows[0]["relative_time_to_serial"]), 1)
        tail = next(row for row in rows if row["layout"] == "tail")
        self.assertAlmostEqual(float(tail["assigned_work_max_over_mean_median"]), 6 / 5.5)

    def test_actual_retained_run(self):
        run = SCRIPTS.parent / "results/macos-arm64-smoke-2026-10-31-v3"
        hashes = source_hashes(run); _, rows, groups, summary = load_run(run)
        self.assertEqual((len(rows), len(groups)), (99, 33))
        self.assertEqual(summary, json.loads((run / "summary.json").read_text()))
        self.assertEqual(hashes, source_hashes(run))


TOY = r'''#!PYTHON
import json, sys
from pathlib import Path
counter = Path(COUNTER)
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
if FAIL_AFTER and count > FAIL_AFTER:
    sys.stderr.write("synthetic executable failure\n"); sys.exit(7)
a = sys.argv[1:]
if a[0] in ["serial", "local", "mutex"]:
    mode, t, n, seed = a; n=int(n); hit=n//2
    row=dict(mode=mode,threads=int(t),samples=n,seed=int(seed),count=hit,pi=4.0*hit/n)
else:
    policy,t,chunk,n,light,heavy,layout=a; n=int(n); light=int(light); heavy=int(heavy)
    counts=[n//4+(i<n%4) for i in range(4)]
    workers=[dict(items=c,iterations=c*light,active_ms=0.1) for c in counts]
    if layout=="tail":
        remaining=max(1,n//10)
        for worker in workers:
            add=min(remaining,worker["items"]);worker["iterations"]+=add*(heavy-light);remaining-=add
    row=dict(policy=policy,threads=int(t),chunk=int(chunk),items=n,light=light,heavy=heavy,layout=layout,
             actual_threads=4,workers=workers,checksum=123 if layout=="uniform" else 456,verified=True)
row["ms"]=1.0
print(json.dumps(row))
'''


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="benchmark-runner-test-")
        self.base = Path(self.temp.name); self.build = self.base / "bin"; self.build.mkdir()
        self.out = self.base / "run"

    def tearDown(self): self.temp.cleanup()

    def invoke(self, *extra):
        return subprocess.run([sys.executable, str(SCRIPTS / "run_benchmarks.py"), "--build", str(self.build),
                               "--out", str(self.out), "--samples", "10", "--items", "10", "--light", "2",
                               "--heavy", "4", "--repeats", "1", *extra], capture_output=True, text=True,
                              env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=30)

    def toy(self, fail_after=0):
        text = TOY.replace("#!PYTHON", "#!" + sys.executable).replace("COUNTER", repr(str(self.base / "counter"))).replace("FAIL_AFTER", str(fail_after))
        for name in ["pi_mc", "scheduling"]:
            p=self.build/name; p.write_text(text); p.chmod(0o755)

    def test_preflight_does_not_create_empty_run(self):
        self.assertNotEqual(self.invoke().returncode, 0)
        self.assertFalse(self.out.exists())

    def test_invalid_timeout_preflight(self):
        self.toy(); self.assertNotEqual(self.invoke("--timeout", "nan").returncode, 0)
        self.assertFalse(self.out.exists())

    def test_complete_synthetic_pipeline_and_no_overwrite(self):
        self.toy(); result=self.invoke(); self.assertEqual(result.returncode, 0, result.stderr)
        meta, rows, groups, summary=load_run(self.out)
        self.assertEqual((meta["status"], len(rows), len(groups)), ("complete", 33, 33))
        old=source_hashes(self.out); again=self.invoke()
        self.assertNotEqual(again.returncode, 0); self.assertEqual(old, source_hashes(self.out))

    def test_failure_preserves_partial_evidence(self):
        self.toy(fail_after=35); result=self.invoke()
        self.assertEqual(result.returncode, 1)
        meta=json.loads((self.out/"metadata.json").read_text())
        self.assertEqual(meta["status"], "failed"); self.assertEqual(meta["completed_records"], 2)
        self.assertEqual(len((self.out/"raw.jsonl").read_text().splitlines()), 2)
        self.assertIn("synthetic executable failure", (self.out / "failure.log").read_text())
        with self.assertRaises(ValueError): load_run(self.out)

    def test_timeout_records_failed_status(self):
        for name in ["pi_mc", "scheduling"]:
            p=self.build/name; p.write_text("#!"+sys.executable+"\nimport time\ntime.sleep(2)\n");p.chmod(0o755)
        result=self.invoke("--timeout", "0.05")
        self.assertEqual(result.returncode, 1)
        meta=json.loads((self.out/"metadata.json").read_text())
        self.assertEqual(meta["status"], "failed")
        self.assertEqual(meta["failure"]["type"], "TimeoutExpired")


if __name__ == "__main__": unittest.main()
