import os
import tempfile
import unittest

from app.tracking import DatasetRegistry, ExperimentTracker


class TestTracking(unittest.TestCase):
    def test_tracker_writes_jsonl_and_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(base_dir=tmpdir)
            tracker.log_run({"run_type": "rag_query", "query": "q1", "latency_seconds": 0.12})
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "runs.jsonl")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "runs.csv")))

    def test_dataset_registry_writes_manifest_and_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = DatasetRegistry(base_dir=tmpdir)
            registry.register({"source_file_count": 2, "chunk_size": 800})
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "manifest.jsonl")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "ingestions.csv")))


if __name__ == "__main__":
    unittest.main()
