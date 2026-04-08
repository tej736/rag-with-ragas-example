import csv
import json
import os
from datetime import datetime, timezone
from uuid import uuid4


class ExperimentTracker:
    def __init__(self, base_dir: str = "app/output/experiments"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.base_dir, "runs.jsonl")
        self.csv_path = os.path.join(self.base_dir, "runs.csv")

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def log_run(self, payload: dict) -> dict:
        run = {
            "run_id": payload.get("run_id") or str(uuid4()),
            "timestamp": payload.get("timestamp") or self._timestamp(),
            **payload,
        }

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(run, ensure_ascii=False) + "\n")

        self._append_csv(run)
        return run

    def _append_csv(self, run: dict) -> None:
        fieldnames = sorted(run.keys())
        file_exists = os.path.exists(self.csv_path)

        if file_exists:
            with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                existing_fields = reader.fieldnames or []
            fieldnames = sorted(set(existing_fields).union(run.keys()))
            if set(fieldnames) != set(existing_fields):
                self._rewrite_csv_with_new_fields(fieldnames)

        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: run.get(k, "") for k in fieldnames})

    def _rewrite_csv_with_new_fields(self, fieldnames: list[str]) -> None:
        rows = []
        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})


class DatasetRegistry:
    def __init__(self, base_dir: str = "app/output/datasets"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.base_dir, "manifest.jsonl")
        self.csv_path = os.path.join(self.base_dir, "ingestions.csv")

    def register(self, payload: dict) -> dict:
        record = {
            "dataset_version": payload.get("dataset_version") or str(uuid4())[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }

        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._append_csv(record)
        return record

    def _append_csv(self, record: dict) -> None:
        file_exists = os.path.exists(self.csv_path)
        fieldnames = sorted(record.keys())

        if file_exists:
            with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                existing_fields = reader.fieldnames or []
            fieldnames = sorted(set(existing_fields).union(record.keys()))

        rows = []
        if file_exists:
            with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
            writer.writerow({k: record.get(k, "") for k in fieldnames})
