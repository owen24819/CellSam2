from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter


class SequenceTimer:
    def __init__(self, output_csv):
        self.output_csv = Path(output_csv)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        self.file = open(self.output_csv, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "sequence",
            "num_frames",
            "total_seconds",
            "fps",
        ])

        self.start_time = None

    def start(self):
        
        self.start_time = perf_counter()

    def stop(self, sequence_name, num_frames):
        if self.start_time is None:
            raise RuntimeError("Timer was never started.")

        elapsed = perf_counter() - self.start_time
        fps = num_frames / elapsed if elapsed > 0 else float("inf")

        self.writer.writerow([
            sequence_name,
            num_frames,
            f"{elapsed:.6f}",
            f"{fps:.6f}",
        ])
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()