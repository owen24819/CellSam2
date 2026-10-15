"""Shared helpers for the CellSAM2 parallel-throughput benchmarks."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = REPO_ROOT / "inference"

# Make `sam2` and the inference modules (cell_tracker, frame_timer, ...) importable.
for p in (str(REPO_ROOT), str(INFERENCE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def checkpoint_path(model_name: str, checkpoint_num: int | None) -> Path:
    ckpt_dir = REPO_ROOT / "sam2_logs" / model_name / "checkpoints"
    name = "checkpoint.pt" if checkpoint_num is None else f"checkpoint_{checkpoint_num}.pt"
    return ckpt_dir / name


def load_model(model_name: str, checkpoint_num: int | None, device: str = "cuda"):
    """Build a trained CellSAM2 model exactly as inference/track_cells.py does.

    Returns (model, cfg).
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from sam2.build_sam import build_sam2

    config_dir = REPO_ROOT / "sam2_logs" / model_name
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(config_dir), version_base=None)

    cfg = compose(config_name="config_resolved")
    OmegaConf.resolve(cfg)

    ckpt = checkpoint_path(model_name, checkpoint_num)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    model = build_sam2("config_resolved", str(ckpt), device=device, hydra_overrides_extra=[])
    return model, cfg


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def gpu_memory_used_mib(gpu_index: int = 0) -> int:
    """Device-wide used memory (all processes) as reported by nvidia-smi."""
    out = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(out.strip().splitlines()[0])


def gpu_memory_total_mib(gpu_index: int = 0) -> int:
    out = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(out.strip().splitlines()[0])


class GpuMemoryPoller:
    """Background thread that records the peak device-wide GPU memory and utilisation.

    Uses nvidia-smi so the controller process never creates its own CUDA context.
    """

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.25):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.peak_used_mib = 0
        self.util_samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> tuple[int, int]:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={self.gpu_index}",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        ).stdout
        used, util = (int(x) for x in out.strip().splitlines()[0].split(","))
        return used, util

    def _run(self):
        while not self._stop.is_set():
            try:
                used, util = self._sample()
                self.peak_used_mib = max(self.peak_used_mib, used)
                self.util_samples.append(util)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()

    @property
    def mean_util(self) -> float:
        return sum(self.util_samples) / len(self.util_samples) if self.util_samples else float("nan")
