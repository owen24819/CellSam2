#!/usr/bin/env python3
"""How many positions (movies) can CellSAM2 track in parallel on one GPU?

Emulates a multi-position acquisition by running N independent copies of the
full CellSAM2 tracking pipeline (inference/cell_tracker.py) on the SAME sequence,
concurrently on the same GPU (one process per "position"). N is swept upwards
until the GPU runs out of memory (or another worker failure occurs), optionally
followed by a bisection to find the exact maximum N.

Each row of the output CSV records, for one (model, N):
  - wall-clock time for all N movies to finish
  - aggregate throughput (frames/s summed over positions) and per-position fps
  - peak device-wide VRAM (nvidia-smi) and per-worker torch peak reserved memory
  - a VRAM-only projection of the max N (useful when CPU/RAM limits are hit first)

Workers load the model and run a short warm-up before a shared barrier, so model
loading and CUDA/cuDNN initialisation are excluded from the timings.

Example:
    python benchmarks/parallel_positions_benchmark.py \
        --model_name CellSam2-2D-a0p15 --checkpoint_num 20 \
        --sequence data/2D/test/CTC/10 --max_frames 30 \
        --n_list 1 2 4 8 16 32 64 --refine \
        --out_csv results/parallel_bench/parallel_positions.csv
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import (  # noqa: E402
    REPO_ROOT,
    GpuMemoryPoller,
    gpu_memory_total_mib,
    gpu_memory_used_mib,
)

CSV_FIELDS = [
    "model_name",
    "checkpoint_num",
    "sequence",
    "n_parallel",
    "status",
    "frames_per_seq",
    "total_frames",
    "wall_s",
    "aggregate_fps",
    "per_position_fps_mean",
    "worker_s_mean",
    "worker_s_max",
    "baseline_gpu_mib",
    "peak_gpu_mib",
    "gpu_total_mib",
    "vram_per_worker_mib",
    "torch_peak_reserved_mib_max",
    "torch_peak_alloc_mib_max",
    "trim_memory",
    "projected_max_n_vram",
    "mean_gpu_util_pct",
    "min_host_mem_avail_gib",
    "error",
]


# --------------------------------------------------------------------------- #
# Worker (one "position")
# --------------------------------------------------------------------------- #
def _worker(rank: int, args: dict, barrier, result_q) -> None:
    status, err = "ok", ""
    result = {"rank": rank}
    try:
        import torch

        from bench_common import load_model
        from cell_tracker import SAM2AutomaticCellTracker

        model, cfg = load_model(args["model_name"], args["checkpoint_num"])
        model.trim_memory_history = args["trim_memory"]
        resize_threshold = args["resize_threshold"]
        if resize_threshold is None:
            resize_threshold = getattr(cfg.scratch, "resize_threshold", None)

        # Same settings as inference/track_cells.py defaults.
        tracker = SAM2AutomaticCellTracker(
            model,
            pred_iou_thresh=args["pred_iou_thresh"],
            obj_score_thresh=args["obj_score_thresh"],
            div_obj_score_thresh=args["div_obj_score_thresh"],
            box_nms_thresh=args["box_nms_thresh"],
            segment=False,
            use_heatmap=True,
            resize_threshold=resize_threshold,
            crop_overlap=args["crop_overlap"],
            aux_matching=args["aux_matching"],
        )

        seq = Path(args["sequence"])
        out_dir = Path(args["run_out_dir"]) / f"w{rank:03d}"
        warm_dir = out_dir.with_name(out_dir.name + "_warmup")
        out_dir.mkdir(parents=True, exist_ok=True)
        warm_dir.mkdir(parents=True, exist_ok=True)

        predict_kwargs = dict(
            offload_video_to_cpu=args["offload_video_to_cpu"],
            offload_state_to_cpu=args["offload_state_to_cpu"],
        )

        # Warm-up (cuDNN autotune, allocator growth, lazy init) on a few frames.
        tracker.predict(Path(args["warmup_sequence"]), warm_dir, **predict_kwargs)
        torch.cuda.synchronize()
        shutil.rmtree(warm_dir, ignore_errors=True)
        torch.cuda.reset_peak_memory_stats()

        barrier.wait(timeout=args["barrier_timeout_s"])

        t0 = time.time()
        frame_metrics_csv = None
        if args["frame_metrics_dir"] is not None:
            frame_metrics_csv = (Path(args["frame_metrics_dir"])
                                 / f"{args['model_name']}_n{args['n']:03d}_w{rank:03d}.csv")
        tracker.predict(
            seq,
            out_dir,
            frame_metrics_csv=frame_metrics_csv,
            gt_seg_dir=None if args["gt_dir"] is None else Path(args["gt_dir"]),
            **predict_kwargs,
        )
        torch.cuda.synchronize()
        t1 = time.time()

        result.update(
            t_start=t0,
            t_end=t1,
            elapsed_s=t1 - t0,
            torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            torch_peak_alloc_mib=torch.cuda.max_memory_allocated() / 2**20,
        )
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        status = "oom" if "out of memory" in msg.lower() or "OutOfMemory" in msg else "error"
        err = msg.splitlines()[0][:300]
        traceback.print_exc()
        try:
            barrier.abort()  # release the other workers instead of hanging
        except Exception:
            pass
    result["status"] = status
    result["error"] = err
    result_q.put(result)


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
def _frame_paths(seq: Path) -> list[Path]:
    return sorted(p for p in seq.iterdir() if p.suffix.lower() in {".tif", ".tiff"})


def _truncated_sequence(seq: Path, k: int, dest: Path) -> Path:
    """Directory of symlinks to the first k frames (the tracker always runs every frame)."""
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for f in _frame_paths(seq)[:k]:
        (dest / f.name).symlink_to(f.resolve())
    return dest


def _host_mem_avail_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return float("nan")


def run_once(n: int, args: argparse.Namespace, gpu_total: int) -> dict:
    seq = Path(args.sequence).resolve()
    num_frames = len(_frame_paths(seq))
    frames_per_seq = num_frames if args.max_frames is None else min(args.max_frames, num_frames)

    run_out_dir = Path(args.out_dir) / args.model_name / f"n{n:03d}"
    shutil.rmtree(run_out_dir, ignore_errors=True)
    run_out_dir.mkdir(parents=True, exist_ok=True)
    seq_dir = Path(args.out_dir) / "_sequences"
    warmup_seq = _truncated_sequence(seq, args.warmup_frames, seq_dir / "warmup")
    if frames_per_seq < num_frames:
        seq = _truncated_sequence(seq, frames_per_seq, seq_dir / f"first{frames_per_seq}")

    worker_args = dict(
        model_name=args.model_name,
        checkpoint_num=args.checkpoint_num,
        sequence=str(seq),
        warmup_sequence=str(warmup_seq),
        run_out_dir=str(run_out_dir),
        resize_threshold=args.resize_threshold,
        crop_overlap=args.crop_overlap,
        pred_iou_thresh=args.pred_iou_thresh,
        obj_score_thresh=args.obj_score_thresh,
        div_obj_score_thresh=args.div_obj_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        aux_matching=args.aux_matching,
        offload_video_to_cpu=not args.no_offload_video,
        offload_state_to_cpu=not args.no_offload_state,
        barrier_timeout_s=args.barrier_timeout_s,
        trim_memory=args.trim_memory,
        n=n,
        frame_metrics_dir=None,
        gt_dir=None,
    )
    if args.frame_metrics:
        fm_dir = Path(args.out_csv).parent / "frame_metrics"
        fm_dir.mkdir(parents=True, exist_ok=True)
        worker_args["frame_metrics_dir"] = str(fm_dir.resolve())
        gt_dir = Path(args.gt_dir) if args.gt_dir else Path(args.sequence).parent / f"{Path(args.sequence).name}_GT" / "TRA"
        worker_args["gt_dir"] = str(gt_dir.resolve()) if gt_dir.is_dir() else None

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(n)
    result_q = ctx.Queue()

    baseline = gpu_memory_used_mib(args.gpu)
    poller = GpuMemoryPoller(args.gpu).start()
    min_host_avail = _host_mem_avail_gib()

    procs = [ctx.Process(target=_worker, args=(r, worker_args, barrier, result_q)) for r in range(n)]
    for p in procs:
        p.start()

    results = []
    host_oom = False
    deadline = time.time() + args.run_timeout_s
    while len(results) < n and time.time() < deadline:
        try:
            results.append(result_q.get(timeout=1.0))
        except Exception:
            min_host_avail = min(min_host_avail, _host_mem_avail_gib())
            if min_host_avail < args.min_host_mem_gib:
                host_oom = True  # stop before the kernel OOM killer picks a victim
                for p in procs:
                    if p.is_alive():
                        p.kill()
                break
            # A worker killed outright (e.g. host OOM killer) never reports back.
            dead = [p for p in procs if p.exitcode not in (None, 0)]
            if dead and len(results) + len(dead) >= n:
                break
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    poller.stop()

    statuses = [r["status"] for r in results]
    missing = n - len(results)
    if missing:
        statuses += ["killed"] * missing
    if host_oom:
        status = "host_ram"
    elif all(s == "ok" for s in statuses):
        status = "ok"
    elif "oom" in statuses:
        status = "oom"
    else:
        status = next(s for s in statuses if s != "ok")
    errors = sorted({r["error"] for r in results if r["error"]})
    if host_oom:
        errors.append(f"aborted: host MemAvailable < {args.min_host_mem_gib} GiB")
    elif missing:
        errors.append(f"{missing} worker(s) died without reporting (host OOM killer / timeout?)")

    row = {k: "" for k in CSV_FIELDS}
    row.update(
        model_name=args.model_name,
        checkpoint_num=args.checkpoint_num,
        sequence=str(Path(args.sequence)),
        n_parallel=n,
        status=status,
        frames_per_seq=frames_per_seq,
        baseline_gpu_mib=baseline,
        peak_gpu_mib=poller.peak_used_mib,
        gpu_total_mib=gpu_total,
        trim_memory=args.trim_memory,
        mean_gpu_util_pct=f"{poller.mean_util:.1f}",
        min_host_mem_avail_gib=f"{min_host_avail:.1f}",
        error=" | ".join(errors)[:500],
    )

    vram_per_worker = (poller.peak_used_mib - baseline) / n if n else float("nan")
    if vram_per_worker > 0:
        row["vram_per_worker_mib"] = f"{vram_per_worker:.0f}"
        usable = (gpu_total - baseline) * args.vram_headroom
        row["projected_max_n_vram"] = int(usable // vram_per_worker)

    ok = [r for r in results if r["status"] == "ok"]
    if status == "ok" and ok:
        wall = max(r["t_end"] for r in ok) - min(r["t_start"] for r in ok)
        elapsed = [r["elapsed_s"] for r in ok]
        total_frames = frames_per_seq * n
        row.update(
            total_frames=total_frames,
            wall_s=f"{wall:.3f}",
            aggregate_fps=f"{total_frames / wall:.3f}",
            per_position_fps_mean=f"{sum(frames_per_seq / e for e in elapsed) / len(elapsed):.3f}",
            worker_s_mean=f"{sum(elapsed) / len(elapsed):.3f}",
            worker_s_max=f"{max(elapsed):.3f}",
            torch_peak_reserved_mib_max=f"{max(r['torch_peak_reserved_mib'] for r in ok):.0f}",
            torch_peak_alloc_mib_max=f"{max(r['torch_peak_alloc_mib'] for r in ok):.0f}",
        )

    if not args.keep_outputs:
        shutil.rmtree(run_out_dir, ignore_errors=True)
    return row


def append_row(csv_path: Path, row: dict) -> None:
    new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True, help="Run name under sam2_logs/, e.g. CellSam2-2D-a0p15")
    p.add_argument("--checkpoint_num", type=int, default=20)
    p.add_argument("--sequence", default="data/2D/test/CTC/10",
                   help="Sequence (dir of .tif frames) that every 'position' processes")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Only track the first K frames of the sequence (shorter sweeps)")
    p.add_argument("--warmup_frames", type=int, default=3,
                   help="Frames tracked in the untimed warm-up run of each worker")
    p.add_argument("--n_list", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64],
                   help="Numbers of parallel positions to try (ascending; sweep stops at first failure)")
    p.add_argument("--refine", action="store_true",
                   help="After the first failure, bisect between last success and first failure")
    p.add_argument("--max_cpu_workers", type=int, default=None,
                   help="Skip N values above this (e.g. to stay within host RAM / CPU cores)")
    p.add_argument("--out_csv", default="results/parallel_bench/parallel_positions.csv")
    p.add_argument("--out_dir", default="results/parallel_bench/outputs",
                   help="Where each worker writes its CTC outputs (deleted unless --keep_outputs)")
    p.add_argument("--keep_outputs", action="store_true")
    p.add_argument("--gpu", type=int, default=0, help="GPU index (also set CUDA_VISIBLE_DEVICES to match)")
    p.add_argument("--vram_headroom", type=float, default=0.95,
                   help="Fraction of free VRAM used for the projected max N")
    p.add_argument("--threads_per_worker", type=int, default=1,
                   help="OMP/MKL threads per worker (avoids CPU oversubscription)")
    p.add_argument("--min_host_mem_gib", type=float, default=6.0,
                   help="Abort a run (status host_ram) if host MemAvailable drops below this")
    p.add_argument("--barrier_timeout_s", type=float, default=1800)
    p.add_argument("--run_timeout_s", type=float, default=4 * 3600)
    # Tracker settings (defaults match inference/track_cells.py)
    p.add_argument("--resize_threshold", type=int, default=None)
    p.add_argument("--crop_overlap", type=int, default=128)
    p.add_argument("--pred_iou_thresh", type=float, default=0.7)
    p.add_argument("--obj_score_thresh", type=float, default=0.5)
    p.add_argument("--div_obj_score_thresh", type=float, default=0.4)
    p.add_argument("--box_nms_thresh", type=float, default=0.5)
    p.add_argument("--aux_matching", default="off",
                   choices=["off", "new_cells_only", "segment_then_aux_track"])
    p.add_argument("--frame_metrics", action="store_true",
                   help="Write per-frame time + detected/GT cell counts per worker to "
                        "<out_csv dir>/frame_metrics/ (uses the tracker's FrameMetricsRecorder)")
    p.add_argument("--gt_dir", default=None,
                   help="GT dir for per-frame GT cell counts (default: <sequence>_GT/TRA)")
    p.add_argument("--trim_memory", action="store_true",
                   help="Keep only the last num_maskmem entries of each cell's memory history "
                        "(model.trim_memory_history; outputs unchanged, see verify_memory_trim.py)")
    p.add_argument("--no_offload_video", action="store_true",
                   help="Keep video frames on GPU (track_cells.py offloads them to CPU)")
    p.add_argument("--no_offload_state", action="store_true",
                   help="Keep inference state on GPU (track_cells.py offloads it to CPU)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)  # relative data/result paths resolve against the repo, like track_cells.py

    # Inherited by the spawned workers.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(args.threads_per_worker)
    os.environ.setdefault("PYTHONPATH", str(REPO_ROOT))

    gpu_total = gpu_memory_total_mib(args.gpu)
    csv_path = Path(args.out_csv)
    n_list = sorted(set(args.n_list))
    if args.max_cpu_workers is not None:
        n_list = [n for n in n_list if n <= args.max_cpu_workers]

    print(f"[bench] model={args.model_name} ckpt={args.checkpoint_num} seq={args.sequence} "
          f"max_frames={args.max_frames} GPU total={gpu_total} MiB")

    last_ok, first_fail = 0, None
    for n in n_list:
        print(f"[bench] N={n} ...", flush=True)
        row = run_once(n, args, gpu_total)
        append_row(csv_path, row)
        print(f"[bench]   status={row['status']} wall={row['wall_s']}s agg_fps={row['aggregate_fps']} "
              f"peak_vram={row['peak_gpu_mib']}MiB per_worker={row['vram_per_worker_mib']}MiB "
              f"util={row['mean_gpu_util_pct']}% {row['error']}", flush=True)
        if row["status"] == "ok":
            last_ok = n
        else:
            first_fail = n
            break

    if args.refine and first_fail is not None and first_fail - last_ok > 1:
        lo, hi = last_ok, first_fail
        while hi - lo > 1:
            mid = (lo + hi) // 2
            print(f"[bench] refine N={mid} (ok<={lo}, fail>={hi}) ...", flush=True)
            row = run_once(mid, args, gpu_total)
            append_row(csv_path, row)
            print(f"[bench]   status={row['status']} agg_fps={row['aggregate_fps']} "
                  f"peak_vram={row['peak_gpu_mib']}MiB", flush=True)
            if row["status"] == "ok":
                lo = mid
            else:
                hi = mid
        last_ok, first_fail = lo, hi

    print(f"[bench] {args.model_name}: max parallel positions that succeeded = {last_ok}"
          + (f" (first failure at N={first_fail})" if first_fail else " (no failure within n_list)"))
    print(f"[bench] results appended to {csv_path}")


if __name__ == "__main__":
    main()
