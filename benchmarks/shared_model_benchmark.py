#!/usr/bin/env python3
"""N positions tracked in parallel by ONE shared CellSAM2 model (threads, one process).

Counterpart to parallel_positions_benchmark.py (one process + one model copy per
position). Here a single process loads the model once and N threads each run their
own SAM2AutomaticCellTracker on a copy of the same movie. Every thread gets its own
CUDA stream so GPU work from different positions can overlap. This removes the
per-process CUDA context and duplicated weights, so the VRAM cost per extra
position is only its activations / inference state.

Caveat: the tracker's Python/NumPy/OpenCV post-processing shares one GIL, so
CPU-bound parts do not run fully in parallel (compare with the multi-process mode).

Each N runs in a fresh subprocess so allocator state / OOMs never leak between runs.
Results use the same CSV columns as parallel_positions.csv plus `mode`.

Example:
    python benchmarks/shared_model_benchmark.py \
        --model_name CellSam2-2D-small-a0p15 --checkpoint_num 20 \
        --sequence data/2D/test/CTC/10 --max_frames 30 \
        --n_list 1 2 4 8 16 32 64 128 --refine \
        --out_csv results/parallel_bench/shared_model.csv
"""
from __future__ import annotations

import argparse
import contextlib
import multiprocessing as mp
import os
import shutil
import sys
import threading
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
from parallel_positions_benchmark import (  # noqa: E402
    CSV_FIELDS as _BASE_FIELDS,
    _frame_paths,
    _host_mem_avail_gib,
    _truncated_sequence,
)

CSV_FIELDS = ["mode", *_BASE_FIELDS, "model_weights_mib"]


# --------------------------------------------------------------------------- #
# Subprocess: one model, N threads
# --------------------------------------------------------------------------- #
def _shared_process(args: dict, result_q) -> None:
    out = {"status": "ok", "error": "", "threads": []}
    try:
        import torch

        from bench_common import load_model
        from cell_tracker import SAM2AutomaticCellTracker

        torch.cuda.reset_peak_memory_stats()
        model, cfg = load_model(args["model_name"], args["checkpoint_num"])
        model.trim_memory_history = args["trim_memory"]
        torch.cuda.synchronize()
        out["model_weights_mib"] = torch.cuda.memory_allocated() / 2**20

        resize_threshold = args["resize_threshold"]
        if resize_threshold is None:
            resize_threshold = getattr(cfg.scratch, "resize_threshold", None)

        n = args["n"]
        # One tracker per position (per-call state lives in local inference_state dicts;
        # the tracker only writes identical threshold values onto the shared model).
        trackers = [
            SAM2AutomaticCellTracker(
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
            for _ in range(n)
        ]
        streams = [torch.cuda.Stream() for _ in range(n)] if args["use_streams"] else [None] * n

        predict_kwargs = dict(
            offload_video_to_cpu=args["offload_video_to_cpu"],
            offload_state_to_cpu=args["offload_state_to_cpu"],
        )
        seq = Path(args["sequence"])
        warm_seq = Path(args["warmup_sequence"])
        run_out_dir = Path(args["run_out_dir"])
        gt_dir = None if args["gt_dir"] is None else Path(args["gt_dir"])

        # Peak-memory stats are reset once every thread has finished its warm-up.
        barrier = threading.Barrier(n, action=torch.cuda.reset_peak_memory_stats)
        thread_results: list[dict] = [{} for _ in range(n)]

        def run(i: int) -> None:
            res = thread_results[i]
            stream = streams[i]
            stream_ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            sync = stream.synchronize if stream is not None else torch.cuda.synchronize
            out_dir = run_out_dir / f"t{i:03d}"
            warm_dir = run_out_dir / f"t{i:03d}_warmup"
            out_dir.mkdir(parents=True, exist_ok=True)
            warm_dir.mkdir(parents=True, exist_ok=True)
            try:
                # grad mode is thread-local: new threads start with grad enabled
                with torch.inference_mode(), stream_ctx:
                    trackers[i].predict(warm_seq, warm_dir, **predict_kwargs)
                    sync()
                    shutil.rmtree(warm_dir, ignore_errors=True)

                    barrier.wait(timeout=args["barrier_timeout_s"])

                    frame_metrics_csv = None
                    if args["frame_metrics_dir"] is not None:
                        frame_metrics_csv = (Path(args["frame_metrics_dir"])
                                             / f"{args['model_name']}_shared_n{n:03d}_t{i:03d}.csv")
                    t0 = time.time()
                    trackers[i].predict(seq, out_dir, frame_metrics_csv=frame_metrics_csv,
                                        gt_seg_dir=gt_dir, **predict_kwargs)
                    sync()
                    t1 = time.time()
                res.update(status="ok", t_start=t0, t_end=t1, elapsed_s=t1 - t0)
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                res.update(
                    status="oom" if "out of memory" in msg.lower() else "error",
                    error=msg.splitlines()[0][:300],
                )
                traceback.print_exc()
                barrier.abort()

        threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        out["threads"] = thread_results
        out["torch_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        out["torch_peak_alloc_mib"] = torch.cuda.max_memory_allocated() / 2**20
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        out["status"] = "oom" if "out of memory" in msg.lower() else "error"
        out["error"] = msg.splitlines()[0][:300]
        traceback.print_exc()
    result_q.put(out)


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
def run_once(n: int, args: argparse.Namespace, gpu_total: int) -> dict:
    seq = Path(args.sequence).resolve()
    num_frames = len(_frame_paths(seq))
    frames_per_seq = num_frames if args.max_frames is None else min(args.max_frames, num_frames)

    run_out_dir = Path(args.out_dir) / f"{args.model_name}_shared" / f"n{n:03d}"
    shutil.rmtree(run_out_dir, ignore_errors=True)
    run_out_dir.mkdir(parents=True, exist_ok=True)
    seq_dir = Path(args.out_dir) / "_sequences"
    warmup_seq = _truncated_sequence(seq, args.warmup_frames, seq_dir / "warmup")
    if frames_per_seq < num_frames:
        seq = _truncated_sequence(seq, frames_per_seq, seq_dir / f"first{frames_per_seq}")

    proc_args = dict(
        n=n,
        model_name=args.model_name,
        checkpoint_num=args.checkpoint_num,
        sequence=str(seq),
        warmup_sequence=str(warmup_seq),
        run_out_dir=str(run_out_dir),
        use_streams=not args.no_streams,
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
        frame_metrics_dir=None,
        gt_dir=None,
    )
    if args.frame_metrics:
        fm_dir = Path(args.out_csv).parent / "frame_metrics"
        fm_dir.mkdir(parents=True, exist_ok=True)
        proc_args["frame_metrics_dir"] = str(fm_dir.resolve())
        gt_dir = Path(args.gt_dir) if args.gt_dir else Path(args.sequence).parent / f"{Path(args.sequence).name}_GT" / "TRA"
        proc_args["gt_dir"] = str(gt_dir.resolve()) if gt_dir.is_dir() else None

    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    baseline = gpu_memory_used_mib(args.gpu)
    poller = GpuMemoryPoller(args.gpu).start()
    min_host_avail = _host_mem_avail_gib()

    proc = ctx.Process(target=_shared_process, args=(proc_args, result_q))
    proc.start()
    out = None
    deadline = time.time() + args.run_timeout_s
    while out is None and time.time() < deadline:
        try:
            out = result_q.get(timeout=1.0)
        except Exception:
            min_host_avail = min(min_host_avail, _host_mem_avail_gib())
            if min_host_avail < args.min_host_mem_gib:
                proc.kill()  # stop before the kernel OOM killer picks a victim
                out = {"status": "host_ram", "threads": [],
                       "error": f"aborted: host MemAvailable < {args.min_host_mem_gib} GiB"}
                break
            if proc.exitcode is not None:
                break
    proc.join(timeout=30)
    if proc.is_alive():
        proc.terminate()
    poller.stop()

    if out is None:
        out = {"status": "killed", "error": "process died without reporting (host OOM killer / timeout?)",
               "threads": []}

    thread_statuses = [t.get("status", "error") for t in out["threads"]]
    if out["status"] != "ok":
        status = out["status"]
    elif all(s == "ok" for s in thread_statuses) and len(thread_statuses) == n:
        status = "ok"
    elif "oom" in thread_statuses:
        status = "oom"
    else:
        status = "error"
    errors = sorted({e for e in [out.get("error", "")] + [t.get("error", "") for t in out["threads"]] if e})

    row = {k: "" for k in CSV_FIELDS}
    row.update(
        mode="shared_threads" if not args.no_streams else "shared_threads_default_stream",
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
    if "model_weights_mib" in out:
        row["model_weights_mib"] = f"{out['model_weights_mib']:.0f}"

    vram_per_pos = (poller.peak_used_mib - baseline) / n
    if vram_per_pos > 0:
        # Conservative: includes the shared fixed cost (context + weights) divided by N.
        # plot_parallel_benchmark.py fits fixed + marginal cost across N for a better estimate.
        row["vram_per_worker_mib"] = f"{vram_per_pos:.0f}"
        row["projected_max_n_vram"] = int((gpu_total - baseline) * args.vram_headroom // vram_per_pos)

    ok = [t for t in out["threads"] if t.get("status") == "ok"]
    if status == "ok":
        wall = max(t["t_end"] for t in ok) - min(t["t_start"] for t in ok)
        elapsed = [t["elapsed_s"] for t in ok]
        total_frames = frames_per_seq * n
        row.update(
            total_frames=total_frames,
            wall_s=f"{wall:.3f}",
            aggregate_fps=f"{total_frames / wall:.3f}",
            per_position_fps_mean=f"{sum(frames_per_seq / e for e in elapsed) / len(elapsed):.3f}",
            worker_s_mean=f"{sum(elapsed) / len(elapsed):.3f}",
            worker_s_max=f"{max(elapsed):.3f}",
            torch_peak_reserved_mib_max=f"{out['torch_peak_reserved_mib']:.0f}",
            torch_peak_alloc_mib_max=f"{out['torch_peak_alloc_mib']:.0f}",
        )

    if not args.keep_outputs:
        shutil.rmtree(run_out_dir, ignore_errors=True)
    return row


def append_row(csv_path: Path, row: dict) -> None:
    import csv

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
    p.add_argument("--sequence", default="data/2D/test/CTC/10")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Only track the first K frames of the sequence (shorter sweeps)")
    p.add_argument("--warmup_frames", type=int, default=3)
    p.add_argument("--n_list", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--refine", action="store_true",
                   help="After the first failure, bisect between last success and first failure")
    p.add_argument("--no_streams", action="store_true",
                   help="All threads use the default CUDA stream (no GPU overlap between positions)")
    p.add_argument("--out_csv", default="results/parallel_bench/shared_model.csv")
    p.add_argument("--out_dir", default="results/parallel_bench/outputs")
    p.add_argument("--keep_outputs", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--vram_headroom", type=float, default=0.95)
    p.add_argument("--threads_per_worker", type=int, default=None,
                   help="OMP/MKL threads for the process (default: library default)")
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
    p.add_argument("--trim_memory", action="store_true",
                   help="Keep only the last num_maskmem entries of each cell's memory history")
    p.add_argument("--frame_metrics", action="store_true",
                   help="Write per-frame time + detected/GT cell counts per thread to "
                        "<out_csv dir>/frame_metrics/ (uses the tracker's FrameMetricsRecorder)")
    p.add_argument("--gt_dir", default=None,
                   help="GT dir for per-frame GT cell counts (default: <sequence>_GT/TRA)")
    p.add_argument("--no_offload_video", action="store_true")
    p.add_argument("--no_offload_state", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)  # relative paths resolve against the repo, like track_cells.py
    if args.threads_per_worker is not None:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(args.threads_per_worker)

    gpu_total = gpu_memory_total_mib(args.gpu)
    csv_path = Path(args.out_csv)
    print(f"[shared] model={args.model_name} ckpt={args.checkpoint_num} seq={args.sequence} "
          f"max_frames={args.max_frames} streams={not args.no_streams} GPU total={gpu_total} MiB")

    last_ok, first_fail = 0, None
    for n in sorted(set(args.n_list)):
        print(f"[shared] N={n} ...", flush=True)
        row = run_once(n, args, gpu_total)
        append_row(csv_path, row)
        print(f"[shared]   status={row['status']} wall={row['wall_s']}s agg_fps={row['aggregate_fps']} "
              f"per_pos_fps={row['per_position_fps_mean']} peak_vram={row['peak_gpu_mib']}MiB "
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
            print(f"[shared] refine N={mid} (ok<={lo}, fail>={hi}) ...", flush=True)
            row = run_once(mid, args, gpu_total)
            append_row(csv_path, row)
            print(f"[shared]   status={row['status']} agg_fps={row['aggregate_fps']} "
                  f"peak_vram={row['peak_gpu_mib']}MiB", flush=True)
            if row["status"] == "ok":
                lo = mid
            else:
                hi = mid
        last_ok, first_fail = lo, hi

    print(f"[shared] {args.model_name}: max parallel positions (shared model) = {last_ok}"
          + (f" (first failure at N={first_fail})" if first_fail else " (no failure within n_list)"))
    print(f"[shared] results appended to {csv_path}")


if __name__ == "__main__":
    main()
