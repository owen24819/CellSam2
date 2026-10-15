#!/usr/bin/env python3
"""Max batch size / throughput of the CellSAM2 image encoder (base vs small).

Complements parallel_positions_benchmark.py: instead of running N full tracking
pipelines, this stacks the SAME preprocessed frame crops along the batch dimension
and runs model.forward_image (Hiera backbone + FPN neck + high-res convs), which is
the part of the model that differs between base_plus and small. Batch size is
doubled until CUDA OOM, then bisected to find the largest batch that fits.

Each crop corresponds to one 512x512 model input, so batch size B ~= B crops; a
2048x2048 position is tiled into 25 crops by the tracker (resize_threshold=600,
crop_overlap=128), i.e. positions_per_step = B / crops_per_frame.

Example:
    python benchmarks/batched_encoder_benchmark.py \
        --model_name CellSam2-2D-small-a0p15 --checkpoint_num 20 \
        --sequence data/2D/test/CTC/10 \
        --out_csv results/parallel_bench/batched_encoder.csv
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import REPO_ROOT, count_params, load_model  # noqa: E402

import torch  # noqa: E402

CSV_FIELDS = [
    "model_name",
    "checkpoint_num",
    "params_m",
    "dtype",
    "batch_size",
    "status",
    "ms_per_batch",
    "crops_per_s",
    "crops_per_frame",
    "position_frames_per_s",
    "peak_alloc_mib",
    "peak_reserved_mib",
    "gpu_total_mib",
]


def crops_for_frame(height: int, width: int, target: int, overlap: int, resize_threshold: int | None) -> int:
    """Same tiling rule as SAM2AutomaticCellTracker._compute_crop_boxes."""
    if resize_threshold is not None and height <= resize_threshold and width <= resize_threshold:
        return 1
    stride = max(1, target - overlap)
    cols = 1 if width <= target else math.ceil((width - target) / stride) + 1
    rows = 1 if height <= target else math.ceil((height - target) / stride) + 1
    return rows * cols


def load_crops(seq: Path, image_size: int, n_frames: int) -> torch.Tensor:
    """Load the first n_frames frames, take a top-left image_size crop of each and preprocess."""
    from sam2.utils.misc import read_image
    from sam2.utils.transforms import SAM2Transforms

    transforms = SAM2Transforms(resolution=image_size, mask_threshold=0.0)
    frames = sorted(p for p in seq.iterdir() if p.suffix.lower() in {".tif", ".tiff"})[:n_frames]
    crops = []
    for f in frames:
        img = read_image(str(f))
        w, h = img.size
        crop = img.crop((0, 0, min(w, image_size), min(h, image_size)))
        transforms._set_hw_params(crop, image_size)  # as in SAM2AutomaticCellTracker.init_states
        crops.append(transforms(crop))
    return torch.stack(crops)  # (F, 3, S, S)


def time_batch(model, batch: torch.Tensor, iters: int, warmup: int, amp_dtype) -> float:
    ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
    with torch.inference_mode(), ctx:
        for _ in range(warmup):
            model.forward_image(batch)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model.forward_image(batch)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def try_batch(model, crops: torch.Tensor, bs: int, args, amp_dtype) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        reps = math.ceil(bs / crops.shape[0])
        batch = crops.repeat(reps, 1, 1, 1)[:bs].to("cuda", non_blocking=True)
        sec = time_batch(model, batch, args.iters, args.warmup, amp_dtype)
        out = dict(
            status="ok",
            ms_per_batch=f"{sec * 1e3:.2f}",
            crops_per_s=f"{bs / sec:.1f}",
            peak_alloc_mib=f"{torch.cuda.max_memory_allocated() / 2**20:.0f}",
            peak_reserved_mib=f"{torch.cuda.max_memory_reserved() / 2**20:.0f}",
        )
        out["_crops_per_s"] = bs / sec
        del batch
        return out
    except torch.cuda.OutOfMemoryError:
        return dict(status="oom")
    except RuntimeError as e:
        # Very large batches can exceed CUDA's kernel launch grid limits in the windowed
        # attention (scaled_dot_product_attention) before running out of memory.
        if "invalid configuration argument" in str(e):
            return dict(status="kernel_limit")
        raise
    finally:
        torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint_num", type=int, default=20)
    p.add_argument("--sequence", default="data/2D/test/CTC/10")
    p.add_argument("--n_frames", type=int, default=16, help="Distinct frames to cycle through in a batch")
    p.add_argument("--start_bs", type=int, default=1)
    p.add_argument("--max_bs", type=int, default=8192)
    p.add_argument("--no_refine", action="store_true", help="Skip bisection after first OOM")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
                   help="fp32 matches the current tracker (no autocast); bf16/fp16 show the headroom")
    p.add_argument("--crop_overlap", type=int, default=128)
    p.add_argument("--out_csv", default="results/parallel_bench/batched_encoder.csv")
    args = p.parse_args()

    seq = Path(args.sequence)
    if not seq.is_absolute():
        seq = REPO_ROOT / seq
    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = REPO_ROOT / out_csv

    torch.backends.cudnn.benchmark = True
    amp_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    model, cfg = load_model(args.model_name, args.checkpoint_num)
    image_size = model.image_size
    crops = load_crops(seq, image_size, args.n_frames)

    from sam2.utils.misc import read_image
    first = read_image(str(sorted(p for p in seq.iterdir() if p.suffix.lower() in {".tif", ".tiff"})[0]))
    w, h = first.size
    cpf = crops_for_frame(h, w, image_size, args.crop_overlap, getattr(cfg.scratch, "resize_threshold", None))
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 2**20
    params_m = count_params(model) / 1e6
    print(f"[enc] {args.model_name}: {params_m:.1f}M params, image_size={image_size}, "
          f"frame {w}x{h} -> {cpf} crops/frame, dtype={args.dtype}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not out_csv.exists()
    f = open(out_csv, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if new:
        writer.writeheader()

    def record(bs: int, res: dict) -> None:
        row = {k: "" for k in CSV_FIELDS}
        row.update(
            model_name=args.model_name,
            checkpoint_num=args.checkpoint_num,
            params_m=f"{params_m:.1f}",
            dtype=args.dtype,
            batch_size=bs,
            crops_per_frame=cpf,
            gpu_total_mib=f"{gpu_total:.0f}",
        )
        row.update({k: v for k, v in res.items() if not k.startswith("_")})
        if res["status"] == "ok":
            row["position_frames_per_s"] = f"{res['_crops_per_s'] / cpf:.2f}"
        writer.writerow(row)
        f.flush()
        print(f"[enc]   bs={bs:5d} {res['status']:>3} "
              + (f"{res['ms_per_batch']} ms/batch, {res['crops_per_s']} crops/s, "
                 f"peak reserved {res['peak_reserved_mib']} MiB" if res["status"] == "ok" else ""),
              flush=True)

    last_ok, first_fail = 0, None
    bs = args.start_bs
    while bs <= args.max_bs:
        res = try_batch(model, crops, bs, args, amp_dtype)
        record(bs, res)
        if res["status"] != "ok":
            first_fail = bs
            break
        last_ok = bs
        bs *= 2

    if first_fail is not None and not args.no_refine:
        lo, hi = last_ok, first_fail
        while hi - lo > max(1, lo // 32):  # ~3% resolution is plenty
            mid = (lo + hi) // 2
            res = try_batch(model, crops, mid, args, amp_dtype)
            record(mid, res)
            if res["status"] == "ok":
                lo = mid
            else:
                hi = mid
        last_ok = lo

    f.close()
    print(f"[enc] {args.model_name}: max batch (crops) = {last_ok} "
          f"~= {last_ok / cpf:.1f} positions' frames per encoder step; results -> {out_csv}")


if __name__ == "__main__":
    main()
