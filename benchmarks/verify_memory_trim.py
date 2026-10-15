#!/usr/bin/env python3
"""Check that trimming the stored memory history does not change tracking outputs.

Runs the tracker on each sequence three times in the same process with the same model:
as-is (full per-cell memory history kept), trimmed (model.trim_memory_history = True,
only the last num_maskmem entries kept), then as-is again. Reports pixel differences
for as-is vs as-is (run-to-run baseline) and as-is vs trimmed, and CTC metrics (DET, SEG,
TRA, CHOTA) of all three runs against the GT. The tracker is not bit-reproducible run to
run, so PASS means each trimmed metric is within the as-is run-to-run difference
(or --metric_tol, whichever is larger) of the as-is mean. Deterministic CUDA/cuDNN algorithms are enabled by default.
Also reports peak VRAM and runtime of each run.

Example:
    python benchmarks/verify_memory_trim.py --model_name CellSam2-2D-bkg80 \
        --sequences data/2D/test/CTC/05 data/2D/test/CTC/10
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import REPO_ROOT, load_model  # noqa: E402

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
import torch  # noqa: E402

METRICS = ["DET", "SEG", "TRA", "CHOTA"]


def compare_dirs(a: Path, b: Path) -> list[str]:
    diffs = []
    files_a = sorted(p.name for p in a.iterdir() if p.is_file())
    files_b = sorted(p.name for p in b.iterdir() if p.is_file())
    if files_a != files_b:
        diffs.append(f"file lists differ: {set(files_a) ^ set(files_b)}")
    for name in sorted(set(files_a) & set(files_b)):
        pa, pb = a / name, b / name
        if pa.suffix.lower() in {".tif", ".tiff"}:
            xa, xb = tifffile.imread(pa), tifffile.imread(pb)
            if xa.shape != xb.shape or not np.array_equal(xa, xb):
                n = int((xa != xb).sum()) if xa.shape == xb.shape else -1
                diffs.append(f"{name}: {n} pixels differ")
        elif pa.suffix == ".txt":
            if pa.read_text() != pb.read_text():
                diffs.append(f"{name}: text differs")
    return diffs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint_num", type=int, default=20)
    p.add_argument("--sequences", nargs="+", default=["data/2D/test/CTC/05", "data/2D/test/CTC/10"])
    p.add_argument("--out_dir", default="results/parallel_bench/verify_trim")
    p.add_argument("--keep_outputs", action="store_true")
    p.add_argument("--gt_dir", default=None, help="GT dir (default: <sequence>_GT next to the sequence)")
    p.add_argument("--metric_tol", type=float, default=0.005,
                   help="Minimum allowed |trimmed - as-is| per metric (run-to-run noise from 2 runs is crude)")
    p.add_argument("--no_deterministic", action="store_true",
                   help="Do not force deterministic CUDA/cuDNN algorithms")
    args = p.parse_args()

    if not args.no_deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    from cell_tracker import SAM2AutomaticCellTracker

    model, cfg = load_model(args.model_name, args.checkpoint_num)
    if not hasattr(model, "trim_memory_history"):
        raise RuntimeError("This sam2_base.py has no trim_memory_history option")
    tracker = SAM2AutomaticCellTracker(
        model,
        pred_iou_thresh=0.7,
        obj_score_thresh=0.5,
        div_obj_score_thresh=0.4,
        box_nms_thresh=0.5,
        segment=False,
        use_heatmap=True,
        resize_threshold=getattr(cfg.scratch, "resize_threshold", None),
        crop_overlap=128,
    )
    print(f"[verify] {args.model_name}: heatmap_threshold={tracker.heatmap_threshold}")

    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    all_ok = True
    rows: list[dict] = []
    for seq in args.sequences:
        seq = Path(seq) if Path(seq).is_absolute() else REPO_ROOT / seq
        stats = {}
        for name, trim in (("as_is", False), ("trimmed", True), ("as_is_repeat", False)):
            model.trim_memory_history = trim
            out = out_root / args.model_name / seq.name / name
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tracker.predict(seq, out, offload_video_to_cpu=True, offload_state_to_cpu=True)
            torch.cuda.synchronize()
            stats[name] = (time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**30, out)
            print(f"[verify] seq {seq.name} {name}: {stats[name][0]:.1f}s, "
                  f"{stats[name][1]:.2f} GiB peak alloc", flush=True)
        base_diffs = compare_dirs(stats["as_is"][2], stats["as_is_repeat"][2])
        diffs = compare_dirs(stats["as_is"][2], stats["trimmed"][2])
        print(f"[verify] seq {seq.name}: pixel diffs as-is vs as-is repeat: {_summ(base_diffs)}", flush=True)
        print(f"[verify] seq {seq.name}: pixel diffs as-is vs trimmed:      {_summ(diffs)}", flush=True)

        gt_dir = seq.parent / f"{seq.name}_GT"
        if args.gt_dir is not None:
            gt_dir = Path(args.gt_dir)
        if not gt_dir.is_dir():
            print(f"[verify] no GT at {gt_dir}; pass requires pixel-identical outputs")
            seq_ok = not diffs
        else:
            from ctc_metrics import evaluate_sequence

            m = {}
            for name in ("as_is", "trimmed", "as_is_repeat"):
                r = evaluate_sequence(str(stats[name][2]), str(gt_dir), metrics=METRICS)
                m[name] = {k: float(r[k]) for k in METRICS}
            seq_ok = True
            print(f"[verify] seq {seq.name} metrics vs GT:")
            print("[verify]   " + f"{'run':<14}" + "".join(f"{k:>9}" for k in METRICS))
            for name in ("as_is", "as_is_repeat", "trimmed"):
                print("[verify]   " + f"{name:<14}" + "".join(f"{m[name][k]:9.4f}" for k in METRICS))
            for k in METRICS:
                noise = abs(m["as_is"][k] - m["as_is_repeat"][k])
                ref = (m["as_is"][k] + m["as_is_repeat"][k]) / 2
                delta = abs(m["trimmed"][k] - ref)
                ok = delta <= max(noise, args.metric_tol)
                seq_ok &= ok
                print(f"[verify]   {k}: |trimmed - as-is mean| = {delta:.4f}, as-is run-to-run = {noise:.4f}"
                      f" -> {'ok' if ok else 'OUTSIDE NOISE'}")
            rows.extend(
                dict(model=args.model_name, sequence=seq.name, run=name,
                     seconds=f"{stats[name][0]:.1f}", peak_alloc_gib=f"{stats[name][1]:.2f}",
                     **{k: f"{m[name][k]:.6f}" for k in METRICS})
                for name in m
            )
        all_ok &= seq_ok
        if not args.keep_outputs:
            shutil.rmtree(out_root / args.model_name / seq.name, ignore_errors=True)

    if rows:
        import csv

        csv_path = out_root / f"verify_{args.model_name}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[verify] metrics -> {csv_path}")
    print(f"[verify] {args.model_name}: "
          + ("PASS (trimmed within run-to-run variation)" if all_ok else "FAIL (trimmed outside run-to-run variation)"))
    sys.exit(0 if all_ok else 1)


def _summ(diffs: list[str]) -> str:
    if not diffs:
        return "identical"
    px = sum(int(d.split(": ")[1].split()[0]) for d in diffs if "pixels differ" in d)
    return f"{len(diffs)} files differ, {px} pixels total"


if __name__ == "__main__":
    main()
