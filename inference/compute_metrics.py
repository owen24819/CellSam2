#!/usr/bin/env python3
"""
Compute Cell-HOTA (CHOTAMetric) across alpha values and CTC TRA/SEG/OP_CTB,
matching the README figure tables.

Usage:
    python compute_metrics.py \
        --gt_path  /path/to/dataset/test/CTC \
        --res_path results/tracking \
        --output_dir results/metrics
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatcher, IOUMatcher
from traccuracy.metrics import CHOTAMetric, CTCMetrics


def find_sequence_pairs(gt_path: Path, res_path: Path):
    """Find matching XX_GT / XX_RES folder pairs."""
    pairs = []
    for item in sorted(gt_path.iterdir()):
        if item.is_dir() and re.match(r"^\d{2}_GT$", item.name):
            seq_id = item.name.replace("_GT", "")
            res_seq = res_path / f"{seq_id}_RES"
            if res_seq.exists():
                pairs.append((item, res_seq))
            else:
                print(f"  WARNING: No RES folder for {item.name} at {res_seq}")
    return pairs


def load_sequence(gt_folder: Path, res_folder: Path):
    """Load a GT/RES sequence pair."""
    gt_data  = load_ctc_data(
        str(gt_folder / "TRA"),
        str(gt_folder / "TRA" / "man_track.txt"),
    )
    res_data = load_ctc_data(
        str(res_folder),
        str(res_folder / "res_track.txt"),
    )
    return gt_data, res_data


def compute_sequence_metrics(gt_folder: Path, res_folder: Path, alphas: list):
    """Compute all metrics for a single sequence."""
    gt_data, res_data = load_sequence(gt_folder, res_folder)

    seq_results = {"CTC": {}, "CHOTA_by_alpha": {}}

    # ── CTC TRA / SEG (uses CTCMatcher with fixed IoGT > 0.5) ────────────────
    ctc_out = run_metrics(
        gt_data, res_data,
        matcher=CTCMatcher(),
        metrics=[CTCMetrics()],
    )
    ctc = ctc_out[0].results
    tra   = ctc.get("TRA", float("nan"))
    seg   = ctc.get("SEG", float("nan"))
    op_ctb = math.sqrt(tra * seg) if not (math.isnan(tra) or math.isnan(seg)) else float("nan")
    seq_results["CTC"] = {"TRA": tra, "SEG": seg, "OP_CTB": op_ctb}

    # ── CHOTAMetric across alpha values (alpha = IoU threshold for matching) ──
    for alpha in alphas:
        a = round(alpha, 2)
        hota_out = run_metrics(
            gt_data, res_data,
            matcher=IOUMatcher(iou_threshold=a),
            metrics=[CHOTAMetric()],
        )
        h = hota_out[0].results
        seq_results["CHOTA_by_alpha"][a] = {
            "Cell-HOTA": h.get("CHOTA", float("nan")) * 100,  # scale to 0-100 like the figure
        }

    return seq_results


def aggregate(all_seq: list, alphas: list):
    """Average metrics across sequences."""
    agg_ctc = {
        k: np.nanmean([s["CTC"][k] for s in all_seq])
        for k in ("TRA", "SEG", "OP_CTB")
    }
    agg_hota = {}
    for alpha in alphas:
        a = round(alpha, 2)
        agg_hota[a] = {
            "Cell-HOTA": np.nanmean([s["CHOTA_by_alpha"][a]["Cell-HOTA"] for s in all_seq]),
        }
    return agg_ctc, agg_hota


def print_summary(agg_ctc, agg_hota, alpha_05=0.5):
    a = round(alpha_05, 2)
    h = agg_hota.get(a, {})

    print("\n" + "=" * 55)
    print("TABLE — Cell-HOTA at alpha=0.5")
    print("=" * 55)
    print(f"  {'Cell-HOTA_0.5':>16}")
    print(f"  {'CellSam2':>16}   {h.get('Cell-HOTA', float('nan')):>6.2f}")

    print("\n" + "=" * 55)
    print("TABLE — CTC metrics")
    print("=" * 55)
    print(f"  {'OP_CTB':>8}  {'TRA':>8}  {'SEG':>8}")
    print(f"  {agg_ctc['OP_CTB']:>8.3f}  {agg_ctc['TRA']:>8.3f}  {agg_ctc['SEG']:>8.3f}")

    print("\n" + "=" * 55)
    print("FIGURE — Cell-HOTA curve across alpha values")
    print("=" * 55)
    print(f"  {'alpha':>8}  {'Cell-HOTA':>12}")
    for av, mv in sorted(agg_hota.items()):
        print(f"  {av:>8.2f}  {mv['Cell-HOTA']:>12.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path",    required=True)
    parser.add_argument("--res_path",   required=True)
    parser.add_argument("--output_dir", default="results/metrics")
    args = parser.parse_args()

    gt_path  = Path(args.gt_path)
    res_path = Path(args.res_path)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Alpha range matching Figure A: 0.05 → 0.95 in steps of 0.05
    alphas = [round(a, 2) for a in np.arange(0.05, 1.00, 0.05)]

    pairs = find_sequence_pairs(gt_path, res_path)
    if not pairs:
        raise RuntimeError(
            f"No GT/RES pairs found.\n  GT:  {gt_path}\n  RES: {res_path}"
        )
    print(f"Found {len(pairs)} sequence(s): {[p[0].name for p in pairs]}")

    all_seq = []
    for gt_folder, res_folder in pairs:
        print(f"\nProcessing {gt_folder.name} ...")
        all_seq.append(compute_sequence_metrics(gt_folder, res_folder, alphas))

    agg_ctc, agg_hota = aggregate(all_seq, alphas)

    # Save outputs
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump({"CTC": agg_ctc, "CHOTA_by_alpha": agg_hota}, f, indent=2)

    pd.DataFrame(
        [{"alpha": a, **v} for a, v in sorted(agg_hota.items())]
    ).to_csv(out_dir / "cell_hota_curve.csv", index=False)

    pd.DataFrame([agg_ctc]).to_csv(out_dir / "ctc_metrics.csv", index=False)

    print_summary(agg_ctc, agg_hota)
    print(f"\nResults saved to: {out_dir}/")


if __name__ == "__main__":
    main()