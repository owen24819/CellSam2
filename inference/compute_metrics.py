#!/usr/bin/env python3
"""
Compute Cell-HOTA (+ DetA, AssA, DivA) across alpha values,
and CTC TRA/SEG/OP_CTB metrics, matching the README figure tables.

Usage:
    python compute_metrics.py \
        --gt_path  /path/to/dataset/test/CTC \
        --res_path results/tracking \
        --output_dir results/metrics
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CellHOTA, CTCMetrics


def find_sequence_pairs(gt_path: Path, res_path: Path):
    """
    Find matching GT (XX_GT) and RES (XX_RES) sequence folder pairs.
    Returns list of (gt_tra_path, res_path) tuples.
    """
    pairs = []
    for item in sorted(gt_path.iterdir()):
        if item.is_dir() and re.match(r"^\d{2}_GT$", item.name):
            seq_id = item.name.replace("_GT", "")
            res_seq = res_path / f"{seq_id}_RES"
            if res_seq.exists():
                pairs.append((item, res_seq))
            else:
                print(f"  WARNING: No RES folder found for {item.name} at {res_seq}")
    return pairs


def compute_sequence_metrics(gt_folder: Path, res_folder: Path, alphas: list[float]):
    """Compute all metrics for a single sequence."""
    gt_data  = load_ctc_data(str(gt_folder / "TRA"), str(gt_folder / "TRA" / "man_track.txt"))
    res_data = load_ctc_data(str(res_folder), str(res_folder / "res_track.txt"))

    seq_results = {"CTC": {}, "CellHOTA_by_alpha": {}}

    # ── CTC TRA / SEG ──────────────────────────────────────────────────────────
    ctc_results = run_metrics(
        gt_data,
        res_data,
        matcher=CTCMatcher(),
        metrics=[CTCMetrics()],
    )
    ctc = ctc_results[0].results
    tra = ctc.get("TRA", float("nan"))
    seg = ctc.get("SEG", float("nan"))
    op_ctb = math.sqrt(tra * seg) if not (math.isnan(tra) or math.isnan(seg)) else float("nan")
    seq_results["CTC"] = {"TRA": tra, "SEG": seg, "OP_CTB": op_ctb}

    # ── Cell-HOTA across alpha values ──────────────────────────────────────────
    for alpha in alphas:
        hota_results = run_metrics(
            gt_data,
            res_data,
            matcher=CTCMatcher(),
            metrics=[CellHOTA(threshold=alpha)],
        )
        h = hota_results[0].results
        seq_results["CellHOTA_by_alpha"][round(alpha, 3)] = {
            "Cell-HOTA": h.get("HOTA",  float("nan")),
            "DetA":      h.get("DetA",  float("nan")),
            "AssA":      h.get("AssA",  float("nan")),
            "DivA":      h.get("DivA",  float("nan")),
        }

    return seq_results


def aggregate_across_sequences(all_seq_results: list[dict], alphas: list[float]):
    """Average metrics across sequences (CTC convention: mean over sequences)."""
    # CTC metrics
    tra_vals  = [r["CTC"]["TRA"]   for r in all_seq_results]
    seg_vals  = [r["CTC"]["SEG"]   for r in all_seq_results]
    op_vals   = [r["CTC"]["OP_CTB"] for r in all_seq_results]
    agg_ctc = {
        "TRA":    np.nanmean(tra_vals),
        "SEG":    np.nanmean(seg_vals),
        "OP_CTB": np.nanmean(op_vals),
    }

    # Cell-HOTA metrics per alpha
    agg_hota = {}
    for alpha in alphas:
        a = round(alpha, 3)
        keys = ["Cell-HOTA", "DetA", "AssA", "DivA"]
        agg_hota[a] = {}
        for k in keys:
            vals = [r["CellHOTA_by_alpha"][a][k] for r in all_seq_results]
            agg_hota[a][k] = np.nanmean(vals)

    return agg_ctc, agg_hota


def print_summary(agg_ctc: dict, agg_hota: dict, alpha_05: float = 0.5):
    """Print tables matching the README figure format."""
    a = round(alpha_05, 3)
    h = agg_hota.get(a, {})

    print("\n" + "=" * 60)
    print("TABLE B / C (left) — Cell-HOTA metrics at alpha=0.5")
    print("=" * 60)
    print(f"  {'Cell-HOTA_0.5':>16}  {'DetA_0.5':>10}  {'AssA_0.5':>10}  {'DivA_0.5':>10}")
    print(f"  {'CellSam2':>16}  "
          f"{h.get('Cell-HOTA', float('nan')):>10.2f}  "
          f"{h.get('DetA',      float('nan')):>10.2f}  "
          f"{h.get('AssA',      float('nan')):>10.2f}  "
          f"{h.get('DivA',      float('nan')):>10.2f}")

    print("\n" + "=" * 60)
    print("TABLE C (right) — CTC metrics")
    print("=" * 60)
    print(f"  {'OP_CTB':>8}  {'TRA':>8}  {'SEG':>8}")
    print(f"  {agg_ctc['OP_CTB']:>8.3f}  {agg_ctc['TRA']:>8.3f}  {agg_ctc['SEG']:>8.3f}")

    print("\n" + "=" * 60)
    print("FIGURE A — Cell-HOTA curve across alpha values")
    print("=" * 60)
    print(f"  {'alpha':>8}  {'Cell-HOTA':>12}  {'DetA':>8}  {'AssA':>8}  {'DivA':>8}")
    for alpha_val, metrics in sorted(agg_hota.items()):
        print(f"  {alpha_val:>8.2f}  "
              f"{metrics['Cell-HOTA']:>12.2f}  "
              f"{metrics['DetA']:>8.2f}  "
              f"{metrics['AssA']:>8.2f}  "
              f"{metrics['DivA']:>8.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path",    required=True, help="Path to CTC GT folder (contains XX_GT/ dirs)")
    parser.add_argument("--res_path",   required=True, help="Path to inference results (contains XX_RES/ dirs)")
    parser.add_argument("--output_dir", default="results/metrics", help="Where to save JSON/CSV outputs")
    args = parser.parse_args()

    gt_path  = Path(args.gt_path)
    res_path = Path(args.res_path)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Alpha range matching Figure A (0.05 to 0.95 in steps of 0.05)
    alphas = [round(a, 2) for a in np.arange(0.05, 1.00, 0.05)]

    pairs = find_sequence_pairs(gt_path, res_path)
    if not pairs:
        raise RuntimeError(f"No matching GT/RES sequence pairs found in:\n  GT:  {gt_path}\n  RES: {res_path}")

    print(f"Found {len(pairs)} sequence(s): {[p[0].name for p in pairs]}")

    all_seq_results = []
    for gt_folder, res_folder in pairs:
        print(f"\nProcessing {gt_folder.name} ...")
        seq_res = compute_sequence_metrics(gt_folder, res_folder, alphas)
        all_seq_results.append(seq_res)

    agg_ctc, agg_hota = aggregate_across_sequences(all_seq_results, alphas)

    # ── Save outputs ───────────────────────────────────────────────────────────
    summary = {"CTC": agg_ctc, "CellHOTA_by_alpha": agg_hota}
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # CSV of HOTA curve (for easy plotting)
    rows = [{"alpha": a, **v} for a, v in sorted(agg_hota.items())]
    pd.DataFrame(rows).to_csv(out_dir / "cell_hota_curve.csv", index=False)

    # CSV of CTC metrics
    pd.DataFrame([agg_ctc]).to_csv(out_dir / "ctc_metrics.csv", index=False)

    print_summary(agg_ctc, agg_hota)
    print(f"\nResults saved to: {out_dir}/")


if __name__ == "__main__":
    main()