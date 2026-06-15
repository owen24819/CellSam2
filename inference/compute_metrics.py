#!/usr/bin/env python3
"""
Compute Cell-HOTA / CHOTA metrics across alpha values and CTC TRA/SEG/OP_CTB.

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


def safe_percent(results: dict, key: str) -> float:
    """
    Pull a metric from a traccuracy results dict and convert to percent.
    Returns NaN if missing.
    """
    value = results.get(key, float("nan"))
    if value is None or math.isnan(value):
        return float("nan")
    return value * 100


def find_sequence_pairs(gt_path: Path, res_path: Path):
    """
    Match XX_GT folders in gt_path to prediction folders in res_path.

    Supports both:
        results/.../01/
    and:
        results/.../01_RES/
    """
    pairs = []

    for item in sorted(gt_path.iterdir()):
        if item.is_dir() and re.match(r"^\d{2}_GT$", item.name):
            seq_id = item.name.replace("_GT", "")

            possible_res_dirs = [
                res_path / seq_id,
                res_path / f"{seq_id}_RES",
            ]

            res_seq = None
            for candidate in possible_res_dirs:
                if candidate.exists():
                    res_seq = candidate
                    break

            if res_seq is not None:
                pairs.append((item, res_seq))
            else:
                print(f"WARNING: No result folder found for {item.name}")
                print(f"  Tried:")
                for candidate in possible_res_dirs:
                    print(f"    {candidate}")

    return pairs


def load_sequence(gt_folder: Path, res_folder: Path):
    gt_data = load_ctc_data(
        str(gt_folder / "TRA"),
        str(gt_folder / "TRA" / "man_track.txt"),
    )

    res_data = load_ctc_data(
        str(res_folder),
        str(res_folder / "res_track.txt"),
    )

    return gt_data, res_data


def compute_sequence_metrics(gt_folder: Path, res_folder: Path, alphas: list[float]):
    gt_data, res_data = load_sequence(gt_folder, res_folder)

    seq_results = {
        "sequence": gt_folder.name.replace("_GT", ""),
        "CTC": {},
        "CHOTA_by_alpha": {},
    }

    # CTC TRA / SEG
    ctc_results, _ = run_metrics(
        gt_data,
        res_data,
        matcher=CTCMatcher(),
        metrics=[CTCMetrics()],
    )

    ctc = ctc_results[0]["results"]

    tra = ctc.get("TRA", float("nan"))
    seg = ctc.get("SEG", float("nan"))

    # The table in your screenshot appears to use arithmetic mean:
    # OP_CTB = (TRA + SEG) / 2
    if math.isnan(tra) or math.isnan(seg):
        op_ctb = float("nan")
    else:
        op_ctb = (tra + seg) / 2

    seq_results["CTC"] = {
        "TRA": tra,
        "SEG": seg,
        "OP_CTB": op_ctb,
    }

    # CHOTAMetric across alpha values
    for alpha in alphas:
        a = round(alpha, 2)

        hota_results, _ = run_metrics(
            gt_data,
            res_data,
            matcher=IOUMatcher(iou_threshold=a),
            metrics=[CHOTAMetric()],
        )

        h = hota_results[0]["results"]

        # Useful for debugging if traccuracy changes key names
        print(f"Alpha {a:.2f} CHOTA result keys: {sorted(h.keys())}")

        seq_results["CHOTA_by_alpha"][a] = {
            "Cell-HOTA": safe_percent(h, "CHOTA"),
            "DetA": safe_percent(h, "DetA"),
            "AssA": safe_percent(h, "AssA"),
            "DivA": safe_percent(h, "DivA"),
        }

    return seq_results


def aggregate(all_seq: list[dict], alphas: list[float]):
    """
    Average metrics across sequences.
    """
    agg_ctc = {
        k: np.nanmean([s["CTC"][k] for s in all_seq])
        for k in ("TRA", "SEG", "OP_CTB")
    }

    agg_hota = {}

    for alpha in alphas:
        a = round(alpha, 2)
        agg_hota[a] = {
            metric_name: np.nanmean(
                [s["CHOTA_by_alpha"][a][metric_name] for s in all_seq]
            )
            for metric_name in ("Cell-HOTA", "DetA", "AssA", "DivA")
        }

    return agg_ctc, agg_hota


def make_alpha_curve_df(agg_hota: dict):
    rows = []

    for alpha, values in sorted(agg_hota.items()):
        rows.append(
            {
                "alpha": alpha,
                "Cell-HOTA": values["Cell-HOTA"],
                "DetA": values["DetA"],
                "AssA": values["AssA"],
                "DivA": values["DivA"],
            }
        )

    return pd.DataFrame(rows)


def make_summary_df(agg_ctc: dict, agg_hota: dict, alpha_05: float = 0.5):
    alpha_05 = round(alpha_05, 2)
    alpha_05_values = agg_hota[alpha_05]

    hota_df = make_alpha_curve_df(agg_hota)

    summary = {
        "Cell-HOTA_0.5": alpha_05_values["Cell-HOTA"],
        "DetA_0.5": alpha_05_values["DetA"],
        "AssA_0.5": alpha_05_values["AssA"],
        "DivA_0.5": alpha_05_values["DivA"],

        "Cell-HOTA_mean": hota_df["Cell-HOTA"].mean(),
        "DetA_mean": hota_df["DetA"].mean(),
        "AssA_mean": hota_df["AssA"].mean(),
        "DivA_mean": hota_df["DivA"].mean(),

        "OP_CTB": agg_ctc["OP_CTB"],
        "TRA": agg_ctc["TRA"],
        "SEG": agg_ctc["SEG"],
    }

    return pd.DataFrame([summary])


def print_summary(agg_ctc, agg_hota, alpha_05=0.5):
    a = round(alpha_05, 2)
    h = agg_hota.get(a, {})

    hota_df = make_alpha_curve_df(agg_hota)

    print("\n" + "=" * 70)
    print("TABLE B — Cell-HOTA components at alpha=0.5")
    print("=" * 70)
    print(
        f"  {'Cell-HOTA_0.5':>16}"
        f"  {'DetA_0.5':>10}"
        f"  {'AssA_0.5':>10}"
        f"  {'DivA_0.5':>10}"
    )
    print(
        f"  {h.get('Cell-HOTA', float('nan')):>16.2f}"
        f"  {h.get('DetA', float('nan')):>10.2f}"
        f"  {h.get('AssA', float('nan')):>10.2f}"
        f"  {h.get('DivA', float('nan')):>10.2f}"
    )

    print("\n" + "=" * 70)
    print("TABLE C — Mean Cell-HOTA components across alpha values")
    print("=" * 70)
    print(
        f"  {'Cell-HOTA':>12}"
        f"  {'DetA':>10}"
        f"  {'AssA':>10}"
        f"  {'DivA':>10}"
    )
    print(
        f"  {hota_df['Cell-HOTA'].mean():>12.2f}"
        f"  {hota_df['DetA'].mean():>10.2f}"
        f"  {hota_df['AssA'].mean():>10.2f}"
        f"  {hota_df['DivA'].mean():>10.2f}"
    )

    print("\n" + "=" * 70)
    print("TABLE C — CTC metrics")
    print("=" * 70)
    print(f"  {'OP_CTB':>8}  {'TRA':>8}  {'SEG':>8}")
    print(f"  {agg_ctc['OP_CTB']:>8.3f}  {agg_ctc['TRA']:>8.3f}  {agg_ctc['SEG']:>8.3f}")

    print("\n" + "=" * 70)
    print("FIGURE A — Cell-HOTA curve across alpha values")
    print("=" * 70)
    print(
        f"  {'alpha':>8}"
        f"  {'Cell-HOTA':>12}"
        f"  {'DetA':>10}"
        f"  {'AssA':>10}"
        f"  {'DivA':>10}"
    )

    for _, row in hota_df.iterrows():
        print(
            f"  {row['alpha']:>8.2f}"
            f"  {row['Cell-HOTA']:>12.2f}"
            f"  {row['DetA']:>10.2f}"
            f"  {row['AssA']:>10.2f}"
            f"  {row['DivA']:>10.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--res_path", required=True)
    parser.add_argument("--output_dir", default="results/metrics")
    args = parser.parse_args()

    gt_path = Path(args.gt_path)
    res_path = Path(args.res_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Alpha range matching Figure A: 0.05 to 0.95 in steps of 0.05
    alphas = [round(a, 2) for a in np.arange(0.05, 1.00, 0.05)]

    pairs = find_sequence_pairs(gt_path, res_path)

    if not pairs:
        raise RuntimeError(
            f"No GT/RES pairs found.\n"
            f"  GT:  {gt_path}\n"
            f"  RES: {res_path}"
        )

    print(f"Found {len(pairs)} sequence(s):")
    for gt_folder, res_folder in pairs:
        print(f"  GT:  {gt_folder}")
        print(f"  RES: {res_folder}")

    all_seq = []

    for gt_folder, res_folder in pairs:
        print(f"\nProcessing {gt_folder.name} ...")
        all_seq.append(compute_sequence_metrics(gt_folder, res_folder, alphas))

    agg_ctc, agg_hota = aggregate(all_seq, alphas)

    alpha_curve_df = make_alpha_curve_df(agg_hota)
    summary_df = make_summary_df(agg_ctc, agg_hota)

    # Save JSON
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(
            {
                "CTC": agg_ctc,
                "CHOTA_by_alpha": agg_hota,
                "per_sequence": all_seq,
            },
            f,
            indent=2,
        )

    # Save CSVs
    alpha_curve_df.to_csv(out_dir / "cell_hota_curve.csv", index=False)
    pd.DataFrame([agg_ctc]).to_csv(out_dir / "ctc_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "metrics_summary_table.csv", index=False)

    print_summary(agg_ctc, agg_hota)

    print(f"\nResults saved to: {out_dir}/")
    print(f"  {out_dir / 'cell_hota_curve.csv'}")
    print(f"  {out_dir / 'ctc_metrics.csv'}")
    print(f"  {out_dir / 'metrics_summary_table.csv'}")
    print(f"  {out_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()