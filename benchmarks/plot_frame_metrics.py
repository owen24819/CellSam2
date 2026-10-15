#!/usr/bin/env python3
"""Per-frame inference time vs cell count from shared_model_benchmark.py --frame_metrics.

Reads <results_dir>/frame_metrics/<model>_shared_nNNN_tTTT.csv (one file per thread/copy)
and, per (model, N, frame), aggregates across the N copies:
  mean_s / min_s / max_s : per-copy frame latency (wall clock, current-stream synced)
  span_s                 : max(t_end) - min(t_start), i.e. first copy starts -> last copy done
Writes frame_metrics_summary.csv and two figures:
  frame_time_vs_cells_per_N.{pdf,png}   one panel per N, x = GT cells in the frame, base vs small
  frame_time_vs_total_cells.{pdf,png}   one panel per model, x = N x GT cells (total load on the GPU)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType: editable text, accepted by journals
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

MODEL_LABELS = {"CellSam2-2D-bkg80": "base", "CellSam2-2D-small-bkg80": "small"}
MODEL_COLORS = {"base": "#2a78d6", "small": "#eb6834"}  # categorical slots 1, 2
N_RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"]  # blue, light -> dark
FNAME = re.compile(r"(?P<model>.+)_shared_n(?P<n>\d+)_t(?P<t>\d+)\.csv$")


def load(fm_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(fm_dir.glob("*_shared_n*_t*.csv")):
        m = FNAME.match(f.name)
        if m is None:
            continue
        df = pd.read_csv(f)
        df["model"] = MODEL_LABELS.get(m["model"], m["model"])
        df["n"] = int(m["n"])
        df["copy"] = int(m["t"])
        df["frame_idx"] = range(len(df))
        frames.append(df)
    if not frames:
        raise SystemExit(f"no frame metrics in {fm_dir}")
    return pd.concat(frames, ignore_index=True)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "n", "frame_idx", "frame"])
    out = g.agg(
        gt_cells=("gt_cells", "first"),
        detected_cells_mean=("detected_cells", "mean"),
        copies=("copy", "nunique"),
        mean_s=("seconds", "mean"),
        min_s=("seconds", "min"),
        max_s=("seconds", "max"),
        t_start_min=("t_start", "min"),
        t_end_max=("t_end", "max"),
    ).reset_index()
    out["span_s"] = out.t_end_max - out.t_start_min
    out["total_gt_cells"] = out.gt_cells * out.n
    return out.drop(columns=["t_start_min", "t_end_max"])


def style(ax) -> None:
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=9)


def save(fig, stem: Path, formats: list[str]) -> None:
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=150)
    plt.close(fig)


def plot_per_n(s: pd.DataFrame, stem: Path, formats: list[str]) -> None:
    ns = sorted(s.n.unique())
    ncols = min(4, len(ns))
    nrows = -(-len(ns) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), sharey=True,
                             squeeze=False)
    for ax in axes.flat[len(ns):]:
        ax.set_visible(False)
    for ax, n in zip(axes.flat, ns):
        for model, color in MODEL_COLORS.items():
            d = s[(s.n == n) & (s.model == model)].sort_values("frame_idx")
            if d.empty:
                continue
            yerr = None if n == 1 else [d.mean_s - d.min_s, d.max_s - d.mean_s]
            ax.errorbar(d.gt_cells, d.mean_s, yerr=yerr, color=color, fmt="o", markersize=4,
                        elinewidth=1, ecolor=color, alpha=0.85, label=model)
        ax.set_title(f"N = {n}", fontsize=11, color="#0b0b0b")
        ax.set_xlabel("GT cells in frame", color="#52514e")
        style(ax)
    for row in axes:
        row[0].set_ylabel("frame time per copy (s)", color="#52514e")
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle("Shared model, seq 17: per-frame time vs cell count\n"
                 "(dot = mean across the N copies, bar = min-max)", fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    save(fig, stem, formats)


def plot_total_cells(s: pd.DataFrame, stem: Path, formats: list[str]) -> None:
    models = [m for m in MODEL_COLORS if m in set(s.model)]
    ns = sorted(s.n.unique())
    # Spread the N values evenly over the ramp so a short N list still uses light -> dark.
    idx = [round(i * (len(N_RAMP) - 1) / max(1, len(ns) - 1)) for i in range(len(ns))]
    ramp = {n: N_RAMP[j] for n, j in zip(ns, idx)}
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4), sharey=True,
                             squeeze=False)
    for ax, model in zip(axes[0], models):
        for n in ns:
            d = s[(s.n == n) & (s.model == model)].sort_values("frame_idx")
            if d.empty:
                continue
            ax.plot(d.total_gt_cells, d.mean_s, color=ramp[n], linestyle="none", marker="o",
                    markersize=4, label=f"N = {n}")
        ax.set_title(model, fontsize=11, color="#0b0b0b")
        ax.set_xlabel("total GT cells across copies (N x cells in frame)", color="#52514e")
        style(ax)
        ax.legend(frameon=False, fontsize=9)
    axes[0][0].set_ylabel("mean frame time per copy (s)", color="#52514e")
    fig.suptitle("Shared model, seq 17:\nper-frame time vs total cell load on the GPU",
                 fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    save(fig, stem, formats)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default="results/parallel_bench_bkg80_seq17_shared_frametime")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"],
                   help="Figure formats to write (pdf is vector)")
    args = p.parse_args()
    rdir = Path(args.results_dir)

    s = summarise(load(rdir / "frame_metrics"))
    s.to_csv(rdir / "frame_metrics_summary.csv", index=False)
    plot_per_n(s, rdir / "frame_time_vs_cells_per_N", args.formats)
    plot_total_cells(s, rdir / "frame_time_vs_total_cells", args.formats)

    totals = s.assign(spread_s=s.max_s - s.min_s).groupby(["model", "n"]).agg(
        frames=("frame_idx", "size"),
        sum_mean_s=("mean_s", "sum"),
        median_frame_s=("mean_s", "median"),
        max_spread_s=("spread_s", "max"),
    )
    print(totals.round(3).to_string())
    print(f"wrote {rdir}/frame_metrics_summary.csv and figures")


if __name__ == "__main__":
    main()
