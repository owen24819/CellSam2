#!/usr/bin/env python3
"""Summarise and plot parallel_positions.csv / shared_model.csv / batched_encoder.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def fit_vram(ok: pd.DataFrame, headroom: float = 0.95) -> tuple[float, float, float]:
    """Linear fit peak_vram = fixed + marginal * N; returns (fixed, marginal, projected max N)."""
    if ok.n_parallel.nunique() < 2:
        return float("nan"), float("nan"), float("nan")
    marginal, fixed = np.polyfit(ok.n_parallel, ok.peak_gpu_mib, 1)
    total = ok.gpu_total_mib.iloc[0]
    return fixed, marginal, float(np.floor((total * headroom - fixed) / marginal))


def summarise_positions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, model), g in df.groupby(["mode", "model_name"]):
        ok = g[g.status == "ok"]
        fail = g[g.status != "ok"]
        best = ok.loc[ok.aggregate_fps.idxmax()] if len(ok) else None
        fixed, marginal, proj = fit_vram(ok)
        rows.append(
            dict(
                mode=mode,
                model=model,
                max_parallel_ok=int(ok.n_parallel.max()) if len(ok) else 0,
                first_fail_n=int(fail.n_parallel.min()) if len(fail) else None,
                first_fail_status=fail.sort_values("n_parallel").status.iloc[0] if len(fail) else "",
                fps_at_n1=float(ok[ok.n_parallel == 1].aggregate_fps.iloc[0]) if (ok.n_parallel == 1).any() else None,
                best_aggregate_fps=float(best.aggregate_fps) if best is not None else None,
                n_at_best_fps=int(best.n_parallel) if best is not None else None,
                vram_fixed_mib=round(fixed),
                vram_marginal_per_pos_mib=round(marginal),
                projected_max_n_vram_fit=proj,
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/parallel_bench")
    args = p.parse_args()
    rdir = Path(args.results_dir)

    frames = []
    if (rdir / "parallel_positions.csv").exists():
        frames.append(pd.read_csv(rdir / "parallel_positions.csv").assign(mode="multi_process"))
    if (rdir / "shared_model.csv").exists():
        frames.append(pd.read_csv(rdir / "shared_model.csv"))
    if frames:
        df = pd.concat(frames, ignore_index=True)
        # keep the latest row for each (mode, model, N) if a sweep was re-run
        df = df.drop_duplicates(subset=["mode", "model_name", "n_parallel"], keep="last").sort_values("n_parallel")
        summary = summarise_positions(df)
        print(summary.to_string(index=False))
        summary.to_csv(rdir / "parallel_positions_summary.csv", index=False)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for (mode, model), g in df.groupby(["mode", "model_name"]):
            ok = g[g.status == "ok"]
            ls = "o-" if mode == "multi_process" else "s--"
            lbl = f"{model} [{mode}]"
            axes[0].plot(ok.n_parallel, ok.aggregate_fps, ls, label=lbl)
            axes[1].plot(ok.n_parallel, ok.per_position_fps_mean, ls, label=lbl)
            axes[2].plot(ok.n_parallel, ok.peak_gpu_mib / 1024, ls, label=lbl)
            for _, r in g[g.status != "ok"].iterrows():
                axes[2].axvline(r.n_parallel, ls=":", color=axes[2].lines[-1].get_color(), alpha=0.6)
        if len(df):
            axes[2].axhline(df.gpu_total_mib.iloc[0] / 1024, color="k", ls="--", lw=1, label="GPU total")
        axes[0].set(xlabel="parallel positions N", ylabel="aggregate frames / s", title="Total throughput")
        axes[1].set(xlabel="parallel positions N", ylabel="frames / s per position", title="Per-position speed")
        axes[2].set(xlabel="parallel positions N", ylabel="peak VRAM (GiB)", title="GPU memory (dotted = failure)")
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(rdir / "parallel_positions.png", dpi=150)
        print(f"saved {rdir / 'parallel_positions.png'}")

    enc_csv = rdir / "batched_encoder.csv"
    if enc_csv.exists():
        de = pd.read_csv(enc_csv)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        for (model, dtype), g in de.groupby(["model_name", "dtype"]):
            ok = g[g.status == "ok"].sort_values("batch_size")
            lbl = f"{model} ({dtype})"
            ax[0].plot(ok.batch_size, ok.crops_per_s, "o-", label=lbl)
            ax[1].plot(ok.batch_size, ok.peak_reserved_mib / 1024, "o-", label=lbl)
            print(f"{lbl}: max batch = {ok.batch_size.max()} crops, "
                  f"best {ok.crops_per_s.max():.0f} crops/s")
        ax[0].set(xlabel="batch size (512px crops)", ylabel="crops / s", title="Image encoder throughput")
        ax[1].set(xlabel="batch size (512px crops)", ylabel="peak reserved (GiB)", title="Image encoder memory")
        for a in ax:
            a.set_xscale("log", base=2)
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(rdir / "batched_encoder.png", dpi=150)
        print(f"saved {rdir / 'batched_encoder.png'}")


if __name__ == "__main__":
    main()
