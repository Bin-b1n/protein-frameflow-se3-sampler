"""Plot designability comparisons between FrameFlow samplers.

Input is the per-design CSV produced by analyze_designability.py. The script
writes presentation-ready PNG figures comparing Euler-5 and Heun-5 by folding
confidence, folded-vs-backbone agreement, and designable rate.

Example:
python scripts/plot_designability.py \
  --designs-csv inference_outputs/weights/pdb/published/unconditional/designability_standard_top20_designs.csv \
  --out-dir inference_outputs/weights/pdb/published/unconditional/designability_figures
"""

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "euler": "#2F6BFF",
    "heun": "#E76F51",
    "ab2": "#2A9D8F",
}


def _sampler_method(sampler: str) -> str:
    return sampler.split("_", 1)[0].lower()


def _sampler_label(sampler: str) -> str:
    match = re.match(r"^(?P<method>[A-Za-z0-9]+)_(?P<steps>\d+)(?:_n(?P<n>\d+))?$", sampler)
    if match is None:
        return sampler
    method = match.group("method").capitalize()
    steps = match.group("steps")
    n = match.group("n")
    suffix = f", n={n}" if n else ""
    return f"{method} {steps}{suffix}"


def _sampler_color(sampler: str) -> str:
    return PALETTE.get(_sampler_method(sampler), "#6C757D")


def _prepare_df(path: Path, min_plddt: float, min_ptm: float, max_sc_rmsd: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sc_rmsd" not in df.columns and "ca_rmsd" in df.columns:
        df["sc_rmsd"] = df["ca_rmsd"]
    numeric_cols = [
        "length",
        "mpnn_score",
        "standard_plddt",
        "standard_ptm",
        "standard_iptm",
        "sc_rmsd",
        "ca_rmsd",
        "tm_norm_folded",
        "tm_norm_backbone",
        "designable",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["sampler", "length"])
    if "standard_plddt" in df.columns and df["standard_plddt"].max(skipna=True) <= 1.5:
        df["standard_plddt"] = df["standard_plddt"] * 100.0
    df["designable"] = (
        (df["standard_plddt"] >= min_plddt)
        & (df["standard_ptm"] >= min_ptm)
        & (df["sc_rmsd"] <= max_sc_rmsd)
    ).astype(int)
    df["sampler_label"] = df["sampler"].map(_sampler_label)
    df["sampler_color"] = df["sampler"].map(_sampler_color)
    return df


def _network_calls(sampler: str) -> int | None:
    match = re.match(r"^(?P<method>euler|heun|ab2)_(?P<steps>\d+)", sampler)
    if match is None:
        return None
    method = match.group("method")
    steps = int(match.group("steps"))
    if method == "heun":
        return 2 * (steps - 1) + 1
    return steps


def _runtime_seconds(runtime_root: Path | None, sampler: str) -> float | None:
    if runtime_root is None:
        return None
    runtime_path = runtime_root / sampler / "runtime_seconds.txt"
    if not runtime_path.is_file():
        return None
    text = runtime_path.read_text().strip()
    if not text:
        return None
    return float(text)


def _metric_sem(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def _annotate_bars(ax, bars, fmt: str) -> None:
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_overview(df: pd.DataFrame, out_dir: Path, show_error_bars: bool) -> Path:
    metrics = [
        ("standard_plddt", "Mean pLDDT", "pLDDT", "{:.1f}"),
        ("standard_ptm", "Mean pTM", "pTM", "{:.2f}"),
        ("sc_rmsd", "Mean scRMSD", "Angstrom", "{:.2f}"),
        ("designable", "Designable Rate", "Fraction", "{:.2f}"),
    ]
    samplers = sorted(df["sampler"].unique())
    labels = [_sampler_label(sampler) for sampler in samplers]
    colors = [_sampler_color(sampler) for sampler in samplers]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), dpi=180)
    axes = axes.flatten()
    for ax, (col, title, ylabel, fmt) in zip(axes, metrics):
        means = [df.loc[df["sampler"] == sampler, col].mean() for sampler in samplers]
        sems = [_metric_sem(df.loc[df["sampler"] == sampler, col]) for sampler in samplers]
        yerr = sems if show_error_bars else None
        bars = ax.bar(labels, means, yerr=yerr, color=colors, capsize=4, width=0.62)
        _annotate_bars(ax, bars, fmt)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        if col in {"standard_ptm", "designable"}:
            ax.set_ylim(0, max(1.0, max(means) * 1.18 if means else 1.0))
        if col == "standard_plddt":
            ax.set_ylim(0, 100)
        for tick in ax.get_xticklabels():
            tick.set_rotation(10)
            tick.set_ha("right")

    fig.suptitle("Designability Comparison: Standard ESMFold2 Review", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = out_dir / "designability_sampler_overview.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_by_length(df: pd.DataFrame, out_dir: Path) -> Path:
    metrics = [
        ("standard_plddt", "pLDDT", "pLDDT"),
        ("standard_ptm", "pTM", "pTM"),
        ("sc_rmsd", "scRMSD", "Angstrom"),
        ("designable", "Designable Rate", "Fraction"),
    ]
    samplers = sorted(df["sampler"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), dpi=180)
    axes = axes.flatten()

    for ax, (col, title, ylabel) in zip(axes, metrics):
        for sampler in samplers:
            group = (
                df[df["sampler"] == sampler]
                .groupby("length", as_index=False)[col]
                .mean()
                .sort_values("length")
            )
            ax.plot(
                group["length"],
                group[col],
                marker="o",
                linewidth=2.2,
                label=_sampler_label(sampler),
                color=_sampler_color(sampler),
            )
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("Backbone length")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if col == "standard_plddt":
            ax.set_ylim(0, 100)
        if col in {"standard_ptm", "designable"}:
            ax.set_ylim(0, 1.0)

    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Designability Metrics by Backbone Length", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = out_dir / "designability_by_length.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_confidence_vs_rmsd(
    df: pd.DataFrame,
    out_dir: Path,
    min_plddt: float,
    max_sc_rmsd: float,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 5.6), dpi=180)
    for sampler, group in df.groupby("sampler"):
        designable = group["designable"].fillna(0).astype(int) == 1
        ax.scatter(
            group.loc[~designable, "sc_rmsd"],
            group.loc[~designable, "standard_plddt"],
            s=42,
            color=_sampler_color(sampler),
            alpha=0.35,
            label=f"{_sampler_label(sampler)} not pass",
            marker="o",
            linewidths=0,
        )
        ax.scatter(
            group.loc[designable, "sc_rmsd"],
            group.loc[designable, "standard_plddt"],
            s=58,
            color=_sampler_color(sampler),
            alpha=0.9,
            label=f"{_sampler_label(sampler)} pass",
            marker="o",
            edgecolors="black",
            linewidths=0.55,
        )
    ax.axhline(min_plddt, color="#495057", linestyle="--", linewidth=1.2)
    ax.axvline(max_sc_rmsd, color="#495057", linestyle="--", linewidth=1.2)
    ax.text(max_sc_rmsd, ax.get_ylim()[1], f"  scRMSD <= {max_sc_rmsd:g}", va="top", fontsize=8)
    ax.text(ax.get_xlim()[1], min_plddt, f"pLDDT >= {min_plddt:g}  ", ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("Self-consistency RMSD, folded vs generated backbone (Angstrom)")
    ax.set_ylabel("Standard ESMFold2 pLDDT")
    ax.set_title("Confidence vs Backbone Recovery", fontsize=13, weight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    out_path = out_dir / "plddt_vs_sc_rmsd.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_mpnn_vs_recovery(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 5.6), dpi=180)
    for sampler, group in df.groupby("sampler"):
        ax.scatter(
            group["mpnn_score"],
            group["sc_rmsd"],
            s=48,
            color=_sampler_color(sampler),
            alpha=0.75,
            label=_sampler_label(sampler),
        )
    ax.set_xlabel("ProteinMPNN score (lower is better)")
    ax.set_ylabel("Self-consistency RMSD (Angstrom)")
    ax.set_title("Sequence Score vs Backbone Recovery", fontsize=13, weight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = out_dir / "mpnn_score_vs_sc_rmsd.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def write_plot_summary(df: pd.DataFrame, out_dir: Path, runtime_root: Path | None) -> Path:
    summary = (
        df.groupby("sampler", as_index=False)
        .agg(
            count=("design_id", "count"),
            designable_count=("designable", "sum"),
            mpnn_score_mean=("mpnn_score", "mean"),
            standard_plddt_mean=("standard_plddt", "mean"),
            standard_ptm_mean=("standard_ptm", "mean"),
            sc_rmsd_mean=("sc_rmsd", "mean"),
            tm_norm_backbone_mean=("tm_norm_backbone", "mean"),
            designable_rate=("designable", "mean"),
        )
    )
    summary["sampler_label"] = summary["sampler"].map(_sampler_label)
    summary["runtime_seconds"] = summary["sampler"].map(lambda sampler: _runtime_seconds(runtime_root, sampler))
    summary["network_calls"] = summary["sampler"].map(_network_calls)
    summary["designable_per_second"] = summary["designable_count"] / summary["runtime_seconds"]
    summary["designable_rate_per_second"] = summary["designable_rate"] / summary["runtime_seconds"]
    summary["designable_per_network_call"] = summary["designable_count"] / summary["network_calls"]
    summary["designable_rate_per_network_call"] = summary["designable_rate"] / summary["network_calls"]
    out_path = out_dir / "plot_summary_by_sampler.csv"
    summary.to_csv(out_path, index=False)
    return out_path


def plot_runtime_normalized(df: pd.DataFrame, out_dir: Path, runtime_root: Path | None) -> Path | None:
    if runtime_root is None:
        return None
    summary = (
        df.groupby("sampler", as_index=False)
        .agg(
            designable_count=("designable", "sum"),
            designable_rate=("designable", "mean"),
        )
    )
    summary["runtime_seconds"] = summary["sampler"].map(lambda sampler: _runtime_seconds(runtime_root, sampler))
    summary["network_calls"] = summary["sampler"].map(_network_calls)
    summary = summary.dropna(subset=["runtime_seconds", "network_calls"])
    if summary.empty:
        return None
    summary["designable_per_second"] = summary["designable_count"] / summary["runtime_seconds"]
    summary["designable_rate_per_second"] = summary["designable_rate"] / summary["runtime_seconds"]
    summary["designable_per_network_call"] = summary["designable_count"] / summary["network_calls"]
    summary["designable_rate_per_network_call"] = summary["designable_rate"] / summary["network_calls"]

    samplers = summary["sampler"].tolist()
    labels = [_sampler_label(sampler) for sampler in samplers]
    colors = [_sampler_color(sampler) for sampler in samplers]
    metrics = [
        ("runtime_seconds", "Sampling Runtime", "Seconds", "{:.1f}"),
        ("designable_rate_per_second", "Designable Rate / Sampling Second", "Rate / s", "{:.4f}"),
        ("designable_rate_per_network_call", "Designable Rate / Network Call", "Rate / call", "{:.4f}"),
        ("designable_per_second", "Reviewed Passes / Sampling Second", "Passes / s", "{:.3f}"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), dpi=180)
    axes = axes.flatten()
    for ax, (col, title, ylabel, fmt) in zip(axes, metrics):
        values = summary[col].tolist()
        bars = ax.bar(labels, values, color=colors, width=0.62)
        _annotate_bars(ax, bars, fmt)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for tick in ax.get_xticklabels():
            tick.set_rotation(10)
            tick.set_ha("right")

    fig.suptitle("Runtime-Normalized Designability", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = out_dir / "runtime_normalized_designability.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot designability comparison figures.")
    parser.add_argument("--designs-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-plddt", type=float, default=70.0)
    parser.add_argument("--min-ptm", type=float, default=0.5)
    parser.add_argument(
        "--max-sc-rmsd",
        "--max-ca-rmsd",
        dest="max_sc_rmsd",
        type=float,
        default=2.0,
        help="Maximum self-consistency RMSD threshold. --max-ca-rmsd is kept as a compatibility alias.",
    )
    parser.add_argument(
        "--runtime-root",
        default=None,
        help="Directory containing sampler run folders with runtime_seconds.txt.",
    )
    parser.add_argument(
        "--no-error-bars",
        action="store_true",
        help="Hide standard-error bars on sampler overview bar charts.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = Path(args.runtime_root).expanduser().resolve() if args.runtime_root else None
    df = _prepare_df(
        Path(args.designs_csv).expanduser().resolve(),
        args.min_plddt,
        args.min_ptm,
        args.max_sc_rmsd,
    )

    paths = [
        write_plot_summary(df, out_dir, runtime_root),
        plot_overview(df, out_dir, show_error_bars=not args.no_error_bars),
        plot_by_length(df, out_dir),
        plot_confidence_vs_rmsd(df, out_dir, args.min_plddt, args.max_sc_rmsd),
        plot_mpnn_vs_recovery(df, out_dir),
        plot_runtime_normalized(df, out_dir, runtime_root),
    ]
    for path in paths:
        if path is not None:
            print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
