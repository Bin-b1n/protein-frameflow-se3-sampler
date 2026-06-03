"""Plot quality-efficiency curves from FrameFlow benchmark outputs.

Each run directory should contain:
  metrics.csv
  runtime_seconds.txt  # optional when plotting with --x runtime

Example:
python analysis/plot_quality_efficiency.py \
  --root inference_outputs/weights/pdb/published/unconditional \
  --tag n20 \
  --out-dir inference_outputs/weights/pdb/published/unconditional/quality_efficiency_n20
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUN_RE = re.compile(r"^(?P<method>euler|heun|ab2)_(?P<timesteps>\d+)(?:_(?P<tag>.+))?$")


def _read_runtime(run_dir: Path):
    runtime_path = run_dir / "runtime_seconds.txt"
    if not runtime_path.exists():
        return None
    text = runtime_path.read_text().strip()
    return float(text) if text else None


def _network_calls(method: str, timesteps: int) -> int:
    if method == "heun":
        return 2 * (timesteps - 1) + 1
    return timesteps


def collect_runs(root: Path, tag: str | None):
    rows = []
    for metrics_path in sorted(root.glob("*/metrics.csv")):
        run_dir = metrics_path.parent
        match = RUN_RE.match(run_dir.name)
        if match is None:
            continue
        method = match.group("method")
        timesteps = int(match.group("timesteps"))
        run_tag = match.group("tag")
        if tag is not None and run_tag != tag:
            continue

        df = pd.read_csv(metrics_path)
        row = {
            "run": run_dir.name,
            "method": method,
            "timesteps": timesteps,
            "network_calls": _network_calls(method, timesteps),
            "samples": len(df),
            "runtime_seconds": _read_runtime(run_dir),
        }
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            row[col] = df[col].mean()
        rows.append(row)
    if not rows:
        suffix = f" with tag={tag}" if tag is not None else ""
        raise FileNotFoundError(f"No benchmark metrics found under {root}{suffix}")
    return pd.DataFrame(rows).sort_values(["method", "timesteps"])


def _x_values(df: pd.DataFrame, x_name: str):
    if x_name == "runtime":
        if df["runtime_seconds"].isna().any():
            missing = ", ".join(df[df["runtime_seconds"].isna()]["run"].tolist())
            raise ValueError(
                "Missing runtime_seconds.txt for: "
                f"{missing}. Use --x timesteps or --x calls, or rerun with the benchmark script."
            )
        return "runtime_seconds", "Wall time (s)"
    if x_name == "timesteps":
        return "timesteps", "Sampling timesteps"
    if x_name == "calls":
        return "network_calls", "Network calls"
    raise ValueError(f"Unknown x-axis: {x_name}")


def plot_metric(df: pd.DataFrame, metric: str, x_name: str, out_dir: Path):
    x_col, x_label = _x_values(df, x_name)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    for method, group in df.groupby("method"):
        group = group.sort_values(x_col)
        ax.plot(
            group[x_col],
            group[metric],
            marker="o",
            linewidth=2,
            label=method,
        )
        for _, row in group.iterrows():
            ax.annotate(
                str(int(row["timesteps"])),
                (row[x_col], row[metric]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {x_label}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f"{metric}_vs_{x_name}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Directory containing euler/heun/ab2 run folders.")
    parser.add_argument("--tag", default=None, help="Only include runs named method_steps_TAG.")
    parser.add_argument("--out-dir", required=True, help="Directory for summary CSV and plots.")
    parser.add_argument(
        "--x",
        choices=["runtime", "timesteps", "calls"],
        default="runtime",
        help="Efficiency axis. runtime requires runtime_seconds.txt in each run folder.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "ca_ca_deviation",
            "ca_ca_bad_percent",
            "ca_ca_valid_percent",
            "radius_of_gyration",
            "helix_percent",
            "coil_percent",
        ],
        help="Metric columns to plot.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = collect_runs(Path(args.root), args.tag)
    summary_path = out_dir / "benchmark_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")

    available_metrics = [metric for metric in args.metrics if metric in df.columns]
    missing_metrics = sorted(set(args.metrics) - set(available_metrics))
    if missing_metrics:
        print(f"Skipping missing metrics: {', '.join(missing_metrics)}")

    for metric in available_metrics:
        out_path = plot_metric(df, metric, args.x, out_dir)
        print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
