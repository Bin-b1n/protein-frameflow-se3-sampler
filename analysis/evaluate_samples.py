"""Evaluate generated FrameFlow sample.pdb files.

Example:
python analysis/evaluate_samples.py \
  --root inference_outputs/weights/pdb/published/unconditional/euler_5 \
  --out inference_outputs/weights/pdb/published/unconditional/euler_5/metrics.csv
"""

import argparse
import csv
from pathlib import Path

import mdtraj as md
import numpy as np

from analysis import metrics
from openfold.np import residue_constants


def _parse_length_and_sample(path: Path):
    length = None
    sample = None
    for part in path.parts:
        if part.startswith("length_"):
            length = int(part.removeprefix("length_"))
        elif part.startswith("sample_"):
            sample = part.removeprefix("sample_")
    return length, sample


def _load_ca_positions(pdb_path: Path) -> np.ndarray:
    traj = md.load(str(pdb_path))
    ca_indices = traj.topology.select("name CA")
    if len(ca_indices) == 0:
        raise ValueError(f"No CA atoms found in {pdb_path}")
    return traj.xyz[0, ca_indices, :] * 10.0  # mdtraj uses nm; convert to Angstrom.


def _ca_ca_summary(ca_pos: np.ndarray, threshold: float):
    bond_dists = np.linalg.norm(ca_pos[1:] - ca_pos[:-1], axis=-1)
    deviations = np.abs(bond_dists - residue_constants.ca_ca)
    return {
        "ca_ca_mean": float(np.mean(bond_dists)),
        "ca_ca_min": float(np.min(bond_dists)),
        "ca_ca_max": float(np.max(bond_dists)),
        "ca_ca_deviation_max": float(np.max(deviations)),
        "ca_ca_bad_count": int(np.sum(deviations > threshold)),
        "ca_ca_bad_percent": float(np.mean(deviations > threshold)),
    }


def evaluate_file(pdb_path: Path, ca_bond_bad_threshold: float):
    ca_pos = _load_ca_positions(pdb_path)
    mdtraj_metrics = metrics.calc_mdtraj_metrics(str(pdb_path))
    ca_metrics = metrics.calc_ca_ca_metrics(ca_pos)
    ca_summary = _ca_ca_summary(ca_pos, ca_bond_bad_threshold)
    length, sample = _parse_length_and_sample(pdb_path)
    return {
        "path": str(pdb_path),
        "length": length,
        "sample": sample,
        **mdtraj_metrics,
        **ca_metrics,
        **ca_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Directory containing generated sample.pdb files.")
    parser.add_argument("--out", required=True, help="CSV output path.")
    parser.add_argument(
        "--ca-bond-bad-threshold",
        type=float,
        default=0.2,
        help="Angstrom deviation from ideal CA-CA distance counted as abnormal.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    sample_paths = sorted(root.rglob("sample.pdb"))
    if not sample_paths:
        raise FileNotFoundError(f"No sample.pdb files found under {root}")

    rows = [evaluate_file(path, args.ca_bond_bad_threshold) for path in sample_paths]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    keys = [
        "ca_ca_deviation",
        "ca_ca_valid_percent",
        "num_ca_ca_clashes",
        "radius_of_gyration",
        "helix_percent",
        "strand_percent",
        "coil_percent",
    ]
    print(f"Wrote {len(rows)} rows to {out_path}")
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=float)
        print(f"{key}: mean={values.mean():.4f}, min={values.min():.4f}, max={values.max():.4f}")


if __name__ == "__main__":
    main()
