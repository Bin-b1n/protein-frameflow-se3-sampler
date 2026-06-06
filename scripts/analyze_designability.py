"""Analyze designability from ProteinMPNN and ESMFold2 review results.

The script joins selected ProteinMPNN candidates with standard ESMFold2 results,
computes self-consistency RMSD (scRMSD) between the folded structure and the
generated backbone, then writes
per-design and grouped summary CSVs.

Example:
python scripts/analyze_designability.py \
  --selection-csv inference_outputs/weights/pdb/published/unconditional/standard_review_top20.csv \
  --fold-csv inference_outputs/weights/pdb/published/unconditional/esmfold2_standard_top20/esmfold2_results.csv \
  --out-designs inference_outputs/weights/pdb/published/unconditional/designability_standard_top20_designs.csv \
  --out-summary inference_outputs/weights/pdb/published/unconditional/designability_standard_top20_summary.csv
"""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _format(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{value:.6f}"


def _normalize_plddt(value: float) -> float:
    if math.isnan(value):
        return value
    if value <= 1.5:
        return value * 100.0
    return value


def _load_ca_positions(path: Path) -> np.ndarray:
    import mdtraj as md

    traj = md.load(str(path))
    ca_indices = traj.topology.select("name CA")
    if len(ca_indices) == 0:
        raise ValueError(f"No CA atoms found in {path}")
    return traj.xyz[0, ca_indices, :] * 10.0


def _kabsch_rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    ref_centered = reference - reference.mean(axis=0, keepdims=True)
    mob_centered = mobile - mobile.mean(axis=0, keepdims=True)
    covariance = mob_centered.T @ ref_centered
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    aligned = mob_centered @ rotation
    diff = aligned - ref_centered
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))


def _tm_score(reference: np.ndarray, mobile: np.ndarray, sequence: str) -> tuple[float, float]:
    try:
        from tmtools import tm_align
    except ImportError:
        return math.nan, math.nan
    length = min(len(reference), len(mobile), len(sequence))
    if length < 3:
        return math.nan, math.nan
    result = tm_align(
        mobile[:length],
        reference[:length],
        sequence[:length],
        sequence[:length],
    )
    return float(result.tm_norm_chain1), float(result.tm_norm_chain2)


def _metric_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = _float(row, key)
        if not math.isnan(value):
            values.append(value)
    return values


def _summarize_group(group_name: str, group_value: str, rows: list[dict[str, str]]) -> dict[str, str]:
    summary = {
        "group": group_name,
        "value": group_value,
        "count": str(len(rows)),
    }
    for key in [
        "mpnn_score",
        "standard_plddt",
        "standard_ptm",
        "sc_rmsd",
        "ca_rmsd",
        "tm_norm_folded",
        "tm_norm_backbone",
    ]:
        values = _metric_values(rows, key)
        if values:
            summary[f"{key}_mean"] = _format(mean(values))
            summary[f"{key}_median"] = _format(median(values))
            summary[f"{key}_min"] = _format(min(values))
            summary[f"{key}_max"] = _format(max(values))
        else:
            summary[f"{key}_mean"] = ""
            summary[f"{key}_median"] = ""
            summary[f"{key}_min"] = ""
            summary[f"{key}_max"] = ""
    designable = [row["designable"] == "1" for row in rows]
    summary["designable_count"] = str(sum(designable))
    summary["designable_rate"] = _format(sum(designable) / len(designable))
    return summary


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge ESMFold2 results and compute backbone designability metrics."
    )
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--fold-csv", required=True)
    parser.add_argument("--out-designs", required=True)
    parser.add_argument("--out-summary", required=True)
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
        "--skip-tm-score",
        action="store_true",
        help="Skip tmtools TM-score calculation and only report scRMSD.",
    )
    args = parser.parse_args()

    selection_rows = _read_csv(Path(args.selection_csv).expanduser().resolve())
    fold_rows = _read_csv(Path(args.fold_csv).expanduser().resolve())
    fold_by_id = {row["design_id"]: row for row in fold_rows if row.get("status", "ok") == "ok"}

    design_rows: list[dict[str, str]] = []
    for selection_row in selection_rows:
        design_id = selection_row["design_id"]
        fold_row = fold_by_id.get(design_id)
        if fold_row is None:
            continue

        backbone_path = Path(selection_row["backbone_pdb_path"]).expanduser()
        folded_path = Path(fold_row["output_cif"]).expanduser()
        if not backbone_path.is_absolute():
            backbone_path = backbone_path.resolve()
        if not folded_path.is_absolute():
            folded_path = folded_path.resolve()

        backbone_ca = _load_ca_positions(backbone_path)
        folded_ca = _load_ca_positions(folded_path)
        length = min(len(backbone_ca), len(folded_ca))
        if length < 3:
            raise ValueError(f"Too few matched CA atoms for {design_id}")
        sc_rmsd = _kabsch_rmsd(backbone_ca[:length], folded_ca[:length])

        if args.skip_tm_score:
            tm_folded = tm_backbone = math.nan
        else:
            tm_folded, tm_backbone = _tm_score(
                backbone_ca[:length],
                folded_ca[:length],
                selection_row["sequence"][:length],
            )

        plddt = _normalize_plddt(_float(fold_row, "plddt"))
        ptm = _float(fold_row, "ptm")
        designable = (
            not math.isnan(plddt)
            and not math.isnan(ptm)
            and plddt >= args.min_plddt
            and ptm >= args.min_ptm
            and sc_rmsd <= args.max_sc_rmsd
        )

        design_rows.append(
            {
                "design_id": design_id,
                "sampler": selection_row["sampler"],
                "length": selection_row["length"],
                "sample": selection_row["sample"],
                "selection_rank": selection_row.get("selection_rank", ""),
                "mpnn_rank": selection_row.get("rank", ""),
                "mpnn_score": selection_row["mpnn_score"],
                "standard_plddt": _format(plddt),
                "standard_ptm": fold_row.get("ptm", ""),
                "standard_iptm": fold_row.get("iptm", ""),
                "sc_rmsd": _format(sc_rmsd),
                # Kept for backward compatibility with earlier local notebooks.
                "ca_rmsd": _format(sc_rmsd),
                "tm_norm_folded": _format(tm_folded),
                "tm_norm_backbone": _format(tm_backbone),
                "designable": "1" if designable else "0",
                "matched_ca_count": str(length),
                "sequence_length": selection_row["sequence_length"],
                "backbone_pdb_path": str(backbone_path),
                "folded_cif_path": str(folded_path),
            }
        )

    if not design_rows:
        raise RuntimeError("No matching successful ESMFold2 designs found.")

    summary_rows: list[dict[str, str]] = []
    summary_rows.append(_summarize_group("overall", "all", design_rows))

    samplers = sorted({row["sampler"] for row in design_rows})
    for sampler in samplers:
        rows = [row for row in design_rows if row["sampler"] == sampler]
        summary_rows.append(_summarize_group("sampler", sampler, rows))

    sampler_lengths = sorted({(row["sampler"], row["length"]) for row in design_rows})
    for sampler, length in sampler_lengths:
        rows = [
            row for row in design_rows
            if row["sampler"] == sampler and row["length"] == length
        ]
        summary_rows.append(_summarize_group("sampler_length", f"{sampler}_length_{length}", rows))

    _write_csv(Path(args.out_designs).expanduser().resolve(), design_rows)
    _write_csv(Path(args.out_summary).expanduser().resolve(), summary_rows)
    print(f"Wrote per-design metrics to {args.out_designs}")
    print(f"Wrote grouped summary to {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
