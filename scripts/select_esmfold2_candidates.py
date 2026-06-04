"""Select candidates from ESMFold2-Fast results for standard ESMFold2 review.

Inputs:
  * designability_top3.csv from collect_top_mpnn_sequences.py
  * esmfold2_results.csv from run_esmfold2_batch.py with the Fast model

The script joins both tables by design_id, sorts candidates within each
sampler/length group, and writes a smaller FASTA for the standard ESMFold2 run.

Example:
python scripts/select_esmfold2_candidates.py \
  --design-csv inference_outputs/weights/pdb/published/unconditional/designability_top3.csv \
  --fast-csv inference_outputs/weights/pdb/published/unconditional/esmfold2_fast_top3/esmfold2_results.csv \
  --top-per-group 20 \
  --out-fasta inference_outputs/weights/pdb/published/unconditional/standard_review_top20.fasta \
  --out-csv inference_outputs/weights/pdb/published/unconditional/standard_review_top20.csv
"""

import argparse
import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str, default: float) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "selection_group",
        "selection_rank",
        "design_id",
        "sampler",
        "length",
        "sample",
        "rank",
        "sequence",
        "sequence_length",
        "mpnn_score",
        "fast_plddt",
        "fast_ptm",
        "fast_iptm",
        "fast_runtime_seconds",
        "fast_output_cif",
        "backbone_pdb_path",
        "fasta_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_fasta(path: Path, rows: list[dict[str, str]], line_width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(
                f">{row['design_id']} "
                f"sampler={row['sampler']} length={row['length']} "
                f"sample={row['sample']} fast_rank={row['selection_rank']} "
                f"fast_plddt={row['fast_plddt']} fast_ptm={row['fast_ptm']} "
                f"mpnn_score={row['mpnn_score']}\n"
            )
            sequence = row["sequence"]
            for start in range(0, len(sequence), line_width):
                handle.write(sequence[start:start + line_width] + "\n")


def _sort_key(row: dict[str, str]) -> tuple[float, float, float]:
    # Higher confidence first, then lower ProteinMPNN score.
    return (
        -_float(row, "fast_plddt", default=-1.0),
        -_float(row, "fast_ptm", default=-1.0),
        _float(row, "mpnn_score", default=999999.0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select ESMFold2-Fast candidates for standard ESMFold2 review."
    )
    parser.add_argument("--design-csv", required=True)
    parser.add_argument("--fast-csv", required=True)
    parser.add_argument("--top-per-group", type=int, default=5)
    parser.add_argument("--min-plddt", type=float, default=None)
    parser.add_argument("--min-ptm", type=float, default=None)
    parser.add_argument("--out-fasta", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--line-width", type=int, default=80)
    args = parser.parse_args()

    design_rows = _read_csv(Path(args.design_csv).expanduser().resolve())
    fast_rows = _read_csv(Path(args.fast_csv).expanduser().resolve())
    fast_by_id = {row["design_id"]: row for row in fast_rows if row.get("status", "ok") == "ok"}

    joined: list[dict[str, str]] = []
    for design_row in design_rows:
        design_id = design_row["design_id"]
        fast_row = fast_by_id.get(design_id)
        if fast_row is None:
            continue

        fast_plddt = _float(fast_row, "plddt", default=-1.0)
        fast_ptm = _float(fast_row, "ptm", default=-1.0)
        if args.min_plddt is not None and fast_plddt < args.min_plddt:
            continue
        if args.min_ptm is not None and fast_ptm < args.min_ptm:
            continue

        joined.append(
            {
                "selection_group": f"{design_row['sampler']}_length_{design_row['length']}",
                "selection_rank": "",
                "design_id": design_id,
                "sampler": design_row["sampler"],
                "length": design_row["length"],
                "sample": design_row["sample"],
                "rank": design_row["rank"],
                "sequence": design_row["sequence"],
                "sequence_length": design_row["sequence_length"],
                "mpnn_score": design_row["mpnn_score"],
                "fast_plddt": fast_row.get("plddt", ""),
                "fast_ptm": fast_row.get("ptm", ""),
                "fast_iptm": fast_row.get("iptm", ""),
                "fast_runtime_seconds": fast_row.get("runtime_seconds", ""),
                "fast_output_cif": fast_row.get("output_cif", ""),
                "backbone_pdb_path": design_row["backbone_pdb_path"],
                "fasta_path": design_row["fasta_path"],
            }
        )

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in joined:
        grouped.setdefault(row["selection_group"], []).append(row)

    selected: list[dict[str, str]] = []
    for group in sorted(grouped):
        rows = sorted(grouped[group], key=_sort_key)
        for index, row in enumerate(rows[: args.top_per_group], start=1):
            row["selection_rank"] = str(index)
            selected.append(row)

    if not selected:
        raise RuntimeError("No candidates selected. Check input CSV paths and thresholds.")

    _write_fasta(Path(args.out_fasta).expanduser().resolve(), selected, args.line_width)
    _write_csv(Path(args.out_csv).expanduser().resolve(), selected)
    print(f"Joined {len(joined)} Fast-folded designs")
    print(f"Selected {len(selected)} candidates from {len(grouped)} sampler/length groups")
    print(f"Wrote FASTA to {args.out_fasta}")
    print(f"Wrote CSV to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
