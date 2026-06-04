"""Collect top ProteinMPNN sequences into one FASTA for folding.

The script reads one or more proteinmpnn_manifest.csv files produced by
run_proteinmpnn_on_samples.py, parses each sample.fa, keeps the lowest-score
ProteinMPNN designs per backbone, and writes a combined FASTA plus a metadata
CSV for downstream ESMFold2 runs.

Example:
python scripts/collect_top_mpnn_sequences.py \
  --manifest inference_outputs/weights/pdb/published/unconditional/euler_5_n20/proteinmpnn/proteinmpnn_manifest.csv \
  --manifest inference_outputs/weights/pdb/published/unconditional/heun_5_n20/proteinmpnn/proteinmpnn_manifest.csv \
  --top-k 3 \
  --out-fasta inference_outputs/weights/pdb/published/unconditional/designability_top3.fasta \
  --out-csv inference_outputs/weights/pdb/published/unconditional/designability_top3.csv
"""

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


HEADER_PAIR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^,]+)")
VALID_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYX]+$", re.IGNORECASE)


@dataclass
class FastaRecord:
    header: str
    sequence: str
    metadata: dict[str, str]
    order: int


def _read_fasta(path: Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header = None
    seq_lines: list[str] = []

    def flush() -> None:
        nonlocal header, seq_lines
        if header is None:
            return
        sequence = "".join(seq_lines).replace(" ", "").upper()
        records.append(
            FastaRecord(
                header=header,
                sequence=sequence,
                metadata=_parse_header_metadata(header),
                order=len(records),
            )
        )
        header = None
        seq_lines = []

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
            else:
                seq_lines.append(line)
    flush()
    return records


def _parse_header_metadata(header: str) -> dict[str, str]:
    return {
        match.group(1).strip(): match.group(2).strip().strip("'\"")
        for match in HEADER_PAIR_RE.finditer(header)
    }


def _score(record: FastaRecord) -> float:
    for key in ("score", "global_score"):
        value = record.metadata.get(key)
        if value is not None:
            return float(value)
    return float("inf")


def _is_design_record(record: FastaRecord, include_native: bool) -> bool:
    if include_native:
        return True
    header = record.header.strip()
    if header.startswith("T=") or "sample" in record.metadata:
        return True
    return False


def _safe_id(raw: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "sequence"


def _sampler_from_path(path: Path) -> str:
    for part in path.parts:
        if re.match(r"^(euler|heun|ab2)_\d+(_n\d+)?$", part):
            return part
    return "unknown_sampler"


def _length_and_sample_from_path(path: Path) -> tuple[str, str]:
    length = "unknown_length"
    sample = "unknown_sample"
    for part in path.parts:
        if part.startswith("length_"):
            length = part.removeprefix("length_")
        elif part.startswith("sample_"):
            sample = part.removeprefix("sample_")
    return length, sample


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_fasta(path: Path, rows: list[dict[str, str]], line_width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(
                f">{row['design_id']} "
                f"sampler={row['sampler']} length={row['length']} "
                f"sample={row['sample']} rank={row['rank']} "
                f"mpnn_score={row['mpnn_score']}\n"
            )
            sequence = row["sequence"]
            for start in range(0, len(sequence), line_width):
                handle.write(sequence[start:start + line_width] + "\n")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "design_id",
        "sampler",
        "length",
        "sample",
        "rank",
        "mpnn_score",
        "mpnn_global_score",
        "mpnn_temperature",
        "mpnn_sample",
        "sequence",
        "sequence_length",
        "fasta_path",
        "backbone_pdb_path",
        "proteinmpnn_header",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect top ProteinMPNN designs into a unified FASTA."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Path to proteinmpnn_manifest.csv. Repeat for multiple runs.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out-fasta", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument(
        "--include-native",
        action="store_true",
        help="Also allow ProteinMPNN's first/native record to be selected.",
    )
    parser.add_argument("--line-width", type=int, default=80)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    missing_fastas: list[str] = []
    skipped_fastas: list[str] = []

    for manifest_raw in args.manifest:
        manifest_path = Path(manifest_raw).expanduser().resolve()
        for manifest_row in _load_manifest(manifest_path):
            fasta_path = Path(manifest_row["fasta_path"]).expanduser().resolve()
            backbone_pdb_path = Path(manifest_row["pdb_path"]).expanduser().resolve()
            if not fasta_path.is_file():
                missing_fastas.append(str(fasta_path))
                continue

            records = [
                record
                for record in _read_fasta(fasta_path)
                if _is_design_record(record, args.include_native)
                and VALID_AA_RE.match(record.sequence)
            ]
            if not records:
                skipped_fastas.append(str(fasta_path))
                continue

            records = sorted(records, key=lambda record: (_score(record), record.order))
            sampler = _sampler_from_path(backbone_pdb_path)
            length, sample = _length_and_sample_from_path(backbone_pdb_path)
            for rank, record in enumerate(records[: args.top_k], start=1):
                score = _score(record)
                design_id = _safe_id(
                    f"{sampler}_len{length}_sample{sample}_rank{rank}_score{score:.4f}"
                )
                rows.append(
                    {
                        "design_id": design_id,
                        "sampler": sampler,
                        "length": length,
                        "sample": sample,
                        "rank": str(rank),
                        "mpnn_score": f"{score:.6f}",
                        "mpnn_global_score": record.metadata.get("global_score", ""),
                        "mpnn_temperature": record.metadata.get("T", ""),
                        "mpnn_sample": record.metadata.get("sample", ""),
                        "sequence": record.sequence,
                        "sequence_length": str(len(record.sequence)),
                        "fasta_path": str(fasta_path),
                        "backbone_pdb_path": str(backbone_pdb_path),
                        "proteinmpnn_header": record.header,
                    }
                )

    if not rows:
        raise RuntimeError("No ProteinMPNN design sequences were collected.")

    _write_fasta(Path(args.out_fasta).expanduser().resolve(), rows, args.line_width)
    _write_csv(Path(args.out_csv).expanduser().resolve(), rows)

    print(f"Wrote {len(rows)} sequences to {args.out_fasta}")
    print(f"Wrote metadata CSV to {args.out_csv}")
    if missing_fastas:
        print(f"Missing FASTA files: {len(missing_fastas)}")
    if skipped_fastas:
        print(f"Skipped FASTA files without design records: {len(skipped_fastas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
