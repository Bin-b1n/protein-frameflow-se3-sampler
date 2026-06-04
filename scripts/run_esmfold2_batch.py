"""Fold a FASTA file with local ESMFold2 and write mmCIF outputs.

This script follows the Biohub ESMFold2 Python API. Install the ESMFold2
package in your AutoDL environment before running it.

Example:
python scripts/run_esmfold2_batch.py \
  --fasta inference_outputs/weights/pdb/published/unconditional/designability_top3.fasta \
  --out-dir inference_outputs/weights/pdb/published/unconditional/esmfold2_top3 \
  --device cuda \
  --num-sampling-steps 32 \
  --limit 10
"""

import argparse
import csv
import re
import time
from collections.abc import Sequence
from pathlib import Path


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    seq_lines: list[str] = []

    def flush() -> None:
        nonlocal header, seq_lines
        if header is None:
            return
        records.append((header, "".join(seq_lines).replace(" ", "").upper()))
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


def _record_id(header: str) -> str:
    first_token = header.split()[0]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", first_token)
    return safe.strip("_") or "sequence"


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            if hasattr(value, "numel") and value.numel() > 1:
                return float(value.float().mean().item())
            return float(value.item())
        if isinstance(value, Sequence) and not isinstance(value, str):
            values = [float(item) for item in value]
            if values:
                return sum(values) / len(values)
        return float(value)
    except (TypeError, ValueError):
        return None


def _result_metric(result, name: str) -> float | None:
    if hasattr(result, name):
        return _to_float(getattr(result, name))
    if isinstance(result, dict):
        return _to_float(result.get(name))
    return None


def _load_esmfold2(model_name: str, device: str):
    try:
        from esm.models.esmfold2 import ESMFold2InputBuilder
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    except ImportError as exc:
        raise ImportError(
            "Could not import ESMFold2. Install the Biohub ESMFold2 package in "
            "this environment, then rerun this script."
        ) from exc

    model = ESMFold2Model.from_pretrained(model_name)
    if device.startswith("cuda") and hasattr(model, "cuda"):
        model = model.cuda()
    elif hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    input_builder = ESMFold2InputBuilder()
    return model, input_builder


def _fold_sequence(model, input_builder, sequence: str, args):
    try:
        from esm.models.esmfold2 import ProteinInput, StructurePredictionInput
    except ImportError as exc:
        raise ImportError("Could not import ESMFold2 input classes.") from exc

    structure_input = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence=sequence)]
    )
    return input_builder.fold(
        model,
        structure_input,
        num_loops=args.num_loops,
        num_sampling_steps=args.num_sampling_steps,
        num_diffusion_samples=args.num_diffusion_samples,
        seed=args.seed,
    )


def _write_structure(result, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(result, "complex") and hasattr(result.complex, "to_mmcif"):
        out_path.write_text(result.complex.to_mmcif())
        return
    if isinstance(result, dict):
        complex_obj = result.get("complex")
        if complex_obj is not None and hasattr(complex_obj, "to_mmcif"):
            out_path.write_text(complex_obj.to_mmcif())
            return
    raise TypeError("ESMFold2 result does not expose complex.to_mmcif().")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch fold FASTA records with ESMFold2.")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-name", default="biohub/ESMFold2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-loops", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=32)
    parser.add_argument("--num-diffusion-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip records with an existing output .cif file.",
    )
    args = parser.parse_args()

    fasta_path = Path(args.fasta).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    records = _read_fasta(fasta_path)
    records = records[args.start_index :]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("No FASTA records selected for folding.")

    model, input_builder = _load_esmfold2(args.model_name, args.device)
    rows: list[dict[str, str]] = []

    for index, (header, sequence) in enumerate(records, start=args.start_index + 1):
        design_id = _record_id(header)
        out_path = out_dir / f"{design_id}.cif"
        started = time.time()
        status = "ok"
        plddt = ptm = iptm = None

        if args.skip_existing and out_path.is_file():
            status = "skipped_existing"
        else:
            print(f"[{index}] Folding {design_id} length={len(sequence)}")
            result = _fold_sequence(model, input_builder, sequence, args)
            _write_structure(result, out_path)
            plddt = _result_metric(result, "plddt")
            ptm = _result_metric(result, "ptm")
            iptm = _result_metric(result, "iptm")

        rows.append(
            {
                "design_id": design_id,
                "sequence_length": str(len(sequence)),
                "output_cif": str(out_path),
                "plddt": "" if plddt is None else f"{plddt:.6f}",
                "ptm": "" if ptm is None else f"{ptm:.6f}",
                "iptm": "" if iptm is None else f"{iptm:.6f}",
                "status": status,
                "runtime_seconds": f"{time.time() - started:.3f}",
                "header": header,
            }
        )

    summary_path = out_dir / "esmfold2_results.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote ESMFold2 summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
