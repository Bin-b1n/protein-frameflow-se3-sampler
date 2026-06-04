"""Run ProteinMPNN sequence design on FrameFlow sample.pdb files.

This is a thin wrapper around the external ProteinMPNN repository. It does not
ship ProteinMPNN weights or code; point --proteinmpnn-dir at a local checkout.

Example:
python scripts/run_proteinmpnn_on_samples.py ^
  --root inference_outputs/weights/pdb/published/unconditional/euler_20 ^
  --proteinmpnn-dir C:/tools/ProteinMPNN ^
  --out inference_outputs/weights/pdb/published/unconditional/euler_20/proteinmpnn ^
  --num-seq-per-target 8 ^
  --sampling-temp "0.1 0.2" ^
  --batch-size 1
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def _resolve_proteinmpnn_dir(raw_path: str | None) -> Path:
    if raw_path is None:
        raw_path = os.environ.get("PROTEINMPNN_DIR")
    if raw_path is None:
        raise ValueError(
            "ProteinMPNN path is required. Pass --proteinmpnn-dir or set "
            "PROTEINMPNN_DIR."
        )
    proteinmpnn_dir = Path(raw_path).expanduser().resolve()
    script_path = proteinmpnn_dir / "protein_mpnn_run.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing ProteinMPNN runner: {script_path}")
    return proteinmpnn_dir


def _discover_pdbs(root: Path, pdb_glob: str) -> list[Path]:
    pdbs = sorted(path for path in root.glob(pdb_glob) if path.is_file())
    if not pdbs:
        raise FileNotFoundError(f"No PDB files matching {pdb_glob!r} under {root}")
    return pdbs


def _relative_output_dir(root: Path, pdb_path: Path, out_root: Path) -> Path:
    rel_parent = pdb_path.parent.relative_to(root)
    return out_root / rel_parent


def _expected_fasta_path(out_dir: Path, pdb_path: Path) -> Path:
    return out_dir / "seqs" / f"{pdb_path.stem}.fa"


def _build_command(args, proteinmpnn_dir: Path, pdb_path: Path, out_dir: Path) -> list[str]:
    command = [
        args.python_executable,
        str(proteinmpnn_dir / "protein_mpnn_run.py"),
        "--pdb_path",
        str(pdb_path),
        "--out_folder",
        str(out_dir),
        "--num_seq_per_target",
        str(args.num_seq_per_target),
        "--sampling_temp",
        args.sampling_temp,
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
    ]

    if args.pdb_path_chains:
        command.extend(["--pdb_path_chains", args.pdb_path_chains])
    if args.ca_only:
        command.append("--ca_only")
    if args.use_soluble_model:
        command.append("--use_soluble_model")
    if args.model_name:
        command.extend(["--model_name", args.model_name])
    if args.path_to_model_weights:
        command.extend(["--path_to_model_weights", args.path_to_model_weights])

    command.extend(args.extra_arg)
    return command


def _write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdb_path",
        "output_dir",
        "fasta_path",
        "returncode",
        "command",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ProteinMPNN on FrameFlow-generated sample.pdb files."
    )
    parser.add_argument("--root", required=True, help="Root containing sample.pdb files.")
    parser.add_argument(
        "--pdb-glob",
        default="**/sample.pdb",
        help="Glob relative to --root. Defaults to all FrameFlow sample.pdb files.",
    )
    parser.add_argument(
        "--proteinmpnn-dir",
        default=None,
        help="Local ProteinMPNN checkout. Can also be set with PROTEINMPNN_DIR.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output root. Each input PDB gets an output directory mirroring its path.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable for ProteinMPNN. Defaults to this interpreter.",
    )
    parser.add_argument("--num-seq-per-target", type=int, default=8)
    parser.add_argument(
        "--sampling-temp",
        default="0.1 0.2",
        help='ProteinMPNN sampling temperature string, e.g. "0.1" or "0.1 0.2".',
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument(
        "--pdb-path-chains",
        default=None,
        help='Chains to design, e.g. "A". Omit for ProteinMPNN default behavior.',
    )
    parser.add_argument("--ca-only", action="store_true")
    parser.add_argument("--use-soluble-model", action="store_true")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--path-to-model-weights", default=None)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Append one raw ProteinMPNN CLI token. Repeat for multiple tokens.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write a manifest without running ProteinMPNN.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing remaining PDBs if one ProteinMPNN call fails.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    proteinmpnn_dir = _resolve_proteinmpnn_dir(args.proteinmpnn_dir)
    pdbs = _discover_pdbs(root, args.pdb_glob)

    rows = []
    for index, pdb_path in enumerate(pdbs, start=1):
        out_dir = _relative_output_dir(root, pdb_path, out_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        command = _build_command(args, proteinmpnn_dir, pdb_path, out_dir)
        fasta_path = _expected_fasta_path(out_dir, pdb_path)
        print(f"[{index}/{len(pdbs)}] {pdb_path}")
        print(" ".join(command))

        returncode = 0
        if not args.dry_run:
            completed = subprocess.run(command, cwd=str(proteinmpnn_dir), check=False)
            returncode = completed.returncode
            if returncode != 0 and not args.continue_on_error:
                rows.append(
                    {
                        "pdb_path": str(pdb_path),
                        "output_dir": str(out_dir),
                        "fasta_path": str(fasta_path),
                        "returncode": str(returncode),
                        "command": " ".join(command),
                    }
                )
                _write_manifest(out_root / "proteinmpnn_manifest.csv", rows)
                return returncode

        rows.append(
            {
                "pdb_path": str(pdb_path),
                "output_dir": str(out_dir),
                "fasta_path": str(fasta_path),
                "returncode": str(returncode),
                "command": " ".join(command),
            }
        )

    _write_manifest(out_root / "proteinmpnn_manifest.csv", rows)
    print(f"Wrote manifest for {len(rows)} PDBs to {out_root / 'proteinmpnn_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
