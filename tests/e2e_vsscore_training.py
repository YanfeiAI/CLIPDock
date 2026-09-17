"""Train VS-score on a small subset and run inference with its checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from e2e_utils import (
    DOCKED_ROOT,
    assert_file,
    cleanup_docked_split,
    csv_row_count,
    default_output_dir,
    load_cases,
    python_command,
    run_command,
    select_cases,
    temporary_split,
)


def ligand_path(case_dir: Path, case_id: str) -> Path:
    path = case_dir / f"{case_id}_ligand.sdf"
    if not path.is_file():
        raise FileNotFoundError(f"VS-score inference test requires an SDF ligand: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split", default="pdbpl_train")
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    source = load_cases(args.source_split)
    train_ids = select_cases(source, args.train_count)
    val_ids = select_cases(source, args.val_count, offset=args.train_count)
    output_dir = (args.output_dir or default_output_dir("vsscore_training")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_split = val_split = None
    try:
        with temporary_split(source, train_ids, "train") as train_split, \
                temporary_split(source, val_ids, "val") as val_split:
            run_command(python_command("model.train_vs") + [
                "--train_split", train_split,
                "--val_split", val_split,
                "--batch_size", str(args.batch_size),
                "--epochs", str(args.epochs),
                "--device", str(args.device),
                "--num_workers", str(args.num_workers),
                "--data_suffix", "e2e_r1p2",
                "--output_dir", str(output_dir),
            ], args.timeout)

            for split in (train_split, val_split):
                assert_file(DOCKED_ROOT / f"{split}_e2e_r1p2_vs.pkl")

            checkpoint = output_dir / "last.ckpt"
            assert_file(checkpoint)

            case_id = val_ids[0]
            pocket = DOCKED_ROOT / f"{val_split}_pocket" / case_id / f"{case_id}_pocket.pdb"
            assert_file(pocket)
            score_prefix = output_dir / "trained_checkpoint_inference"
            run_command(python_command("model.graph_score") + [
                "--pocket", str(pocket),
                "--ligands", str(ligand_path(source.directory / case_id, case_id)),
                "--model", str(checkpoint),
                "--output", str(score_prefix),
                "--device", str(args.device),
                "--batch_size", "1",
                "--num_workers", "0",
            ], args.timeout)
            score_csv = score_prefix.with_suffix(".csv")
            if csv_row_count(score_csv) != 1:
                raise AssertionError(f"Expected one VS-score inference row: {score_csv}")
        print(f"VS-score training and inference test passed. Artifacts: {output_dir}")
    finally:
        if not args.keep_cache:
            for split in (train_split, val_split):
                if split is not None:
                    cleanup_docked_split(split)


if __name__ == "__main__":
    main()
