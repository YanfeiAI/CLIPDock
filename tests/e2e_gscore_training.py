"""Train and export G-score on a small installed-data subset."""

from __future__ import annotations

import argparse
from pathlib import Path

from e2e_utils import (
    assert_file,
    cleanup_docked_split,
    default_output_dir,
    load_cases,
    python_command,
    run_command,
    select_cases,
    temporary_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split", default="pdbpl_train")
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--val-count", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--pose-num", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    source = load_cases(args.source_split)
    train_ids = select_cases(source, args.train_count)
    val_ids = select_cases(source, args.val_count, offset=args.train_count)
    output_dir = (args.output_dir or default_output_dir("gscore_training")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_r{args.restarts}p{args.pose_num}"

    train_split = val_split = None
    try:
        with temporary_split(source, train_ids, "train") as train_split, \
                temporary_split(source, val_ids, "val") as val_split:
            command = python_command("model.train") + [
                "--output_dir", str(output_dir),
                "--train_split", train_split,
                "--val_split", val_split,
                "--data_suffix", suffix,
                "--restarts", str(args.restarts),
                "--pose_num", str(args.pose_num),
                "--epoch", str(args.epochs),
            ]
            if args.devices is not None:
                command += ["--devices", args.devices]
            run_command(command, args.timeout)

        model_path = output_dir / "model.pth"
        table_path = output_dir / "gscore.csv"
        assert_file(model_path)
        assert_file(table_path)
        header = table_path.read_text().splitlines()[0]
        if not header.startswith("lig,rec,g1"):
            raise AssertionError(f"Unexpected G-score table header: {header}")
        print(f"G-score training test passed. Artifacts: {output_dir}")
    finally:
        if not args.keep_cache:
            for split in (train_split, val_split):
                if split is not None:
                    cleanup_docked_split(split)


if __name__ == "__main__":
    main()
