"""Dock and evaluate about ten complexes from an installed benchmark dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from e2e_utils import (
    DOCKED_ROOT,
    cleanup_docked_split,
    csv_row_count,
    default_output_dir,
    load_cases,
    python_command,
    select_cases,
    temporary_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split", default="fragment_test")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--pose-num", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    source = load_cases(args.source_split)
    case_ids = select_cases(source, args.count)
    output_dir = (args.output_dir or default_output_dir("benchmark")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_r{args.restarts}p{args.pose_num}"

    split = None
    try:
        with temporary_split(source, case_ids, "benchmark") as split:
            metrics_path = output_dir / "metrics.csv"
            run_log = output_dir / "test_dock.log"
            command = python_command("script.test_dock") + [
                "--split", split,
                "--suffix", suffix,
                "--restarts", str(args.restarts),
                "--pose_num", str(args.pose_num),
                "--cpx_num", str(args.count),
                "--metrics_output", str(metrics_path),
            ]
            print("+", " ".join(command), flush=True)
            import subprocess

            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                timeout=args.timeout,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            run_log.write_text(completed.stdout)
            print(completed.stdout, end="")
            if completed.returncode != 0:
                raise RuntimeError(f"Benchmark command failed with exit code {completed.returncode}")

            with metrics_path.open(newline="") as handle:
                metrics = list(csv.DictReader(handle))
            if len(metrics) != 1 or not {
                "dataset", "method", "repeat", "success/tot_num"
            }.issubset(metrics[0]):
                raise AssertionError(f"Unexpected benchmark metrics CSV: {metrics_path}")

            details = DOCKED_ROOT / f"{split}{suffix}" / "rmsd_info.csv"
            if csv_row_count(details) != args.count:
                raise AssertionError(
                    f"Expected {args.count} evaluated complexes in {details}, "
                    f"found {csv_row_count(details)}"
                )
        print(f"Benchmark test passed for {args.count} complexes. Artifacts: {output_dir}")
    finally:
        if split is not None and not args.keep_cache:
            cleanup_docked_split(split)


if __name__ == "__main__":
    main()
