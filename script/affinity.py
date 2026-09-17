#!/usr/bin/env python3
"""Run and evaluate the paper's 16-series relative-affinity benchmark."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_values(path: Path, name_column: str, value_column: str) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {name_column, value_column} <= set(reader.fieldnames):
            raise RuntimeError(
                f"{path} must contain columns {name_column!r} and {value_column!r}"
            )
        rows = {row[name_column].lower(): float(row[value_column]) for row in reader}
    if len(rows) < 2:
        raise RuntimeError(f"At least two unique ligands are required in {path}")
    return rows


def select_targets(data_root: Path, requested: list[str] | None) -> list[Path]:
    targets = sorted(path for path in data_root.iterdir() if path.is_dir())
    if requested:
        index = {path.name: path for path in targets}
        missing = sorted(set(requested) - set(index))
        if missing:
            raise RuntimeError(f"Unknown affinity target(s): {', '.join(missing)}")
        targets = [index[name] for name in requested]
    if not targets:
        raise RuntimeError(f"No affinity targets found under {data_root}")
    return targets


def run_target(target: Path, output: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-m",
        "clipdock.main",
        "vs",
        "--input_file",
        str(target / "library.csv"),
        "--output_dir",
        str(output),
        "--receptor",
        str(target / "protein.pdb"),
        "--ref",
        str(target / "ref_ligand.sdf"),
        "--restarts",
        str(args.restarts),
        "--top_pose_num",
        "1",
    ]
    if args.workers is not None:
        command.extend(("--workers", str(args.workers)))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "affinity_benchmark"
    )
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "data" / "docked" / "affinity_benchmark"
    )
    parser.add_argument("--target", action="append", help="Run one prepared target; repeat as needed")
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--analyze-only", action="store_true", help="Evaluate existing results.csv files")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve() / f"r{args.restarts}"
    if not data_root.is_dir():
        raise RuntimeError(
            f"Prepared affinity data not found: {data_root}; run prepare_affinity.py first"
        )
    targets = select_targets(data_root, args.target)
    output_root.mkdir(parents=True, exist_ok=True)

    metrics: list[tuple[str, int, float]] = []
    for target in targets:
        for required in ("library.csv", "experimental.csv", "protein.pdb", "ref_ligand.sdf"):
            if not (target / required).is_file():
                raise RuntimeError(f"Missing prepared input: {target / required}")
        target_output = output_root / target.name
        if not args.analyze_only:
            run_target(target, target_output, args)
        result_path = target_output / "results.csv"
        if not result_path.is_file():
            raise RuntimeError(f"Missing CLIPDock result: {result_path}")

        experimental = read_values(target / "experimental.csv", "name", "experimental_dg_kcal_mol")
        predicted = read_values(result_path, "name", "CLIPDock score")
        failed = sorted(name for name, score in predicted.items() if score >= 9999.0)
        if failed:
            raise RuntimeError(f"{target.name}: docking failed for {failed[:5]}")
        missing = sorted(set(experimental) - set(predicted))
        if missing:
            raise RuntimeError(f"{target.name}: predictions missing for {missing[:5]}")
        names = sorted(experimental)
        correlation = float(
            pearsonr(
                [experimental[name] for name in names],
                [predicted[name] for name in names],
            ).statistic
        )
        metrics.append((target.name, len(names), correlation))

    per_target_path = output_root / "per_series_pearson.csv"
    with per_target_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("series", "n_ligands", "pearson_r"))
        for name, count, correlation in metrics:
            writer.writerow((name, count, f"{correlation:.6f}"))
    mean_r = float(np.mean([row[2] for row in metrics]))
    summary_path = output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("n_series", "mean_pearson_r"))
        writer.writerow((len(metrics), f"{mean_r:.6f}"))
    print(f"Evaluated {len(metrics)} series; unweighted mean Pearson r = {mean_r:.4f}")
    print(f"Per-series results: {per_target_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
