#!/usr/bin/env python3
"""Run CLIPDock + VS-score and evaluate TrueDecoy and RandomDecoy enrichment."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit.ML.Scoring.Scoring import CalcEnrichment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRS = {"truedecoy": "TrueDecoy", "randomdecoy": "RandomDecoy"}


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def read_library(path: Path) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 2:
                raise RuntimeError(f"{path}:{line_number}: expected SMILES and compound name")
            smiles, name = fields[0], safe_name(fields[1])
            key = name.lower()
            if key in seen:
                raise RuntimeError(f"{path}:{line_number}: duplicate compound name {name!r}")
            seen.add(key)
            label = int(name.lower().startswith(("active", "bdb")))
            rows.append((name, smiles, label))
    if not rows or not any(row[2] for row in rows):
        raise RuntimeError(f"No labelled active compounds found in {path}")
    return rows


def write_library_csv(rows: list[tuple[str, str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("name", "smiles"))
        writer.writerows((name, smiles) for name, smiles, _ in rows)


def score_target(
    target: str,
    rows: list[tuple[str, str, int]],
    true_target: Path,
    output: Path,
    args: argparse.Namespace,
) -> None:
    proteins = list(true_target.glob("*_optimal.pdb"))
    if len(proteins) != 1:
        raise RuntimeError(f"Expected one *_optimal.pdb for {target}, found {len(proteins)}")
    reference = true_target / "crystal_ligand.sdf"
    if not reference.is_file():
        raise RuntimeError(f"Missing crystal ligand: {reference}")
    library_csv = output / "library.csv"
    write_library_csv(rows, library_csv)
    dock_command = [
        sys.executable, "-m", "clipdock.main", "vs",
        "--input_file", str(library_csv),
        "--output_dir", str(output),
        "--receptor", str(proteins[0]),
        "--ref", str(reference),
        "--restarts", str(args.restarts),
        "--top_pose_num", "1",
    ]
    if args.workers is not None:
        dock_command.extend(("--workers", str(args.workers)))
    if args.gscore is not None:
        dock_command.extend(("--gscore", str(args.gscore.resolve())))
    subprocess.run(dock_command, cwd=PROJECT_ROOT, check=True)

    pocket = output / f"{proteins[0].stem}_pocket.pdb"
    score_command = [
        sys.executable, "-m", "model.graph_score",
        "--pocket", str(pocket),
        "--lig_dir", str(output / f"pose_r{args.restarts}"),
        "--output", str(output / "vs_scores"),
        "--device", args.device,
        "--num_workers", str(args.score_workers),
    ]
    if args.model is not None:
        score_command.extend(("--model", str(args.model.resolve())))
    subprocess.run(score_command, cwd=PROJECT_ROOT, check=True)


def enrichment(labels: list[int], fraction: float) -> float:
    return float(CalcEnrichment([[label] for label in labels], 0, [fraction])[0])


def evaluate_target(
    score_path: Path, library_rows: list[tuple[str, str, int]]
) -> tuple[int, int, float, float, float]:
    labels = {name.lower(): label for name, _, label in library_rows}
    with score_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"name", "score"} <= set(reader.fieldnames):
            raise RuntimeError(f"{score_path} must contain name and score columns")
        scored: dict[str, float] = {}
        for row in reader:
            key = row["name"].lower()
            if key in labels:
                scored[key] = max(scored.get(key, float("-inf")), float(row["score"]))
    if not scored:
        raise RuntimeError(f"No benchmark compounds were scored in {score_path}")
    ranked_labels = [labels[name] for name, _ in sorted(scored.items(), key=lambda item: item[1], reverse=True)]
    if not any(ranked_labels):
        raise RuntimeError(f"No active compounds were successfully scored in {score_path}")
    return (
        len(ranked_labels),
        sum(ranked_labels),
        enrichment(ranked_labels, 0.005),
        enrichment(ranked_labels, 0.01),
        enrichment(ranked_labels, 0.05),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("datasets", nargs="+", choices=("all", *DATASET_DIRS))
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "data" / "docked" / "virtual_screening",
    )
    parser.add_argument("--target", action="append", help="Run one target; repeat as needed")
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--workers", type=int, default=None, help="Parallel docking workers")
    parser.add_argument("--device", default="0", help="CUDA device index, or -1 for CPU")
    parser.add_argument("--score-workers", type=int, default=12)
    parser.add_argument("--model", type=Path, default=None, help="VS-score checkpoint")
    parser.add_argument("--gscore", type=Path, default=None,
                        help="Path to the exported G-score parameter CSV")
    parser.add_argument("--analyze-only", action="store_true", help="Evaluate existing vs_scores.csv files")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.model is not None and not args.model.is_file():
        raise RuntimeError(f"VS-score checkpoint does not exist: {args.model}")
    if args.gscore is not None and not args.gscore.is_file():
        raise RuntimeError(f"G-score parameter file does not exist: {args.gscore}")
    requested = list(DATASET_DIRS) if "all" in args.datasets else list(dict.fromkeys(args.datasets))
    data_root = args.data_root.resolve()
    true_root = data_root / "TrueDecoy"
    if not true_root.is_dir():
        raise RuntimeError(
            f"Prepared screening data not found under {data_root}; run prepare_vs.py first"
        )
    output_root = args.output_root.resolve() / f"r{args.restarts}"
    output_root.mkdir(parents=True, exist_ok=True)
    per_target: list[tuple[str, str, int, int, float, float, float]] = []

    for dataset in requested:
        dataset_root = data_root / DATASET_DIRS[dataset]
        if not dataset_root.is_dir():
            raise RuntimeError(f"Missing prepared dataset: {dataset_root}")
        targets = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
        if args.target:
            missing = sorted(set(args.target) - set(targets))
            if missing:
                raise RuntimeError(f"Unknown {dataset} target(s): {', '.join(missing)}")
            targets = [target for target in args.target]
        if not targets:
            raise RuntimeError(f"No targets found in {dataset_root}")

        for target in targets:
            library = read_library(dataset_root / target / "active_decoys.smi")
            output = output_root / dataset / target
            if not args.analyze_only:
                score_target(target, library, true_root / target, output, args)
            score_path = output / "vs_scores.csv"
            if not score_path.is_file():
                raise RuntimeError(f"Missing VS-score result: {score_path}")
            n_scored, n_active, ef05, ef1, ef5 = evaluate_target(score_path, library)
            per_target.append((dataset, target, n_scored, n_active, ef05, ef1, ef5))

    per_target_path = output_root / "per_target_metrics.csv"
    with per_target_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "target", "n_scored", "n_active", "EF_0.5%", "EF_1%", "EF_5%"))
        for row in per_target:
            writer.writerow((*row[:4], *(f"{value:.6f}" for value in row[4:])))

    summary_path = output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "n_targets", "mean_EF_0.5%", "mean_EF_1%", "mean_EF_5%"))
        for dataset in requested:
            rows = [row for row in per_target if row[0] == dataset]
            means = np.mean([[row[4], row[5], row[6]] for row in rows], axis=0)
            writer.writerow((dataset, len(rows), *(f"{value:.6f}" for value in means)))
            print(
                f"{dataset}: {len(rows)} targets; mean EF0.5%={means[0]:.3f}, "
                f"EF1%={means[1]:.3f}, EF5%={means[2]:.3f}"
            )
    print(f"Per-target results: {per_target_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
