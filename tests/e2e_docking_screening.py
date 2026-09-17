"""Exercise the documented docking, screening, and VS-score inference workflows."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from clipdock.utils.mol import smi_gen_3d


DATA_DIR = PROJECT_ROOT / "data"
PROTEIN = DATA_DIR / "9iwy_protein.pdb"
REFERENCE_LIGAND = DATA_DIR / "9iwy_ligand.sdf"
SCREENING_LIBRARY = DATA_DIR / "raw" / "screening_examples.csv"
GSCORE_PARAMS = PROJECT_ROOT / "clipdock" / "gscore.csv"
VS_MODEL_CHECKPOINT = PROJECT_ROOT / "model" / "vs_model.ckpt"


def run_command(command: list[str], timeout: int) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")


def assert_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"Expected a non-empty file: {path}")


def assert_docked_sdf(path: Path) -> None:
    assert_file(path)
    molecules = [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol is not None]
    if not molecules:
        raise AssertionError(f"No readable molecules in {path}")
    if not molecules[0].HasProp("CLIPDock score"):
        raise AssertionError(f"Missing CLIPDock score property in {path}")


def sample_screening_library(input_path: Path, output_path: Path, seed: int,
                             count: int = 10) -> list[list[str]]:
    with input_path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < count + 1:
        raise AssertionError(f"{input_path} contains fewer than {count} data rows")
    header, data = rows[0], [row for row in rows[1:] if len(row) >= 2]
    rng = random.Random(seed)
    rng.shuffle(data)
    selected: list[list[str]] = []
    for row in data:
        try:
            screenable = smi_gen_3d(row[1])
        except Exception:
            screenable = None
        if screenable is not None:
            selected.append(row)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise AssertionError(
            f"Only {len(selected)} screening-library rows passed 3D screenability preflight; "
            f"need {count}"
        )
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(selected)
    return selected


def run_core_workflows(output_dir: Path, protein: Path, reference_ligand: Path,
                       restarts: int, workers: int, pose_num: int, timeout: int) -> None:
    ligand_output = output_dir / "ligand_docking.sdf"
    run_command([
        sys.executable, "-m", "clipdock.main",
        "-r", str(protein),
        "-l", str(reference_ligand),
        "-o", str(ligand_output),
        "--visualize",
        "--restarts", str(restarts),
        "--pose_num", str(pose_num),
        "--workers", str(workers),
        "--gscore", str(GSCORE_PARAMS),
    ], timeout)
    assert_docked_sdf(ligand_output)

    smiles_output = output_dir / "smiles_docking.sdf"
    smiles = "c1c(S(=O)(=O)C(C)C)ccc2nccc(Nc3cc4nc(sc4cc3)C3CCCC3)c12"
    run_command([
        sys.executable, "-m", "clipdock.main",
        "-r", str(protein),
        "-s", smiles,
        "--ref", str(reference_ligand),
        "-o", str(smiles_output),
        "--visualize",
        "--restarts", str(restarts),
        "--pose_num", str(pose_num),
        "--workers", str(workers),
        "--gscore", str(GSCORE_PARAMS),
    ], timeout)
    assert_docked_sdf(smiles_output)


def run_screening_example_vs_and_rescore(output_dir: Path, protein: Path,
                                         reference_ligand: Path, restarts: int,
                                         workers: int, seed: int, timeout: int) -> None:
    sampled_table = output_dir / "screening_examples_random10.csv"
    selected = sample_screening_library(SCREENING_LIBRARY, sampled_table, seed)
    clipdock_dir = output_dir / "01_clipdock"
    run_command([
        sys.executable, "-m", "clipdock.main", "vs",
        "-i", str(sampled_table),
        "-o", str(clipdock_dir),
        "-r", str(protein),
        "--ref", str(reference_ligand),
        "--restarts", str(restarts),
        "--top_pose_num", "1",
        "--workers", str(workers),
        "--gscore", str(GSCORE_PARAMS),
    ], timeout)
    assert_file(clipdock_dir / "results.csv")
    assert_file(clipdock_dir / "results_sorted.csv")

    pocket = clipdock_dir / f"{protein.stem}_pocket.pdb"
    if not pocket.is_file():
        pocket = protein
    pose_dir = clipdock_dir / f"pose_r{restarts}"
    if not pose_dir.is_dir():
        raise AssertionError(f"Missing VS pose directory: {pose_dir}")
    pose_count = len(list(pose_dir.glob("*.sdf")))
    if pose_count != len(selected):
        raise AssertionError(f"Expected {len(selected)} docked poses, found {pose_count}")

    vs_score_dir = output_dir / "02_vs_score"
    rescore_prefix = vs_score_dir / "vs_scores"
    top_dir = vs_score_dir / "top100"
    run_command([
        sys.executable, "-m", "clipdock.main", "vs-score",
        "-p", str(pocket),
        "--lig_dir", str(pose_dir),
        "-o", str(rescore_prefix),
        "--top_num", "100",
        "--top_dir", str(top_dir),
        "--model", str(VS_MODEL_CHECKPOINT),
        "--device", "-1",
        "--batch_size", str(max(1, pose_count)),
        "--num_workers", "0",
    ], timeout)

    rescore_csv = rescore_prefix.with_suffix(".csv")
    assert_file(rescore_csv)
    assert_file(vs_score_dir / "vs_scores_top100.csv")
    if not top_dir.is_dir():
        raise AssertionError(f"Missing VS-score top directory: {top_dir}")
    top_pose_count = len(list(top_dir.glob("*.sdf")))
    if top_pose_count != min(100, pose_count):
        raise AssertionError(
            f"Expected {min(100, pose_count)} VS-score top poses, found {top_pose_count}"
        )
    with rescore_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != pose_count or any("score" not in row for row in rows):
        raise AssertionError(f"Unexpected VS rescore output: {rescore_csv}")


def benchmark_available(split: str) -> bool:
    if split == "posebusters":
        return (DATA_DIR / "raw" / "posebusters_test.txt").is_file() and \
            (DATA_DIR / "raw" / "posebusters").is_dir()
    return (DATA_DIR / "raw" / split).is_dir()


def run_benchmarks(strict: bool, timeout: int) -> None:
    for split in ("posebusters", "posex_sd", "posex_cd"):
        if not benchmark_available(split):
            message = f"Skipping {split}: local benchmark dataset is not present"
            if strict:
                raise RuntimeError(message)
            print(message)
            continue
        run_command([
            sys.executable, "-m", "script.test_dock",
            "--split", split,
            "--suffix", "_readme_test",
            "--bust",
        ], timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--full", action="store_true",
                        help="Use README-scale 32 restarts and 9 saved poses")
    parser.add_argument("--include-benchmarks", action="store_true")
    parser.add_argument("--strict-benchmarks", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "tmp" / f"e2e_docking_screening_{stamp}"
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not PROTEIN.is_file() or not REFERENCE_LIGAND.is_file():
        raise FileNotFoundError("The README example receptor or ligand is missing")
    if not SCREENING_LIBRARY.is_file():
        raise FileNotFoundError(
            f"Missing screening example library: {SCREENING_LIBRARY}. Run "
            "python script/prepare_screening_examples.py first."
        )
    if not VS_MODEL_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Missing VS-score checkpoint: {VS_MODEL_CHECKPOINT}. Download it from GitHub Releases."
        )
    restarts = 32 if args.full else args.restarts
    workers = 8 if args.full else args.workers
    pose_num = 9 if args.full else 1
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    test_protein = input_dir / PROTEIN.name
    test_ligand = input_dir / REFERENCE_LIGAND.name
    shutil.copy2(PROTEIN, test_protein)
    shutil.copy2(REFERENCE_LIGAND, test_ligand)

    run_core_workflows(output_dir, test_protein, test_ligand,
                       restarts, workers, pose_num, args.timeout)
    run_screening_example_vs_and_rescore(output_dir, test_protein, test_ligand,
                                         restarts, workers, args.seed, args.timeout)
    if args.include_benchmarks:
        run_benchmarks(args.strict_benchmarks, args.timeout)
    print(f"Integration tests passed. Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
