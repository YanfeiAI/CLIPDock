"""Shared helpers for the executable end-to-end tests."""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
DOCKED_ROOT = DATA_ROOT / "docked"


@dataclass(frozen=True)
class CaseSource:
    directory: Path
    ids: list[str]


def load_cases(split: str) -> CaseSource:
    """Resolve an installed dataset and return cases with usable input files."""
    if split.startswith("pdbpl"):
        directory = RAW_ROOT / "pdbpl"
        list_path = RAW_ROOT / f"{split}.txt"
    else:
        directory = RAW_ROOT / split
        list_path = RAW_ROOT / f"{split}.txt"

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Missing dataset directory {directory}. Install the reproduction data first."
        )
    if list_path.is_file():
        candidates = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    else:
        candidates = sorted(path.name for path in directory.iterdir() if path.is_dir())

    ids = [case_id for case_id in candidates if case_inputs_exist(directory / case_id, case_id)]
    if not ids:
        raise RuntimeError(f"No complete complexes found for split {split!r} in {directory}")
    return CaseSource(directory=directory.resolve(), ids=ids)


def case_inputs_exist(case_dir: Path, case_id: str) -> bool:
    receptor = any((case_dir / f"{case_id}{suffix}").is_file() for suffix in (
        "_pocket.pdb", "_protein_pocket.pdb", "_protein.pdb",
    ))
    ligand = any((case_dir / f"{case_id}{suffix}").is_file() for suffix in (
        "_ligand.sdf", "_ligand.mol2", "_peptide.pdb",
    ))
    return receptor and ligand


def select_cases(source: CaseSource, count: int, offset: int = 0) -> list[str]:
    selected = source.ids[offset:offset + count]
    if len(selected) != count:
        raise RuntimeError(
            f"Requested {count} cases at offset {offset}, but {source.directory} "
            f"contains only {len(source.ids)} usable cases"
        )
    return selected


@contextmanager
def temporary_split(source: CaseSource, case_ids: list[str], role: str) -> Iterator[str]:
    """Expose selected cases as a uniquely named split without copying structures."""
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:10]
    split = f"e2e_{token}_{role}"
    destination = RAW_ROOT / split
    destination.mkdir()
    try:
        for case_id in case_ids:
            (destination / case_id).symlink_to(source.directory / case_id, target_is_directory=True)
        yield split
    finally:
        if destination.is_dir():
            for path in destination.iterdir():
                if path.is_symlink():
                    path.unlink()
                else:
                    raise RuntimeError(f"Refusing to remove unexpected test artifact: {path}")
            destination.rmdir()


def cleanup_docked_split(split: str) -> None:
    """Remove only caches carrying a generated e2e split identifier."""
    if not split.startswith("e2e_"):
        raise ValueError(f"Refusing to clean a non-e2e split: {split}")
    if not DOCKED_ROOT.is_dir():
        return
    for path in DOCKED_ROOT.glob(f"{split}*"):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()


def default_output_dir(workflow: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "tmp" / f"e2e_{workflow}_{stamp}"


def run_command(command: list[str], timeout: int) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")


def assert_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"Expected a non-empty file: {path}")


def csv_row_count(path: Path) -> int:
    assert_file(path)
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def python_command(module: str) -> list[str]:
    return [sys.executable, "-m", module]
