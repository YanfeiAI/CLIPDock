#!/usr/bin/env python3
"""Verify and install the CLIPDock training, validation, and test datasets."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Dataset:
    name: str
    archive: Path
    metadata: Path
    expected_count: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(release_dir: Path) -> dict[Path, str]:
    manifest_path = release_dir / "SHA256SUMS"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing checksum manifest: {manifest_path}")
    checksums: dict[Path, str] = {}
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise RuntimeError(f"Invalid SHA256SUMS line {line_number}: {line!r}")
        relative = PurePosixPath(parts[1].lstrip("*"))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe path in SHA256SUMS: {relative}")
        checksums[Path(*relative.parts)] = parts[0].lower()
    return checksums


def verify_files(release_dir: Path, relative_paths: list[Path]) -> None:
    checksums = read_manifest(release_dir)
    for relative in relative_paths:
        source = release_dir / relative
        if not source.is_file():
            raise FileNotFoundError(f"Missing release file: {source}")
        expected = checksums.get(relative)
        if expected is None:
            raise RuntimeError(f"No SHA-256 entry for {relative} in {release_dir / 'SHA256SUMS'}")
        actual = sha256(source)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {source}: expected {expected}, got {actual}")
        print(f"Verified {relative}: {actual}")


def safe_member_path(member_name: str, archive_root: str) -> Path | None:
    member = PurePosixPath(member_name)
    try:
        relative = member.relative_to(PurePosixPath(archive_root))
    except ValueError:
        return None
    if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        return None
    return Path(*relative.parts)


def extract_dataset(archive: Path, archive_root: str, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"Refusing to replace existing dataset: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as tmp:
        staging = Path(tmp) / destination.name
        staging.mkdir()
        extracted = 0
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                relative = safe_member_path(member.name, archive_root)
                if relative is None:
                    continue
                if member.isdir():
                    (staging / relative).mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise RuntimeError(f"Unsupported archive member: {member.name}")
                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Cannot read archive member: {member.name}")
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted += 1
        if extracted == 0:
            raise RuntimeError(f"No files below {archive_root!r} in {archive}")
        os.replace(staging, destination)


def validate_dataset(dataset: Dataset, data_root: Path) -> None:
    destination = data_root / dataset.name
    cases = sorted(path for path in destination.iterdir() if path.is_dir())
    if len(cases) != dataset.expected_count:
        raise RuntimeError(
            f"Expected {dataset.expected_count} cases in {destination}, found {len(cases)}"
        )
    missing: list[str] = []
    for case in cases:
        case_id = case.name
        receptor = any((case / f"{case_id}{suffix}").is_file() for suffix in (
            "_pocket.pdb", "_protein_pocket.pdb", "_protein.pdb",
        ))
        ligand = any((case / f"{case_id}{suffix}").is_file() for suffix in (
            "_ligand.sdf", "_ligand.mol2", "_peptide.pdb",
        ))
        if not receptor or not ligand:
            missing.append(case_id)
    if missing:
        raise RuntimeError("Dataset is missing required inputs: " + ", ".join(missing[:5]))
    print(f"Installed {dataset.name}: {len(cases)} cases")


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
        temporary = Path(output.name)
        with source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output)
    os.replace(temporary, destination)


def install_pdbpl(release_dir: Path, data_root: Path) -> None:
    files = [
        Path("pdbpl.tar.gz"),
        Path("metadata.csv"),
        Path("pdbpl_train.txt"),
        Path("pdbpl_val.txt"),
    ]
    verify_files(release_dir, files)
    dataset = Dataset("pdbpl", files[0], files[1], 42058)
    destination = data_root / dataset.name
    if not destination.exists():
        extract_dataset(release_dir / dataset.archive, dataset.name, destination)
    copy_file(release_dir / dataset.metadata, data_root / "pdbpl_info.csv")
    copy_file(release_dir / files[2], data_root / files[2].name)
    copy_file(release_dir / files[3], data_root / files[3].name)
    validate_dataset(dataset, data_root)


def install_test_datasets(release_dir: Path, data_root: Path) -> None:
    datasets = [
        Dataset("fragment_test", Path("fragment_test/structures.tar.gz"),
                Path("fragment_test/metadata.csv"), 331),
        Dataset("nucleic_acid_associated_test",
                Path("nucleic_acid_associated_test/structures.tar.gz"),
                Path("nucleic_acid_associated_test/metadata.csv"), 146),
        Dataset("fabp4_test", Path("fabp4_test/structures.tar.gz"),
                Path("fabp4_test/metadata.csv"), 179),
    ]
    files = [path for dataset in datasets for path in (dataset.archive, dataset.metadata)]
    verify_files(release_dir, files)
    for dataset in datasets:
        destination = data_root / dataset.name
        if not destination.exists():
            extract_dataset(release_dir / dataset.archive, dataset.name, destination)
        copy_file(release_dir / dataset.metadata, destination / "metadata.csv")
        validate_dataset(dataset, data_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pdbpl-dir", type=Path,
                        help="Downloaded PDBPL Zenodo release directory")
    parser.add_argument("--test-data-dir", type=Path,
                        help="Downloaded CLIPDock test-datasets Zenodo release directory")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "raw",
                        help="Destination raw-data directory")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pdbpl_dir is None and args.test_data_dir is None:
        parser.error("provide --pdbpl-dir, --test-data-dir, or both")
    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    if args.pdbpl_dir is not None:
        install_pdbpl(args.pdbpl_dir.resolve(), data_root)
    if args.test_data_dir is not None:
        install_test_datasets(args.test_data_dir.resolve(), data_root)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, tarfile.TarError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
