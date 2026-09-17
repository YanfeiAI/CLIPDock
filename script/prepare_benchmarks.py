#!/usr/bin/env python3
"""Download and adapt PoseBusters and PoseX for ``script.test_dock``.

Only Python's standard library is needed.  Downloads are cached under
``data/downloads`` and verified against pinned checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Archive:
    filename: str
    url: str
    checksum: str
    checksum_name: str


POSEBUSTERS_ARCHIVE = Archive(
    filename="posebusters_paper_data.zip",
    url=(
        "https://zenodo.org/api/records/8278563/files/"
        "posebusters_paper_data.zip/content"
    ),
    checksum="f004ac7c4e68317b5348497d2bb6bee6",
    checksum_name="md5",
)
POSEX_ARCHIVE = Archive(
    filename="posex_set.zip",
    url=(
        "https://huggingface.co/datasets/CataAI/PoseX/resolve/"
        "d4aed23b45ece3b4907b27003b984be1302bb291/posex_set.zip"
    ),
    checksum="66e2314ddcd9eea769a8fb000ffe21be5d6bbf799eb3ebe288ad657490eae186",
    checksum_name="sha256",
)
POSEBUSTERS_SPLIT = Archive(
    filename="posebusters_pdb_ccd_ids.txt",
    url=(
        "https://github.com/maabuu/posebusters/files/14516485/"
        "posebusters_pdb_ccd_ids.txt"
    ),
    checksum="a69a7b6b9a5a52531933078ef983e6c069e3a987a1d7a733bd7d72cbe1793de6",
    checksum_name="sha256",
)


DATASETS = {
    "posebusters": (POSEBUSTERS_ARCHIVE, "posebusters_benchmark_set", 428),
    "posex_sd": (POSEX_ARCHIVE, "posex_set/posex_self_docking_set", 718),
    "posex_cd": (POSEX_ARCHIVE, "posex_set/posex_cross_docking_set", 1312),
}


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download(archive: Archive, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    destination = download_dir / archive.filename
    if destination.is_file() and digest(destination, archive.checksum_name) == archive.checksum:
        print(f"Using verified download: {destination}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(archive.url, headers={"User-Agent": "CLIPDock"})
    print(f"Downloading {archive.url}")
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        written = 0
        next_report = 64 * 1024 * 1024
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
            written += len(chunk)
            if written >= next_report:
                progress = f"/{total / 1024**2:.0f} MiB" if total else ""
                print(f"  {written / 1024**2:.0f}{progress} MiB")
                next_report += 64 * 1024 * 1024

    actual = digest(partial, archive.checksum_name)
    if actual != archive.checksum:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {archive.filename}: expected {archive.checksum}, got {actual}"
        )
    os.replace(partial, destination)
    print(f"Verified {archive.checksum_name}: {actual}")
    return destination


def safe_relative_member(member: str, source_prefix: str) -> Path | None:
    path = PurePosixPath(member)
    prefix = PurePosixPath(source_prefix)
    try:
        relative = path.relative_to(prefix)
    except ValueError:
        return None
    if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        return None
    return Path(*relative.parts)


def validate_dataset(path: Path, expected_count: int) -> None:
    cases = sorted(item for item in path.iterdir() if item.is_dir())
    if len(cases) != expected_count:
        raise RuntimeError(f"Expected {expected_count} cases in {path}, found {len(cases)}")
    missing: list[str] = []
    for case in cases:
        case_id = case.name
        for suffix in ("_protein.pdb", "_ligand.sdf", "_ligand_start_conf.sdf"):
            if not (case / f"{case_id}{suffix}").is_file():
                missing.append(f"{case_id}/{case_id}{suffix}")
                break
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"Dataset is missing required CLIPDock inputs: {preview}")


def install_dataset(archive_path: Path, source_prefix: str, destination: Path,
                    expected_count: int) -> None:
    if destination.is_dir():
        validate_dataset(destination, expected_count)
        print(f"Dataset already installed: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Refusing to replace non-directory path: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temp:
        staging = Path(temp) / destination.name
        staging.mkdir()
        extracted = 0
        with zipfile.ZipFile(archive_path) as zipped:
            for info in zipped.infolist():
                if info.is_dir():
                    continue
                relative = safe_relative_member(info.filename, source_prefix)
                if relative is None:
                    continue
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted += 1
        if extracted == 0:
            raise RuntimeError(f"No files found below {source_prefix!r} in {archive_path}")
        validate_dataset(staging, expected_count)
        os.replace(staging, destination)
    print(f"Installed {destination.name}: {expected_count} cases in {destination}")


def install_annotations(dataset: str, data_root: Path, download_dir: Path) -> None:
    if dataset != "posebusters":
        return
    source = download(POSEBUSTERS_SPLIT, download_dir)
    case_ids = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    if len(case_ids) != 308 or len(set(case_ids)) != 308:
        raise RuntimeError(
            f"Expected 308 unique PoseBusters IDs in {source}, found {len(set(case_ids))}"
        )
    missing = [case_id for case_id in case_ids if not (data_root / dataset / case_id).is_dir()]
    if missing:
        raise RuntimeError(
            "Public PoseBusters split references cases absent from the structure archive: "
            + ", ".join(missing[:5])
        )
    destination = data_root / "posebusters_test.txt"
    shutil.copyfile(source, destination)
    print(f"Installed public 308-case PoseBusters split: {destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="+",
        choices=("all", *DATASETS),
        help="Benchmark datasets to prepare",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="CLIPDock raw-data directory",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "downloads",
        help="Directory used to cache source archives",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    requested = list(DATASETS) if "all" in args.datasets else list(dict.fromkeys(args.datasets))
    archives: dict[Archive, Path] = {}
    for name in requested:
        archive, source_prefix, expected_count = DATASETS[name]
        if archive not in archives:
            archives[archive] = download(archive, args.download_dir.resolve())
        install_dataset(
            archives[archive], source_prefix, args.data_root.resolve() / name, expected_count
        )
        install_annotations(name, args.data_root.resolve(), args.download_dir.resolve())


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
