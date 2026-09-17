#!/usr/bin/env python3
"""Download and prepare the TrueDecoy and RandomDecoy screening benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Download:
    filename: str
    url: str
    checksum: str
    algorithm: str


SCREENING_DOWNLOAD = Download(
    filename="VSDS_vd.rar",
    url="https://zenodo.org/api/records/14874127/files/VSDS_vd.rar/content",
    checksum="cc53b779cdc340cc6b8d5c82c76aceb5",
    algorithm="md5",
)
UNRAR_DOWNLOAD = Download(
    filename="rarlinux-x64-723.tar.gz",
    url="https://www.rarlab.com/rar/rarlinux-x64-723.tar.gz",
    checksum="759b4b6aa0d9f77131882162951193f3a0e54bf60e1d8dc4255aa308accab588",
    algorithm="sha256",
)


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download(spec: Download, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / spec.filename
    if destination.is_file() and digest(destination, spec.algorithm) == spec.checksum:
        print(f"Using verified download: {destination}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(spec.url, headers={"User-Agent": "CLIPDock/1.0.0"})
    print(f"Downloading {spec.url}")
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    actual = digest(partial, spec.algorithm)
    if actual != spec.checksum:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {spec.filename}: expected {spec.checksum}, got {actual}"
        )
    os.replace(partial, destination)
    print(f"Verified {spec.algorithm}: {actual}")
    return destination


def find_rar_extractor(download_dir: Path) -> str:
    for executable_name in ("unrar", "unrar-nonfree"):
        executable = shutil.which(executable_name)
        if executable:
            return executable
    if not sys.platform.startswith("linux") or os.uname().machine not in ("x86_64", "amd64"):
        raise RuntimeError("RAR5 extraction requires unrar on this platform")

    tool_dir = download_dir / "tools" / "unrar-7.23"
    executable = tool_dir / "unrar"
    if executable.is_file():
        return str(executable)
    archive = download(UNRAR_DOWNLOAD, download_dir)
    tool_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as packed:
        members = (("rar/unrar", "unrar"), ("rar/license.txt", "license.txt"))
        for member_name, output_name in members:
            source = packed.extractfile(packed.getmember(member_name))
            if source is None:
                raise RuntimeError(f"Missing {member_name} in {archive}")
            with (tool_dir / output_name).open("wb") as output:
                shutil.copyfileobj(source, output)
    executable.chmod(0o755)
    return str(executable)


def validate_screening(true_dir: Path, random_dir: Path) -> None:
    true_targets = sorted(path for path in true_dir.iterdir() if path.is_dir())
    random_targets = sorted(path for path in random_dir.iterdir() if path.is_dir())
    if len(true_targets) != 147 or len(random_targets) != 68:
        raise RuntimeError(
            f"Expected 147 TrueDecoy and 68 RandomDecoy targets; found "
            f"{len(true_targets)} and {len(random_targets)}"
        )
    for target in true_targets:
        required = (target / "active_decoys.smi", target / "crystal_ligand.sdf")
        if len(list(target.glob("*_optimal.pdb"))) != 1 or any(
            not path.is_file() for path in required
        ):
            raise RuntimeError(f"Incomplete TrueDecoy target: {target}")
    missing_refs = [path.name for path in random_targets if not (true_dir / path.name).is_dir()]
    if missing_refs:
        raise RuntimeError(f"RandomDecoy targets lack TrueDecoy receptors: {missing_refs[:5]}")
    for target in random_targets:
        if not (target / "active_decoys.smi").is_file():
            raise RuntimeError(f"Incomplete RandomDecoy target: {target}")


def prepare_screening(archive: Path, data_root: Path, download_dir: Path) -> None:
    true_destination = data_root / "TrueDecoy"
    random_destination = data_root / "RandomDecoy"
    if true_destination.is_dir() and random_destination.is_dir():
        validate_screening(true_destination, random_destination)
        print(f"Screening benchmarks already installed under {data_root}")
        return
    if true_destination.exists() or random_destination.exists():
        raise RuntimeError(f"Existing screening dataset is incomplete under {data_root}")

    data_root.mkdir(parents=True, exist_ok=True)
    extractor = find_rar_extractor(download_dir)
    with tempfile.TemporaryDirectory(prefix=".vsds-vd-", dir=data_root) as tmp:
        temporary = Path(tmp)
        command = [
            extractor,
            "x",
            "-idq",
            "-o+",
            str(archive),
            "VSDS_VD/DTEBV-D/*",
            "VSDS_VD/DRSM-D/*",
            str(temporary) + os.sep,
        ]
        subprocess.run(command, check=True)
        true_source = temporary / "VSDS_VD" / "DTEBV-D"
        random_source = temporary / "VSDS_VD" / "DRSM-D"
        validate_screening(true_source, random_source)
        os.replace(true_source, true_destination)
        os.replace(random_source, random_destination)
    print(f"Installed TrueDecoy (147) and RandomDecoy (68) under {data_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--download-dir", type=Path, default=PROJECT_ROOT / "data" / "downloads")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    download_dir = args.download_dir.resolve()
    prepare_screening(
        download(SCREENING_DOWNLOAD, download_dir), args.data_root.resolve(), download_dir
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError, tarfile.TarError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
