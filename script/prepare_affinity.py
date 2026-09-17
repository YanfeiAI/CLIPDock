#!/usr/bin/env python3
"""Download and prepare the 16-series affinity-ranking benchmark."""

from __future__ import annotations

import argparse
import csv
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
class Download:
    filename: str
    url: str
    checksum: str
    algorithm: str


AFFINITY_DOWNLOAD = Download(
    filename="public_binding_free_energy_benchmark-v1.0.zip",
    url=(
        "https://github.com/schrodinger/public_binding_free_energy_benchmark/"
        "archive/refs/tags/v1.0.zip"
    ),
    checksum="faffc54df6a3ce4598c8e58a96ab9b374faeb06baaa2aee9084b481a35832e61",
    algorithm="sha256",
)
AFFINITY_TARGETS = (
    ("jacs", "bace", "CAT-13a", "bace_out.csv"),
    ("jacs", "cdk2", "17", "cdk2_out.csv"),
    ("jacs", "jnk1_manual_flips", "17124-1", "jnk1_manual_flips_symbmcorr_out.csv"),
    ("jacs", "mcl1_extra_flips", "23", "mcl1_extra_flips_bmcorr_out.csv"),
    ("jacs", "p38", "2aa", "p38_out.csv"),
    ("jacs", "ptp1b", "20667", "ptp1b_out.csv"),
    ("jacs", "thrombin_core", "1a", "thrombin_core_out.csv"),
    ("jacs", "tyk2", "ejm_31", "tyk2_out.csv"),
    ("merck", "cdk8_5cei_new_helix_loop_extra", "13", "cdk8_5cei_new_helix_loop_extra_symbmcorr_no28_out.csv"),
    ("merck", "cmet", "CHEMBL3402741_400", "cmet_symcor_exp_out.csv"),
    ("merck", "eg5_extraprotomers", "CHEMBL1077204", "eg5_extraprotomers_manual_pkacorr_out.csv"),
    ("merck", "hif2a_automap", "1", "hif2a_automap_symbmcorr_out.csv"),
    ("merck", "pfkfb3_automap", "19", "pfkfb3_automap_symbmcorr_out.csv"),
    ("merck", "shp2", "10", "shp2_manual_213_exp_out.csv"),
    ("merck", "syk_4puz_fullmap", "CHEMBL3259820", "syk_pkacorr_out.csv"),
    ("merck", "tnks2_fullmap", "1a", "tnks2_fullmap_symcorr_pkacorr_out.csv"),
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


def safe_zip_path(name: str, prefix: PurePosixPath) -> Path | None:
    path = PurePosixPath(name)
    try:
        relative = path.relative_to(prefix)
    except ValueError:
        return None
    if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        return None
    return Path(*relative.parts)


def extract_affinity_source(archive: Path, destination: Path) -> None:
    prefix = PurePosixPath("public_binding_free_energy_benchmark-1.0")
    wanted = (
        "fep_benchmark_inputs/structure_inputs/jacs_set/",
        "fep_benchmark_inputs/structure_inputs/merck/",
        "21_4_results/ligand_predictions/jacs_set/",
        "21_4_results/ligand_predictions/merck/",
        "LICENSE",
    )
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            relative = safe_zip_path(info.filename, prefix)
            if relative is None or info.is_dir():
                continue
            posix = relative.as_posix()
            if not any(posix == item or posix.startswith(item) for item in wanted):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def prepare_affinity(archive: Path, data_root: Path) -> None:
    from rdkit import Chem

    destination = data_root / "affinity_benchmark"
    if destination.exists():
        targets = [path for path in destination.iterdir() if path.is_dir()]
        required = ("protein.pdb", "ref_ligand.sdf", "library.csv", "experimental.csv")
        if len(targets) != 16 or any(
            not (target / name).is_file() for target in targets for name in required
        ):
            raise RuntimeError(f"Existing affinity dataset is incomplete: {destination}")
        print(f"Affinity benchmark already installed: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".affinity-", dir=destination.parent) as tmp:
        temporary = Path(tmp)
        source = temporary / "source"
        extract_affinity_source(archive, source)
        staging = temporary / destination.name
        staging.mkdir()

        for dataset, stem, reference_name, experimental_name in AFFINITY_TARGETS:
            upstream_group = "jacs_set" if dataset == "jacs" else "merck"
            structure_dir = source / "fep_benchmark_inputs" / "structure_inputs" / upstream_group
            result_dir = source / "21_4_results" / "ligand_predictions" / upstream_group
            protein = structure_dir / f"{stem}_protein.pdb"
            ligand_sdf = structure_dir / f"{stem}_ligands.sdf"
            experimental = result_dir / experimental_name
            for required in (protein, ligand_sdf, experimental):
                if not required.is_file():
                    raise RuntimeError(f"Required upstream file is missing: {required}")

            target_name = f"{upstream_group}_{stem}"
            target = staging / target_name
            target.mkdir()
            shutil.copyfile(protein, target / "protein.pdb")

            molecules = {
                molecule.GetProp("_Name"): molecule
                for molecule in Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
                if molecule is not None and molecule.HasProp("_Name")
            }
            with experimental.open(newline="", encoding="utf-8-sig") as handle:
                experimental_rows = list(csv.DictReader(handle))
            names = [row["Ligand name"] for row in experimental_rows]
            missing = [name for name in names if name not in molecules]
            if missing:
                raise RuntimeError(f"{target_name}: missing ligand structures: {missing[:5]}")
            if reference_name not in molecules:
                raise RuntimeError(f"{target_name}: reference ligand {reference_name!r} is absent")

            with Chem.SDWriter(str(target / "ref_ligand.sdf")) as writer:
                writer.write(molecules[reference_name])
            with (target / "library.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("name", "smiles"))
                for name in names:
                    safe_name = name.replace("/", "_").replace(" ", "_")
                    writer.writerow((safe_name, Chem.MolToSmiles(Chem.RemoveHs(molecules[name]))))
            with (target / "experimental.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("name", "experimental_dg_kcal_mol"))
                for row in experimental_rows:
                    safe_name = row["Ligand name"].replace("/", "_").replace(" ", "_")
                    writer.writerow((safe_name, row["Exp. dG (kcal/mol)"]))

        license_path = source / "LICENSE"
        if license_path.is_file():
            shutil.copyfile(license_path, staging / "UPSTREAM_LICENSE")
        os.replace(staging, destination)
    print(f"Installed 16 affinity series under {destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--download-dir", type=Path, default=PROJECT_ROOT / "data" / "downloads")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepare_affinity(
        download(AFFINITY_DOWNLOAD, args.download_dir.resolve()), args.data_root.resolve()
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
