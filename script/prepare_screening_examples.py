#!/usr/bin/env python3
"""Download and prepare the screening example library."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRUCTURES_URL = (
    "https://unmtid-dbs.net/download/DrugCentral/2021_09_01/"
    "structures.smiles.tsv"
)
DEFAULT_SOURCE_OUTPUT = PROJECT_ROOT / "data" / "raw" / "structures.smiles.tsv"
DEFAULT_LIBRARY_OUTPUT = PROJECT_ROOT / "data" / "raw" / "screening_examples.csv"
REQUIRED_COLUMNS = {"smiles", "id", "inn"}


def validate_source(path: Path) -> int:
    """Validate the source TSV and return the number of data rows."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise RuntimeError(f"Downloaded file is empty: {path}")

        columns = {column.strip().lower() for column in header}
        missing = REQUIRED_COLUMNS - columns
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise RuntimeError(f"Downloaded file is missing columns: {missing_text}")

        rows = sum(1 for row in reader if row and any(cell.strip() for cell in row))
    if rows == 0:
        raise RuntimeError(f"Downloaded file has no data rows: {path}")
    return rows


def download_source(url: str, destination: Path) -> int:
    """Download *url* atomically, validate it, and return its row count."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/tab-separated-values,text/plain;q=0.9,*/*;q=0.1",
            "User-Agent": "CLIPDock/1.0.0",
        },
    )
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)

        rows = validate_source(temporary)
        os.replace(temporary, destination)
        temporary = None
        return rows
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_library(source: Path, destination: Path) -> int:
    """Extract names and SMILES into a two-column CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            writer = csv.writer(output)
            writer.writerow(("name", "smiles"))
            rows = 0
            with source.open(newline="", encoding="utf-8-sig") as input_file:
                reader = csv.DictReader(input_file, delimiter="\t")
                fields = {field.strip().lower(): field for field in (reader.fieldnames or [])}
                smiles_field = fields.get("smiles")
                name_field = fields.get("inn")
                if smiles_field is None or name_field is None:
                    raise RuntimeError(f"Source lacks SMILES and INN columns: {source}")
                for row in reader:
                    name = (row.get(name_field) or "").strip()
                    smiles = (row.get(smiles_field) or "").strip()
                    if not name or not smiles:
                        continue
                    writer.writerow((name, smiles))
                    rows += 1

        if rows == 0:
            raise RuntimeError(f"No usable name/SMILES rows found in {source}")
        os.replace(temporary, destination)
        temporary = None
        return rows
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default=STRUCTURES_URL,
                        help="Source structures.smiles.tsv URL")
    parser.add_argument("--source-output", type=Path, default=DEFAULT_SOURCE_OUTPUT,
                        help="Destination for the source TSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_LIBRARY_OUTPUT,
                        help="Destination for the prepared screening library")
    parser.add_argument("--force", action="store_true",
                        help="Redownload the source TSV even when it exists")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source = args.source_output.resolve()
    destination = args.output.resolve()

    if source.is_file() and not args.force:
        source_rows = validate_source(source)
        print(f"Using existing source: {source} ({source_rows} rows)")
    else:
        print(f"Downloading {args.url}")
        source_rows = download_source(args.url, source)
        print(f"Downloaded {source_rows} rows to {source}")

    library_rows = prepare_library(source, destination)
    print(f"Prepared {library_rows} name/SMILES rows to {destination}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
