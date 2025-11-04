#!/usr/bin/env python3
"""
Chunk the 20251103_scocen_quicklook.csv target list into N smaller CSV files.

By default this script creates 16 chunks and writes them to
targetlists/20251103_scocen_quicklook_chunk{index}.csv while preserving the header.
"""

import argparse
import csv
from math import ceil
from pathlib import Path

from wrapunpopular.core import _emit

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a CSV file into N chunks while preserving the header."
    )
    parser.add_argument(
        "--input",
        default="targetlists/20251103_scocen_quicklook.csv",
        help="Path to the source CSV file (default: targetlists/20251103_scocen_quicklook.csv).",
    )
    parser.add_argument(
        "--chunks",
        "-n",
        type=int,
        default=16,
        help="Number of chunks to split the CSV into (default: 16).",
    )
    return parser.parse_args()


def chunk_csv(input_path: Path, n_chunks: int) -> None:
    if n_chunks <= 0:
        raise ValueError("Number of chunks must be a positive integer.")

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.reader(infile)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Input CSV {input_path} is empty.") from exc

        rows = list(reader)

    total_rows = len(rows)
    if total_rows == 0:
        raise ValueError(f"Input CSV {input_path} contains only the header row.")

    chunk_size = ceil(total_rows / n_chunks)
    output_dir = input_path.parent
    stem = input_path.stem

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk_rows = rows[start:end]

        if not chunk_rows:
            break

        output_path = output_dir / f"{stem}_chunk{chunk_idx + 1}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(chunk_rows)
        _emit("INFO", f"Wrote {len(chunk_rows)} rows to {output_path}")


def main() -> None:
    args = parse_args()
    chunk_csv(Path(args.input), args.chunks)


if __name__ == "__main__":
    main()
