#!/usr/bin/env python3
"""
Batch runner to generate light curves for multiple TIC IDs with wrapunpopular.
Provide an absolute path to a CSV containing a `ticid` column when running.
"""

import argparse
from pathlib import Path
import traceback

import pandas as pd

from wrapunpopular import get_unpopular_lightcurve
from wrapunpopular.core import _emit

def parse_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Generate light curves for TIC IDs listed in a CSV file."
    )
    parser.add_argument(
        "csv_path",
        help="Absolute path to a CSV file containing a `ticid` column.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser()
    if not csv_path.is_absolute():
        raise ValueError("CSV path must be absolute.")
    return csv_path


def main(csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "ticid" not in df.columns:
        raise KeyError("CSV must contain a `ticid` column.")

    tic_ids = [
        str(ticid).strip()
        for ticid in df["ticid"].to_list()
        if str(ticid).strip()
    ]

    if not tic_ids:
        raise ValueError("No TIC IDs found in the `ticid` column.")

    root = Path(__file__).resolve().parent.parent

    for tic_id in tic_ids:
        _emit("INFO", f"Starting TIC ID {tic_id}")
        results_dir = root / "results" / f"TIC_{tic_id}"
        results_dir.mkdir(parents=True, exist_ok=True)

        try:
            get_unpopular_lightcurve(
                tic_id,
                ffi_dir=str(results_dir),
                lc_dir=str(results_dir),
            )
        except Exception as exc:  # pylint: disable=broad-except
            error_path = results_dir / "error.txt"
            error_message = (
                f"Failed to process TIC ID {tic_id}\n"
                f"Exception: {exc}\n\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            error_path.write_text(error_message, encoding="utf-8")
            _emit("WRN", f"Failed TIC ID {tic_id}; see {error_path.name} for details.")
            continue

        _emit("INFO", f"Completed TIC ID {tic_id} 🎉")


if __name__ == "__main__":
    main(parse_args())
