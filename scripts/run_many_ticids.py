#!/usr/bin/env python3
"""
Batch runner to generate light curves for multiple TIC IDs with wrapunpopular.
Edit CSVPATH before running so it points at a CSV containing a `ticid` column.
"""

from pathlib import Path
import traceback

import pandas as pd

from wrapunpopular import get_unpopular_lightcurve

# Update this path locally before running, e.g. Path("/path/to/ticids.csv")
CSVPATH = Path("REPLACE_WITH_YOUR_CSV_PATH")


def main() -> None:
    if CSVPATH == Path("REPLACE_WITH_YOUR_CSV_PATH"):
        raise ValueError("Edit CSVPATH in scripts/run_many_ticids.py before running.")

    if not CSVPATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSVPATH}")

    df = pd.read_csv(CSVPATH)

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
        print(f"Starting TIC ID {tic_id}")
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
            print(f"Failed TIC ID {tic_id}; see {error_path.name} for details.")
            continue

        print(f"Completed TIC ID {tic_id} 🎉")


if __name__ == "__main__":
    main()
