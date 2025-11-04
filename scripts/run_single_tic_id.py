#!/usr/bin/env python3
"""
Convenience script to generate the a light curve with wrapunpopular.
"""

from pathlib import Path

from wrapunpopular import get_unpopular_lightcurve


def main() -> None:

    TICID = '264461976'

    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results" / f"TIC_{TICID}"
    results_dir.mkdir(parents=True, exist_ok=True)

    get_unpopular_lightcurve(
        TICID,
        ffi_dir=str(results_dir),
        lc_dir=str(results_dir),
    )


if __name__ == "__main__":
    main()
