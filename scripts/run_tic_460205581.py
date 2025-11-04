#!/usr/bin/env python3
"""
Convenience script to generate the TIC 460205581 light curve with wrapunpopular.
"""

from pathlib import Path

from wrapunpopular import get_unpopular_lightcurve


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results" / "TIC_460205581"
    results_dir.mkdir(parents=True, exist_ok=True)

    get_unpopular_lightcurve(
        "460205581",
        ffi_dir=str(results_dir),
        lc_dir=str(results_dir),
    )


if __name__ == "__main__":
    main()
