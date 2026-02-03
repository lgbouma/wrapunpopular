"""Integration tests for wrapunpopular.

These tests require optional dependencies and network access.
"""

import os

import pytest

from wrapunpopular import core, get_unpopular_lightcurve


def test_get_unpopular_lightcurve_sector_min() -> None:
    """Exercise sector_min to ensure it can run end-to-end."""
    if os.getenv("WRAPUNPOPULAR_INTEGRATION") != "1":
        pytest.skip("Set WRAPUNPOPULAR_INTEGRATION=1 to run integration tests.")

    if not core.astroquery_dependency:
        pytest.skip("astroquery is not installed.")

    if not core.tess_cpm_dependency:
        pytest.skip("tess_cpm (or unpopular) is not installed.")

    tic_id = "300651846"
    paths = get_unpopular_lightcurve(tic_id, sector_min=97)

    assert paths, "Expected at least one light curve path."
