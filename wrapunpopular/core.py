"""
Core functionality for generating unpopular TESS light curves.

Functions
---------
_resolve_cutout_result: Normalize CutoutFactory output to a cached FITS path.
_get_tesscutout_aws: Fetch and cache cutouts from the TESS AWS S3 archive.
_get_tesscutout: Download cutouts via astroquery Tesscut as a fallback.
_plot_cpm_lightcurve: Render diagnostic plots for CPM light curves.
get_unpopular_lightcurve: Orchestrate cutout retrieval and CPM processing.
"""

import os
import shutil
from glob import glob
from datetime import datetime
from os.path import join
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.time import Time


def _emit(level: str, message: str) -> None:
    """Print a timestamped log line without using the logging module."""
    timestamp = datetime.now().strftime("%y%m%d-%H:%M:%S")
    print(f"{timestamp} {level} {message}")


try:
    from astroquery.mast import Tesscut

    astroquery_dependency = True
except ImportError:
    Tesscut = None
    astroquery_dependency = False

try:
    from astrocut import CutoutFactory

    astrocut_dependency = True
except ImportError:
    CutoutFactory = None
    astrocut_dependency = False

try:
    from astropy.coordinates import SkyCoord

    astropy_dependency = True
except ImportError:
    SkyCoord = None
    astropy_dependency = False

try:
    from tess_stars2px import tess_stars2px_function_entry

    tess_point_dependency = True
except ImportError:
    tess_stars2px_function_entry = None
    tess_point_dependency = False

try:
    import tess_cpm

    tess_cpm_dependency = True
except ImportError:
    try:
        import unpopular as tess_cpm

        tess_cpm_dependency = True
    except ImportError:

        tess_cpm = None
        tess_cpm_dependency = False

__all__ = ["get_unpopular_lightcurve"]


def _resolve_cutout_result(result: object, target_path: str, cache_dir: str) -> str:
    """
    Normalize the output of CutoutFactory.cube_cut to a concrete file path.
    """

    target_path = os.path.abspath(target_path)
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    if result is None:
        raise ValueError("No cutout data was returned.")

    if isinstance(result, str):
        source_path = os.path.abspath(result)
        if source_path == target_path:
            return target_path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        if os.path.exists(target_path):
            os.remove(target_path)
        shutil.move(source_path, target_path)
        return target_path

    if isinstance(result, dict):
        if "Local Path" in result:
            return _resolve_cutout_result(result["Local Path"], target_path, cache_dir)
        if "LocalPaths" in result:
            local_paths = result["LocalPaths"]
            if local_paths:
                return _resolve_cutout_result(local_paths[0], target_path, cache_dir)
        raise TypeError("Unsupported dictionary format for cube_cut output.")

    if isinstance(result, (list, tuple)):
        if not result:
            raise ValueError("Empty sequence returned by cube_cut.")
        return _resolve_cutout_result(result[0], target_path, cache_dir)

    if hasattr(result, "writeto"):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        result.writeto(target_path, overwrite=True)
        return target_path

    raise TypeError(f"Unsupported cube_cut return type: {type(result)!r}")


def _filter_cutout_paths_by_sector(
    cutout_paths: Iterable[str],
    sector_min: int,
    sector_max: int,
) -> list[str]:
    """Filter cutout paths by sector bounds, preserving unknown sector paths."""
    if sector_min == 0 and sector_max == 9999:
        return list(cutout_paths)

    filtered_paths = []
    for path in cutout_paths:
        try:
            sector_token = os.path.basename(path).split("_")[0]
            sector_idx = int(sector_token.split("-")[1])
        except (IndexError, ValueError):
            filtered_paths.append(path)
            continue
        if sector_max is None:
            if sector_idx >= sector_min or sector_idx == 1751:
                filtered_paths.append(path)
            continue
        if (sector_min <= sector_idx <= sector_max) or sector_idx == 1751:
            filtered_paths.append(path)
    return filtered_paths


def _current_sector_number() -> int | None:
    """Return the current TESS sector number based on midtimes.

    Returns:
        The latest sector whose midpoint is not in the future, or None if
        unavailable.
    """
    try:
        from tess_stars2px import TESS_Spacecraft_Pointing_Data as tspd
    except ImportError:
        return None

    sectors = np.array(tspd.sectors, dtype=int)
    times = Time(tspd.midtimes, format="jd")
    now = Time.now()

    in_past = times <= now
    if not np.any(in_past):
        return None

    return int(np.max(sectors[in_past]))


def _get_tesscutout_aws(
    tic_id: str,
    cache_dir: str = ".",
    size: int = 50,
    sector: Optional[int] = None,
    sector_min: int = 0,
    sector_max: int = 9999,
    force_download: bool = False,
    verbose: bool = True,
) -> Iterable[str]:
    """
    Download TESS cutouts from AWS S3 using astrocut.

    Parameters
    ----------
    tic_id : str
        The TIC identifier for the target.
    cache_dir : str
        Directory where cutouts should be cached.
    size : int or tuple
        Cutout size in pixels passed to astrocut.
    sector : int, optional
        Restrict downloads to a specific sector.
    sector_min : int
        Minimum sector to include when downloading cutouts.
    sector_max : int
        Maximum sector to include when downloading cutouts.
    force_download : bool
        Redownload even if a cached file is present.
    verbose : bool
        Emit informational output when True.
    """

    missing = []
    if not astrocut_dependency:
        missing.append("astrocut")
    if not astropy_dependency:
        missing.append("astropy")
    if not tess_point_dependency:
        missing.append("tess_stars2px")
    if not astroquery_dependency:
        missing.append("astroquery")

    if missing:
        raise ImportError(
            "The AWS cutout workflow requires the following optional dependencies: "
            + ", ".join(missing)
        )

    if not isinstance(tic_id, str):
        raise TypeError("tic_id must be provided as a string.")

    tic_id_clean = tic_id.strip()
    if not tic_id_clean:
        raise ValueError("tic_id must not be empty.")

    if tic_id_clean.upper().startswith("TIC"):
        tic_id_clean = tic_id_clean[3:].strip()

    tic_id_clean = tic_id_clean.replace(" ", "")

    if not tic_id_clean.isdigit():
        raise ValueError(f"tic_id must be numeric, received '{tic_id}'.")

    os.makedirs(cache_dir, exist_ok=True)

    from astroquery.mast import Catalogs

    catalog_result = Catalogs.query_object(f"TIC {tic_id_clean}", catalog="TIC")
    if len(catalog_result) == 0:
        raise ValueError(f"No TIC catalog entry found for '{tic_id}'.")

    ra = float(catalog_result[0]["ra"])
    dec = float(catalog_result[0]["dec"])
    target_crd = SkyCoord(ra=ra, dec=dec, unit="deg")

    (
        _,
        _,
        _,
        sector_arr,
        camera_arr,
        ccd_arr,
        *_,
    ) = tess_stars2px_function_entry(int(tic_id_clean), ra, dec)

    sector_arr = np.atleast_1d(sector_arr)
    camera_arr = np.atleast_1d(camera_arr)
    ccd_arr = np.atleast_1d(ccd_arr)

    current_sector = _current_sector_number()
    if sector_max == 9999:
        sector_max = current_sector

    pointings = []
    for s_val, cam_val, ccd_val in zip(sector_arr, camera_arr, ccd_arr):
        if not np.isfinite(s_val) or not np.isfinite(cam_val) or not np.isfinite(ccd_val):
            continue
        s_idx = int(s_val)
        cam_idx = int(cam_val)
        ccd_idx = int(ccd_val)
        if (s_idx < sector_min or s_idx > sector_max) and (int(s_idx) != 1751):
            continue
        if sector is not None and s_idx != sector:
            continue
        pointings.append((s_idx, cam_idx, ccd_idx))

    pointings = sorted(set(pointings))

    if not pointings:
        raise ValueError(f"No valid TESS sectors returned for TIC {tic_id_clean}.")

    if verbose:
        pointing_desc = ", ".join(
            f"sector={s_idx},cam={cam_idx},ccd={ccd_idx}" for s_idx, cam_idx, ccd_idx in pointings
        )
        _emit("INFO", f"TIC {tic_id_clean} AWS pointings: {pointing_desc}")

    factory = CutoutFactory()
    cutout_paths = []

    for s_idx, cam_idx, ccd_idx in pointings:
        s3_path = f"s3://stpubdata/tess/public/mast/tess-s{s_idx:04d}-{cam_idx}-{ccd_idx}-cube.fits"
        prefix = f"tess-s{s_idx:04d}-{cam_idx}-{ccd_idx}"
        base_name = f"{prefix}_TIC{tic_id_clean}_cutout.fits"
        local_path = join(cache_dir, base_name)

        if os.path.exists(local_path) and not force_download:
            if verbose:
                _emit("INFO", f"Using cached AWS cutout for TIC {tic_id_clean} at {local_path}")
            cutout_paths.append(local_path)
            continue

        if verbose:
            _emit(
                "INFO",
                f"Requesting AWS cutout for TIC {tic_id_clean} from {s3_path}"
                f"with target {target_crd} size={size}",
            )

        try:
            cutout_result = factory.cube_cut(
                s3_path,
                coordinates=target_crd,
                cutout_size=size,
            )
        except Exception as exc:
            if verbose:
                _emit(
                    "WRN",
                    (
                        f"Failed to retrieve AWS cutout for TIC {tic_id_clean} "
                        f"(sector={s_idx}, cam={cam_idx}, ccd={ccd_idx}): {exc}"
                    ),
                )
            continue

        try:
            resolved_path = _resolve_cutout_result(cutout_result, local_path, cache_dir)
        except Exception as exc:
            if verbose:
                _emit(
                    "WRN",
                    (
                        f"Failed to persist AWS cutout for TIC {tic_id_clean} "
                        f"(sector={s_idx}, cam={cam_idx}, ccd={ccd_idx}): {exc}"
                    ),
                )
            continue

        cutout_paths.append(resolved_path)

    if not cutout_paths:
        raise RuntimeError(f"Unable to obtain AWS TESS cutouts for TIC {tic_id_clean}.")

    return sorted(cutout_paths)


def _get_tesscutout(
    cache_dir: str = ".",
    objectname: Optional[str] = None,
    coordinates: Optional[object] = None,
    size: int = 5,
    sector: Optional[int] = None,
    sector_min: int = 0,
    sector_max: int = 9999,
    inflate: bool = True,
    force_download: bool = False,
    verbose: bool = True,
) -> Iterable[str]:
    """
    Helper function that wraps Tesscut.download_cutouts with a local cache.

    Parameters
    ----------
    cache_dir : str
        The path to which the TESScut FFI will be written.
    objectname : str, optional
        The target around which to search, by name (objectname="M104")
        or TIC ID (objectname="TIC 141914082").
        One and only one of coordinates and objectname must be supplied.
    coordinates : str or `astropy.coordinates` object, optional
        The target around which to search. It may be specified as a
        string or as the appropriate `astropy.coordinates` object.
    size : int, array-like, `~astropy.units.Quantity`
        Optional, default 5 pixels.
        The size of the cutout array. If ``size`` is a scalar number or
        a scalar `~astropy.units.Quantity`, then a square cutout of ``size``
        will be created.  If ``size`` has two elements, they should be in
        ``(ny, nx)`` order.  Scalar numbers in ``size`` are assumed to be in
        units of pixels. `~astropy.units.Quantity` objects must be in pixel or
        angular units.
    sector : int
        Optional.
        The TESS sector to return the cutout from.  If not supplied, cutouts
        from all available sectors on which the coordinate appears will be returned.
    sector_min : int
        Minimum sector to include when downloading cutouts.
    sector_max : int
        Maximum sector to include when downloading cutouts.
    inflate : bool
        Optional, default True.
        Cutout target pixel files are returned from the server in a zip file,
        by default they will be inflated and the zip will be removed.
        Set inflate to false to stop before the inflate step.

    Returns
    -------
    cutout_paths : list
        List of paths to tesscut FITS files.
    """

    if not astroquery_dependency:
        raise ImportError(
            "astroquery is required to download TESScut cutouts "
            "but could not be imported."
        )

    from astroquery.mast.utils import parse_input_location

    target_crd = None
    if objectname is not None:
        tic_id_clean = objectname.strip()
        if tic_id_clean.upper().startswith("TIC"):
            tic_id_clean = tic_id_clean[3:].strip()
        tic_id_clean = tic_id_clean.replace(" ", "")
        if tic_id_clean.isdigit():
            if not astropy_dependency:
                raise ImportError(
                    "astropy is required to resolve TIC coordinates for TESScut."
                )
            from astroquery.mast import Catalogs

            catalog_result = Catalogs.query_object(
                f"TIC {tic_id_clean}", catalog="TIC"
            )
            if len(catalog_result) == 0:
                raise ValueError(f"No TIC catalog entry found for '{objectname}'.")

            ra = float(catalog_result[0]["ra"])
            dec = float(catalog_result[0]["dec"])
            target_crd = SkyCoord(ra=ra, dec=dec, unit="deg")

    if target_crd is None:
        target_crd = parse_input_location(coordinates, objectname)

    current_sector = _current_sector_number()
    if sector_max == 9999:
        sector_max = current_sector

    ra = f"{target_crd.ra.value:.6f}"

    matched = [m for m in glob(join(cache_dir, "*.fits")) if ra in m]

    if matched and not force_download:
        matched = _filter_cutout_paths_by_sector(matched, sector_min, sector_max)
        if matched:
            if verbose:
                _emit(
                    "INFO",
                    f"Found cached FITS files in {cache_dir} with matching RA values: {matched}",
                )
                _emit(
                    "INFO",
                    "Set force_download=True if you want to re-download the cutouts.",
                )
            return matched

    t_paths = Tesscut.download_cutouts(
        coordinates=target_crd,
        size=size,
        sector=sector,
        path=cache_dir,
        inflate=inflate,
        objectname=None if target_crd is not None else objectname,
    )
    cutout_paths = list(t_paths["Local Path"])
    return _filter_cutout_paths_by_sector(cutout_paths, sector_min, sector_max)


def _plot_cpm_lightcurve(df: pd.DataFrame, figpath: str, min_cpm_reg: Optional[float] = None) -> None:

    plt.close("all")
    fig, axs = plt.subplots(nrows=2, figsize=(12, 8), sharex=True)

    axs[0].scatter(
        df.time,
        df.norm_flux,
        c="k",
        s=3,
        label="Normalized Flux",
        zorder=2,
        rasterized=True,
        linewidths=0,
    )
    axs[0].plot(
        df.time,
        df.cpm_pred,
        "-",
        lw=2,
        c="C3",
        alpha=0.8,
        label="CPM Prediction",
        zorder=1,
    )
    # Only plot detrended flux values within four standard deviations to suppress outliers.
    dtr_flux_std = float(np.nanstd(df.dtr_flux))
    if not np.isfinite(dtr_flux_std) or dtr_flux_std == 0:
        dtr_mask = df.dtr_flux.notna()
    else:
        dtr_mask = df.dtr_flux.notna() & (np.abs(df.dtr_flux) <= 4 * dtr_flux_std)
    axs[1].scatter(
        df.time[dtr_mask],
        df.dtr_flux[dtr_mask],
        c="k",
        s=3,
        zorder=2,
        rasterized=True,
        linewidths=0,
    )
    if min_cpm_reg is not None:
        txt = f"Reg = {min_cpm_reg:.2e}"
        axs[1].text(0.03, 0.97, txt, transform=axs[1].transAxes, ha="left", va="top")
    axs[0].legend()

    fig.text(-0.01, 0.5, "Relative flux", va="center", rotation=90)
    fig.text(0.5, -0.01, "Time - 2457000 [Days]", ha="center", va="center")
    fig.tight_layout()

    fig.savefig(figpath, bbox_inches="tight", dpi=300)


def get_unpopular_lightcurve(
    tic_id: str,
    ffi_dir: Optional[str] = None,
    verbose: bool = True,
    overwrite: bool = False,
    lc_dir: Optional[str] = None,
    sector_min: int = 0,
    sector_max: int = 9999,
) -> Optional[Iterable[str]]:
    """Download and create the default light curves using `unpopular`.

    Parameters
    ----------
    tic_id : str
        The TIC ID of the object as a string, e.g., "123456789".

    ffi_dir : str
        The directory to which TESS FFI cutout will be written.  If None,
        defaults to ~/.unpopular_cache.

    lc_dir : str
        The directory to which light curve and diagnostic plots will be
        written.  If None, defaults to `ffi_dir`.

    overwrite : bool
        If true, re-creates the CPM light curves.  Otherwise, pulls from cached
        CSV files.
    sector_min : int
        Minimum sector to include when downloading cutouts.
    sector_max : int
        Maximum sector to include when downloading cutouts.

    Returns
    -------
    csvpaths: list or None
        List of light-curve file paths.  None if none are found and downloaded.
    """

    if not astroquery_dependency:
        _emit("ERR", "The astroquery package is required for this function to work.")
        return None

    if not tess_cpm_dependency:
        _emit("ERR", "The tess_cpm package is required for this function to work.")
        return None

    if not isinstance(tic_id, str):
        raise TypeError("tic_id must be provided as a string.")

    if sector_min > sector_max:
        raise ValueError("sector_min must be less than or equal to sector_max.")

    if ffi_dir is None:
        ffi_dir = os.path.expanduser("~/.unpopular_cache")

    if lc_dir is None:
        lc_dir = ffi_dir

    objectname = f"TIC{tic_id}"
    try:
        cutout_paths = _get_tesscutout_aws(
            tic_id=tic_id,
            cache_dir=ffi_dir,
            size=50,
            sector_min=sector_min,
            sector_max=sector_max,
            verbose=verbose,
        )
    except Exception as exc:
        if verbose:
            _emit("INFO", f"Falling back to astroquery Tesscut download for TIC {tic_id}: {exc}")
        cutout_paths = _get_tesscutout(
            size=50,
            objectname=objectname,
            cache_dir=ffi_dir,
            sector_min=sector_min,
            sector_max=sector_max,
        )

    #
    # Create a default light curve from each sector of data available.
    #
    csvpaths = glob(join(lc_dir, f"TIC{tic_id}_*llc.csv"))

    if csvpaths and not overwrite:
        if verbose:
            _emit("INFO", f"Using cached CPM light curves for TIC {tic_id} in {lc_dir}")
        return csvpaths

    for cutout_path in cutout_paths:

        sector = os.path.basename(cutout_path).split("_")[0].split("-")[1]
        camera = os.path.basename(cutout_path).split("_")[0].split("-")[2]
        ccd = os.path.basename(cutout_path).split("_")[0].split("-")[3]
        starid = f"TIC{tic_id}_{sector}_{camera}_{ccd}"

        # Instantiate tess_cpm Source object, and remove values with non-zero
        # quality flags.
        s = tess_cpm.Source(cutout_path, remove_bad=True)

        # Plot the median image, trimmed at 10-90th percentile.
        figpath = join(lc_dir, f"{starid}_10_90.png")
        s.plot_cutout(figpath=figpath)

        # Select the aperture: whatever pixel the target star landed on.
        s.set_aperture(rowlims=[25, 25], collims=[25, 25])
        # s.set_aperture(rowlims=[23, 26], collims=[23, 26])

        figpath = join(lc_dir, f"{starid}_10_90_aperture.png")
        s.plot_cutout(rowlims=[20, 30], collims=[20, 30], show_aperture=True, figpath=figpath)

        # Plot the zero-centered & median-divided flux.
        figpath = join(lc_dir, f"{starid}_pixbypix_norm.png")
        s.plot_pix_by_pix(data_type="normalized_flux", figpath=figpath)

        s.add_cpm_model(exclusion_size=5, n=64, predictor_method="similar_brightness")

        # This method allows us to see our above choices
        _ = s.models[0][0].plot_model(size_predictors=10, figpath=figpath)

        # Use cross-validation to set the regularization value.  K-fold to
        # split the light curve into `k` contiguous sections, and to predict
        # the `i^th` section using all other sections.  Smaller values ->
        # weaker regularization.
        figpath = join(lc_dir, f"{starid}_regs.png")
        k = 100

        USE_XVALIDATION = 0
        if USE_XVALIDATION:
            cpm_regs = 10.0 ** np.arange(-10, 10)
            MIN_WORKED = False
            try:
                min_cpm_reg, _ = s.calc_min_cpm_reg(cpm_regs, k, figpath=figpath)
                MIN_WORKED = True
            except ValueError as e:
                if verbose:
                    _emit("WRN", f"Failed to compute min CPM regularization: {e}")
                MIN_WORKED = False
        else:
            MIN_WORKED = True
            min_cpm_reg = 0.1

        if MIN_WORKED:
            s.set_regs([min_cpm_reg])
            s.holdout_fit_predict(k=k)

            figpath = join(lc_dir, f"{starid}_pixbypix_cpm_subtracted.png")
            s.plot_pix_by_pix(data_type="cpm_subtracted_flux", split=True, figpath=figpath)

            aperture_normalized_flux = s.get_aperture_lc(
                data_type="normalized_flux"
            )
            aperture_cpm_prediction = s.get_aperture_lc(
                data_type="cpm_prediction", weighting="median"
            )
            detrended_flux = s.get_aperture_lc(data_type="cpm_subtracted_flux")

            #
            # Save the light curve as a CSV file
            #
            out_df = pd.DataFrame(
                {
                    "time": s.time,
                    "norm_flux": aperture_normalized_flux,
                    "cpm_pred": aperture_cpm_prediction,
                    "dtr_flux": 1 + detrended_flux,
                }
            )
            csvpath = join(lc_dir, f"{starid}_cpm_llc.csv")
            out_df.to_csv(csvpath, index=False)
            _emit("INFO", f"Wrote {csvpath}")

            figpath = join(lc_dir, f"{starid}_cpm_llc.png")
            _plot_cpm_lightcurve(out_df, figpath, min_cpm_reg=min_cpm_reg)

            plt.close("all")

    csvpaths = glob(join(lc_dir, f"TIC{tic_id}_*llc.csv"))
    return csvpaths
