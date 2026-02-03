# wrapunpopular

Lightweight Python utilities that wrap the `unpopular` pipeline to produce detrended TESS light curves with minimal setup.

## Features
- Downloads TESS FFI cutouts via MAST with local caching.
- Builds CPM light curves with the `tess_cpm` package.
- Generates diagnostic plots and CSV-ready light curves.

## Installation
Install the required dependencies in your environment:

```bash
pip install astroquery matplotlib numpy pandas
```

Clone this repository and make sure the project root is on your `PYTHONPATH`, or install it as an editable package:

```bash
pip install -e .
```

## Usage
Call the main helper from Python:

```python
from wrapunpopular import get_unpopular_lightcurve

paths = get_unpopular_lightcurve("460205581")
print(paths)
```

A convenience script is included for TIC 460205581:

```bash
python scripts/run_tic_460205581.py
```

Outputs (light curves and plots) will be written to the current working directory unless you pass a custom `lc_dir`. FFI cutouts are cached in `~/.unpopular_cache` by default or in a custom `ffi_dir` if provided.

## Installing `tess_cpm`
The `tess_cpm` package is distributed inside the upstream [`unpopular`](https://github.com/soichiro-hattori/unpopular) repository and is not published on PyPI. To install it:

```bash
git clone https://github.com/soichiro-hattori/unpopular.git
cd unpopular
pip install .
```

After this step the `wrapunpopular` helpers will be able to import `tess_cpm`.
