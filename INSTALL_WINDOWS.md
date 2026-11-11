# Installation on Windows

## Recommended: install directly from GitHub

With Python 3.10 selected through the Windows Python launcher:

```powershell
py -3.10 -m pip install --upgrade pip
py -3.10 -m pip install "git+https://github.com/Daniel-Uhr/pydrlpdid.git"
```

The package supports Python 3.9 or later. Replace `-3.10` with the desired
installed version when necessary.

## Clone the repository with Git

```powershell
git clone https://github.com/Daniel-Uhr/pydrlpdid.git
cd pydrlpdid
py -3.10 -m pip install .
```

## Install from a downloaded repository folder

1. On GitHub, choose **Code > Download ZIP**.
2. Extract the ZIP.
3. Open PowerShell inside the extracted folder that contains `pyproject.toml`.
4. Run:

```powershell
py -3.10 -m pip install --upgrade pip
py -3.10 -m pip install .
```

Verify the installation:

```powershell
py -3.10 -c "import pydrlpdid; from pydrlpdid import DRLPDID, LPDID; print(pydrlpdid.__version__); print(pydrlpdid.__file__)"
```

The expected stable version is `0.8.1`.

## Development installation

```powershell
py -3.10 -m pip install -e ".[dev]"
py -3.10 -m pytest -q
```

## Installing a downloaded wheel

A wheel command works only when the `.whl` file actually exists in the current
folder. After downloading the wheel from a GitHub Release, open PowerShell in
the download folder and run:

```powershell
py -3.10 -m pip install .\pydrlpdid-0.8.1-py3-none-any.whl
```
