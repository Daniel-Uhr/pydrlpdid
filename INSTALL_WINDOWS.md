# Installation on Windows

## Recommended: install from a downloaded repository folder

1. On GitHub, choose **Code > Download ZIP**.
2. Extract the ZIP.
3. Open PowerShell inside the extracted folder that contains `pyproject.toml`.
4. Run:

```powershell
py -m pip install --upgrade pip
py -m pip install .
```

Verify the installation:

```powershell
py -c "import pydrlpdid; from pydrlpdid import DRLPDID, LPDID; print(pydrlpdid.__version__); print(pydrlpdid.__file__)"
```

The expected release-candidate version is `0.8.0rc3`.

## Development installation

```powershell
py -m pip install -e ".[dev]"
py -m pytest -q
```

## Installing a downloaded wheel

A wheel command works only when the `.whl` file actually exists in the current
folder. After downloading the wheel from a GitHub Release, open PowerShell in
the download folder and run:

```powershell
py -m pip install .\pydrlpdid-0.8.0rc3-py3-none-any.whl
```
