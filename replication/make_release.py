"""Create source and complete release archives with SHA-256 manifests."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "release_072"
PREFIX = Path("pydrlpdid-0.7.2")


def selected_files(include_wheel: bool, include_data: bool) -> list[Path]:
    patterns = [
        "*.md",
        "LICENSE",
        "pyproject.toml",
        "requirements-replication.txt",
        "src/pydrlpdid/*.py",
        "tests/*.py",
        "examples/*.py",
        "notebooks/*.ipynb",
        "replication/*.py",
    ]
    if include_wheel:
        patterns.append("dist/pydrlpdid-0.7.2-*.whl")
    if include_data:
        patterns.append("data/*.dta")
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in ROOT.glob(pattern) if path.is_file())
    return sorted(files, key=lambda path: path.as_posix())


def make_zip(path: Path, include_wheel: bool, include_data: bool) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in selected_files(include_wheel, include_data):
            arcname = PREFIX / source.relative_to(ROOT)
            archive.write(source, arcname.as_posix())


def make_notebooks_zip(path: Path) -> None:
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for source in sorted((ROOT / "notebooks").glob("*.ipynb")):
            archive.write(source, (Path("notebooks") / source.name).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wheel_source = ROOT / "dist" / "pydrlpdid-0.7.2-py3-none-any.whl"
    if not wheel_source.exists():
        raise FileNotFoundError("Build the wheel before creating the release.")
    wheel = OUT / wheel_source.name
    shutil.copy2(wheel_source, wheel)

    source_zip = OUT / "pydrlpdid-0.7.2-source.zip"
    complete_zip = OUT / "pydrlpdid-0.7.2-complete.zip"
    notebooks_zip = OUT / "pydrlpdid-0.7.2-notebooks.zip"
    make_zip(source_zip, include_wheel=False, include_data=False)
    make_zip(complete_zip, include_wheel=True, include_data=True)
    make_notebooks_zip(notebooks_zip)

    standalone = [
        ROOT / "examples" / "monte_carlo_absorbing_N500.py",
        ROOT / "notebooks" / "Application_1_Bacon_pydrlpdid_v0.7.2_LOCAL_H19.ipynb",
        ROOT / "notebooks" / "Application_2_Dube_and_Formal_Switching_pydrlpdid_v0.7.2.ipynb",
        ROOT / "notebooks" / "Monte_Carlo_DRLPDID_pydrlpdid_v0.7.2_FINAL.ipynb",
    ]
    for source in standalone:
        shutil.copy2(source, OUT / source.name)

    deliverables = sorted(path for path in OUT.iterdir() if path.is_file())
    manifest = {
        "release": "pydrlpdid 0.7.2",
        "deliverables": [
            {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in deliverables
            if path.name not in {"SHA256SUMS.txt", "release_manifest.json"}
        ],
    }
    (OUT / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    checksum_lines = [
        f"{item['sha256']}  {item['filename']}"
        for item in manifest["deliverables"]
    ]
    checksum_lines.append(
        f"{sha256(OUT / 'release_manifest.json')}  release_manifest.json"
    )
    (OUT / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
