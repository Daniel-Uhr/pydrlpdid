from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "monte_carlo_final_v072_engine.py"
OUTPUT = (
    ROOT.parent
    / "notebooks"
    / "Monte_Carlo_DRLPDID_pydrlpdid_v0.7.2_FINAL.ipynb"
)


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    text = source.strip() + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


engine = ENGINE.read_text(encoding="utf-8")
markers = [
    "# ---------------------------------------------------------------------------\n"
    "# Absorbing-adoption DGP\n"
    "# ---------------------------------------------------------------------------",
    "# ---------------------------------------------------------------------------\n"
    "# Sustained-switch-in DGP\n"
    "# ---------------------------------------------------------------------------",
    "# ---------------------------------------------------------------------------\n"
    "# Checkpointing, summaries, diagnostics, and article files\n"
    "# ---------------------------------------------------------------------------",
]
parts = []
remaining = engine
for marker in markers:
    before, separator, remaining = remaining.partition(marker)
    if not separator:
        raise RuntimeError(f"Notebook split marker not found: {marker}")
    parts.append(before)
    remaining = marker + remaining
parts.append(remaining)

cells = [
    markdown(
        r"""
# Final article Monte Carlo — `pydrlpdid` 0.7.2

This notebook regenerates **all Monte Carlo results retained in the article**
under one frozen implementation:

1. absorbing adoption with \(N=500\), four nuisance specifications and 500
   replications;
2. absorbing-adoption sensitivity to violations of local conditional parallel
   trends;
3. formal sustained switch-ins with \(N\in\{184,368\}\), four nuisance
   specifications and 1,000 replications per cell;
4. the dynamic switching path under correctly specified nuisance functions;
5. switching sensitivity to pre-existing and post-entry untreated-outcome
   drift.

The removed \(N=250\), fixed-design, support-threshold, re-entry, and alternative
target exercises are intentionally **not** part of this notebook.

The public API uses one covariate list. Controlled nuisance misspecification is
implemented here through the package's private replication hook
`DRLPDID._fit_core`, with separate PS and OR bases. This does not expand the
public estimator.
"""
    ),
    markdown(
        r"""
## 0. Local package and reproducibility

Place this notebook in:

```text
C:\Users\danie\OneDrive\1 - Pesquisas\0 - DRLPDID\
pydrlpdid-0.7.2\notebooks
```

Run it with the kernel in which the local package dependencies are installed.
The setup below imports `src/pydrlpdid` from the same project and refuses to run
with a version other than 0.7.2.
"""
    ),
    code(
        r"""
from pathlib import Path
import os
import sys


def find_project_root() -> Path:
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    for candidate in candidates:
        candidate = candidate.resolve()
        if (
            (candidate / "pyproject.toml").exists()
            and (candidate / "src" / "pydrlpdid" / "__init__.py").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not locate the pydrlpdid-0.7.2 project root. "
        "Place this notebook in the project's notebooks directory."
    )


PROJECT_ROOT = find_project_root()
SOURCE_DIR = PROJECT_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

print("Project root:", PROJECT_ROOT)
print("Local source:", SOURCE_DIR)
"""
    ),
    markdown("## 1. Configuration, estimators, and common helpers"),
    code(parts[0]),
    markdown("## 2. Absorbing-adoption data-generating process"),
    code(parts[1]),
    markdown("## 3. Formal sustained-switch-in data-generating process"),
    code(parts[2]),
    markdown("## 4. Checkpoints, article tables, and numerical certification"),
    code(parts[3]),
    markdown(
        r"""
## 5. Prespecified run

`paper` is the final run: 500 absorbing replications and 1,000 switching
replications per sample size. `RUN_SCOPE="all"` executes both designs, while
`RUN_SCOPE="absorbing"` executes only the 500-replication absorbing branch.
The latter certifies the absorbing tables but not the complete article.

The full run can take several hours. Every replication is checkpointed, so
rerunning the cell resumes rather than discards completed work.

For a two-replication wiring check, temporarily change `RUN_MODE` to `smoke`.
A smoke run is never certified for use in the article.
"""
    ),
    code(
        r"""
RUN_MODE = os.environ.get("DRLPDID_MC_MODE", "paper").strip().lower()
RUN_SCOPE = os.environ.get("DRLPDID_MC_SCOPE", "all").strip().lower()
RESULTS_DIR = (
    PROJECT_ROOT
    / "mc_article_v072_results"
    / f"{RUN_MODE}_{RUN_SCOPE}"
)
SETTINGS = make_settings(RUN_MODE, RESULTS_DIR, scope=RUN_SCOPE)

print("Run mode:", SETTINGS.mode)
print("Run scope:", SETTINGS.scope)
if SETTINGS.scope in {"all", "absorbing"}:
    print("Absorbing replications:", SETTINGS.absorbing_replications)
if SETTINGS.scope in {"all", "switching"}:
    print("Switching replications per N:", SETTINGS.switching_replications)
print("Results directory:", RESULTS_DIR)
print("Package:", __version__)
print("Core source:", PACKAGE_SOURCE_FILE)
print("Core SHA-256:", _hash_file(PACKAGE_SOURCE_FILE))
"""
    ),
    markdown(
        r"""
## 6. Execute or resume

No failed fit is dropped. In `paper` mode the final export raises an error
unless all requested cells are complete, every IPT score check passes, every
nested IPT–DRLPDID-IPT identity passes its measured tolerance, and the failure
count is zero.
"""
    ),
    code(
        r"""
RESULTS = run_article_monte_carlo(SETTINGS)
print(json.dumps(RESULTS["certification"], indent=2, ensure_ascii=False))
"""
    ),
    markdown("## 7. Article tables"),
    code(
        r"""
from IPython.display import display

if SETTINGS.scope in {"all", "absorbing"}:
    print("Absorbing adoption: four nuisance scenarios")
    display(RESULTS["absorbing_main"])

    print("Absorbing adoption: conditional-parallel-trends sensitivity")
    display(RESULTS["absorbing_pt"])

    print("Absorbing adoption: horizon-level appendix results")
    display(RESULTS["absorbing_horizon"])

if SETTINGS.scope in {"all", "switching"}:
    print("Sustained switch-ins: four nuisance scenarios")
    display(RESULTS["switching_main"])

    print("Sustained switch-ins: dynamic path")
    display(
        RESULTS["switching_path"].loc[
            RESULTS["switching_path"]["horizon"].isin([0, 2, 4, 6, 8, 10])
        ]
    )

    print("Sustained switch-ins: conditional-parallel-trends sensitivity")
    display(RESULTS["switching_pt"])
"""
    ),
    markdown(
        r"""
## 8. Numerical audit

The package reports the maximum absolute IPT score residual as an observation
mean. The notebook additionally rescales it by the treated-event count, records
the retained and dropped columns in both nuisance systems, and directly compares
IPT and DRLPDID-IPT event-study estimates.
"""
    ),
    code(
        r"""
print("Failures")
display(RESULTS["failures"])

print("IPT score, retained-rank, and nesting audit")
display(RESULTS["diagnostic_summary"])

identity = RESULTS["identity_audit"]
display(
    identity.loc[identity["expected_nested"]]
    .sort_values("absolute_difference", ascending=False)
    .head(25)
)
"""
    ),
    markdown("## 9. Output manifest"),
    code(
        r"""
output_files = sorted(
    path.relative_to(RESULTS_DIR)
    for path in RESULTS_DIR.rglob("*")
    if path.is_file() and "checkpoints" not in path.parts
)
for path in output_files:
    print(path)
"""
    ),
    markdown(
        r"""
## Certification rule

The results may replace the archived 0.6.4 Monte Carlo tables only when
`certification.json` reports:

```text
"mode": "paper"
"paper_replication_counts": true
"zero_failed_fits": true
"complete_replication_counts": true
"all_ipt_mean_score_checks_pass": true
"all_nested_identity_checks_pass": true
"certified_for_requested_scope": true
```

For `RUN_SCOPE="absorbing"`, `certified_absorbing` must be true and
`certified_for_article` remains false because switching was not executed. For
`RUN_SCOPE="all"`, `certified_for_article` must also be true.

The central article-ready files are in `tables/`. The raw replication-level
files, failure ledger, score diagnostics, retained-basis audit, package source
hash, seeds, and SHA-256 manifest remain in the same results directory.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False),
    encoding="utf-8",
)
print(OUTPUT)
