# Running the 0.7.2 replication files

Open a terminal in the extracted `pydrlpdid-0.7.2` directory.

```powershell
py -m pip install -e .
py -m pip install jupyter matplotlib pyreadstat
```

Install the article-compatible `diff_diff` release only if the external
benchmarks in the Bacon notebook are required:

```powershell
py -m pip install diff-diff==3.0.2
```

Start Jupyter:

```powershell
py -m jupyter notebook
```

Run the notebooks from top to bottom:

1. `notebooks/Application_1_Bacon_pydrlpdid_v0.7.2_LOCAL_H19.ipynb`
2. `notebooks/Application_2_Dube_and_Formal_Switching_pydrlpdid_v0.7.2.ipynb`
3. `notebooks/Monte_Carlo_DRLPDID_pydrlpdid_v0.7.2_FINAL.ipynb`

In the application notebooks, `LPDID-RA` denotes the Dube et al.
regression-adjustment benchmark. The RA member of the formal common-stack
family is reported separately as `DRLPDID-RA`. The Monte Carlo and formal
switching blocks study the formal family and therefore use the latter label.

The Bacon data are downloaded only if no local copy exists. The formal
switching notebook does not download the democracy data. Copy the certified
`DDCGdata_final.dta` to `data/`, or put the original
`Aplicacao_2_Dube_DRLPDID_v12_4.zip` in the project root so that the notebook
can extract the data and verify its SHA-256 hash.

The third notebook is the complete Monte Carlo driver. It can run the
absorbing branch only or the full absorbing and switching design. Every
replication is written atomically to a checkpoint directory signed by the
package source hash and run settings.

The smaller dedicated absorbing driver remains available:

```powershell
py examples\monte_carlo_absorbing_N500.py
```

The default is 500 replications with `N=500`. For a non-certified smoke test:

```powershell
$env:DRLPDID_MC_REPS="2"
py examples\monte_carlo_absorbing_N500.py
```

Every application directory contains machine-readable estimates, inference,
diagnostics, and a JSON manifest. The Monte Carlo stops without exporting a
certified summary if any fitted model fails.
