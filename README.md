# When Not to Distill: Student-Relative Gating for Quantum-to-CeNN Knowledge Distillation

Reproducibility repository for the sealed final-holdout benchmark evaluating student-relative selective knowledge distillation from heterogeneous simulated quantum teachers to a classical Temporal CeNN student for multi-horizon time-series forecasting.

## Scope of the evidence

The repository supports the following narrow conclusion: student-relative SoftSR gating reduces negative transfer relative to unconditional NaiveKD across the four pre-specified quantum teacher families. It does **not** establish quantum computational advantage, universal teacher superiority, or a formal no-harm guarantee.

## Frozen confirmatory run

| Field | Value |
|---|---|
| Version | `3.0.0` |
| Run tag | `4b53f0d205ce` |
| Run fingerprint | `4b53f0d205ce3756e5ce9b19a709eebeb26e2c9b52481ddc724208fa32d41d94` |
| Configuration SHA-256 | `6211de525ae086de0589ec87badd2051a29a17805df73c784145b532d3442d53` |
| Code fingerprint | `aec41faea5beb214f7651f63be1380b0379eb405af714751c3538f62e6a7990` |
| Dataset fingerprint | `232dac184cc896304cc751aab2697fe9bb6837bceaea16f55ecb5976822d71b5` |
| Raw results SHA-256 | `de68c22d7ebbb970e67d65a806eb2ea9642e0b0927fe415add3ca1420c75eb95` |
| Final-test rows | `900` |
| Complete teacher blocks | `180/180` |
| Execution environment | Python 3.12.13, PyTorch 2.10.0+cu128, NVIDIA Tesla T4 |

## Repository layout

```text
notebooks/                         Canonical executable notebook
src/                               Linear notebook export for code review
scripts/                           Download, execution, analysis, and verification tools
configs/                           Frozen confirmatory configuration
results/                           Immutable final-test results and analyses
results/manifests/                 Provenance, fingerprints, preprocessing, and window indices
paper/figures/                     Validated manuscript figures
paper/references_RINENG_revised.bib
                                                    Curated bibliography

docs/                              Protocol and release documentation
environment/                        Recorded and minimal software environments
tests/                              Lightweight artifact-integrity tests
.github/workflows/                  Continuous validation workflow
```

## Quick validation

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-core.txt
python scripts/verify_release.py
pytest -q
```

The validation checks the immutable result hash, row counts, run identity, experimental-cell uniqueness, finite metrics, gate bounds, and exact endpoint reuse for `g=0` and `g=1`.

## Reproduce the corrected analysis

```bash
python scripts/corrected_analysis.py \
  --results results/confirmatory_results_final_test_4b53f0d205ce.csv \
  --output build/corrected_analysis
```

The corrected analysis keeps H1--H4 confirmatory conclusions separate from post-confirmatory descriptive analyses. The manuscript-facing reference outputs are in `results/analysis/corrected_4b53f0d205ce/`.

## Download the six datasets

Raw datasets are not redistributed. Download and verify the exact source files with:

```bash
python scripts/download_datasets.py --output datasets --verify
```

The script decompresses Jena and Exchange to the exact CSV bytes used by the run and verifies every SHA-256 digest recorded in `results/manifests/dataset_raw_manifest.csv`.

## Execute the notebook

The canonical notebook contains four modes: `NONE`, `SMOKE`, `PROFILE`, and `FULL`. The helper executes each mode in an isolated workspace so that published results are not overwritten.

```bash
python scripts/run_notebook.py --mode SMOKE
python scripts/run_notebook.py --mode PROFILE
```

Final-test access requires an explicit confirmation flag:

```bash
python scripts/run_notebook.py --mode FULL --confirm-final-test
```

A FULL run is compute-intensive and downloads the datasets into its isolated workspace. The historical frozen run completed in approximately 39.3 minutes on one NVIDIA Tesla T4, but runtime is hardware-dependent.

## Primary artifacts

- `notebooks/Quantum_CeNN_Reproducible_T4_5H_V3_FinalHoldout.ipynb`
- `results/confirmatory_results_final_test_4b53f0d205ce.csv`
- `scripts/corrected_analysis.py`
- `configs/confirmatory_config.json`
- `results/manifests/run_identity_4b53f0d205ce.json`
- `results/manifests/artifact_manifest_sha256_4b53f0d205ce.csv`
- `release_manifest_sha256.csv`

## GitHub and Zenodo release

Follow `docs/GITHUB_ZENODO_RELEASE.md`. Before publication, replace repository placeholders in the manuscript with the immutable GitHub release URL, Git commit SHA, and Zenodo DOI.

## License and dataset terms

Repository code and documentation are released under the MIT License. The source datasets remain subject to their original providers' terms. See `docs/DATASETS.md`.

## Citation

GitHub reads `CITATION.cff`. Zenodo release metadata are provided in `.zenodo.json`. Verify author names, affiliations, ORCID identifiers, and the final release date before publishing the release.
