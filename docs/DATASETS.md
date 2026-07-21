# Dataset provenance

The repository does not redistribute raw datasets. `scripts/download_datasets.py` downloads and decompresses the exact source files and verifies the frozen SHA-256 digests.

| Dataset | Target | Seasonality | Source form |
|---|---|---:|---|
| ETTh1 | `OT` | 24 | CSV |
| ETTm2 | `OT` | 96 | CSV |
| Energy (PJME) | `PJME_MW` | 24 | CSV |
| Exchange | last numeric column (`7`) | 5 | gzip-compressed text |
| Jena | `T (degC)` | 144 | ZIP-compressed CSV |
| AAPL | `Close` | 5 | CSV |

Exact URLs, file sizes, and hashes are in `results/manifests/dataset_raw_manifest.csv`. Users remain responsible for complying with each provider's terms and citation requirements.
