# Reproducibility protocol

## Chronological partitions

The target timeline is divided into Train (70%), Calibration (15%), Development holdout (7.5%), and Final Test holdout (7.5%). The Final Test arrays are not materialized during `SMOKE` or `PROFILE`; only `FULL` can access them.

## Statistical unit

Seeds `(11, 23, 37, 51, 79)` are stochastic replications. Datasets are the top-level inferential units. Confirmatory inference averages seed-level results within each dataset.

## Pairing and endpoint invariance

For a fixed dataset and seed, student conditions use the same initialization and minibatch order. `g=0` reuses Standalone exactly and `g=1` reuses NaiveKD exactly. Only intermediate gates require a distinct student fit.

## Frozen protocol

- Lookback: 96
- Horizon: 24
- Quantum teachers: QELM, VQC, QRC, QKRR
- Secondary classical controls: RFF, ESN
- Soft thresholds: rejection `-0.02`, acceptance `0.05`
- Hard threshold: `0.00`
- Non-inferiority margin: `0.02`
- Gate bootstrap: 1,000 moving-block replications
- Statistical bootstrap: 5,000 dataset-level replications

## Confirmatory/secondary separation

H1--H4 are confirmatory. Harm-avoidance counts, benefit-retention counts, gate activity, intermediate cases, false acceptances, and quantum/classical descriptive comparisons are secondary post-confirmatory analyses.
