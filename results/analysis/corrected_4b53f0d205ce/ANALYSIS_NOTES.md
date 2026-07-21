# Corrected analysis notes — run 4b53f0d205ce

## Immutable confirmatory input

- File: `confirmatory_results_final_test_4b53f0d205ce.csv`
- SHA-256: `de68c22d7ebbb970e67d65a806eb2ea9642e0b0927fe415add3ca1420c75eb95`
- Rows: 900
- Evaluation split: `final_test` only

The original FULL run is not modified or rerun.

## Confirmatory corrections

1. H1 is reported as H1a (SoftSR vs NaiveKD) and H1b
   (SoftSR vs PersistenceGateKD), rather than one global binary claim.
2. H2a remains the pre-specified non-inferiority analysis.
3. H2b retains the pre-specified benefit-retention estimator, but an effect
   supported by only one eligible dataset is labeled `INSUFFICIENT_EVIDENCE`.
4. H3 uses the same rule: a one-dataset accepted-cell result is descriptive,
   not confirmatory evidence.
5. H4 remains directional/descriptive.

## Secondary post-confirmatory analyses

The following analyses were formulated after inspection of the final test and
must be labeled secondary/descriptive in the manuscript:

- harm avoidance rate;
- benefit retention rate;
- false acceptance cases;
- missed-benefit rate;
- gate activity;
- intermediate attenuation case studies;
- dataset-level absolute metric diagnostics.

Threshold for classifying a NaiveKD effect as practically harmful or beneficial:
`1%` relative to Standalone.

## Key descriptive results

- Harmful NaiveKD quantum cells: 98/120
- Harm avoided by SoftSR: 96/98
  (98.0%)
- Beneficial NaiveKD quantum cells: 21/120
- Benefit retained by SoftSR: 6/21
  (28.6%)
- False acceptance cases remaining harmful: 2
- Intermediate SoftSR gates: 2
