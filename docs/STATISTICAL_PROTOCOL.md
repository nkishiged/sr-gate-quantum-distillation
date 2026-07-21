# Statistical protocol

## H1

For each quantum teacher, SoftSR relative regret is compared separately with NaiveKD (H1a) and PersistenceGateKD (H1b) using one-sided paired Wilcoxon signed-rank tests across six dataset-level observations. Holm correction is applied to the two comparisons within each teacher.

## H2a

SoftSR is non-inferior to HardRejectSR when the upper 95% bootstrap confidence bound of the dataset-level mean regret difference is below `0.02`.

## H2b

Benefit retention is evaluated only where NaiveKD improves over Standalone by at least 1% and SoftSR activates the teacher. Fewer than two eligible datasets are labeled `INSUFFICIENT_EVIDENCE`.

## H3

Accepted-cell regret compares SoftSR with the better of Standalone and NaiveKD. Fewer than two accepted datasets are labeled `INSUFFICIENT_EVIDENCE`.

## H4

Directional generality reports the number of datasets favoring SoftSR over each comparator. It complements but does not replace H1 inference.
