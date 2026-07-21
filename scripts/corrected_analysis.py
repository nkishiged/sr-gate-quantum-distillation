#!/usr/bin/env python3
"""
Corrected confirmatory and secondary analysis for run 4b53f0d205ce.

Usage:
    python corrected_analysis.py \
        --results confirmatory_results_final_test_4b53f0d205ce.csv \
        --output analysis_corrected

This script does NOT alter or rerun the confirmatory experiment.

Confirmatory family:
- H1a/H1b
- H2a
- H2b with corrected inference labeling
- H3 with corrected inference labeling
- H4

Secondary post-confirmatory descriptive analyses:
- harm avoidance
- benefit retention
- false acceptance
- missed benefit
- gate activity
- intermediate attenuation cases

The post-confirmatory analyses must be labeled descriptive/secondary in a paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

RUN_TAG = "4b53f0d205ce"
QUANTUM_TEACHERS = ("QELM", "VQC", "QRC", "QKRR")
DELTA_NI = 0.02
BENEFIT_THRESHOLD = 0.01
N_BOOT = 5000
RNG_SEED = 20260721
EXPECTED_RESULTS_SHA256 = (
    "de68c22d7ebbb970e67d65a806eb2ea9642e0b0927fe415add3ca1420c75eb95"
)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def bootstrap_mean_ci(values, seed):
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, len(x), len(x))
        stats[i] = x[idx].mean()
    return (
        float(x.mean()),
        float(np.quantile(stats, 0.025)),
        float(np.quantile(stats, 0.975)),
    )

def holm_two(p1, p2):
    p = np.array([p1, p2], dtype=float)
    order = np.argsort(p)
    adj = np.empty(2)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (2 - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj

def wilcoxon_less(diff):
    x = np.asarray(diff, dtype=float)
    if np.allclose(x, 0):
        return 0.0, 1.0
    try:
        s, p = wilcoxon(
            x, alternative="less", zero_method="wilcox", method="auto"
        )
        return float(s), float(p)
    except ValueError:
        return np.nan, np.nan

def wilcoxon_greater(diff):
    x = np.asarray(diff, dtype=float)
    if len(x) < 2 or np.allclose(x, 0):
        return np.nan, np.nan
    try:
        s, p = wilcoxon(
            x, alternative="greater", zero_method="wilcox", method="auto"
        )
        return float(s), float(p)
    except ValueError:
        return np.nan, np.nan

def validate(df, results_path):
    if sha256_file(results_path) != EXPECTED_RESULTS_SHA256:
        raise RuntimeError("The confirmatory result SHA-256 does not match the frozen run.")
    if len(df) != 900:
        raise RuntimeError(f"Expected 900 rows, got {len(df)}.")
    if set(df["evaluation_split"].unique()) != {"final_test"}:
        raise RuntimeError("Only final_test rows are allowed.")
    if set(df["run_tag"].astype(str).unique()) != {RUN_TAG}:
        raise RuntimeError("Unexpected run_tag.")
    if df[["dataset","seed","teacher","method"]].duplicated().any():
        raise RuntimeError("Duplicate experimental cells detected.")
    metrics = ["eval_MASE","eval_MAE","eval_RMSE","eval_MSE","eval_sMAPE","gate"]
    if not np.isfinite(df[metrics].to_numpy(float)).all():
        raise RuntimeError("Non-finite metric/gate value detected.")

def build_wide(df):
    wide = df.pivot_table(
        index=["dataset","seed","teacher","teacher_family"],
        columns="method",
        values=["eval_MASE","gate"],
        aggfunc="first",
    )
    wide.columns = [f"{a}__{b}" for a,b in wide.columns]
    wide = wide.reset_index()
    for method in ("NaiveKD","PersistenceGateKD","HardRejectSR","SoftSR"):
        wide[f"regret__{method}"] = (
            wide[f"eval_MASE__{method}"] - wide["eval_MASE__Standalone"]
        ) / np.maximum(wide["eval_MASE__Standalone"], 1e-12)
    return wide



def generate_extended_outputs(df, wide, dataset_level, output):
    """Generate the complete manuscript-facing corrected analysis package."""
    # Add explicit interpretation fields to confirmatory outputs.
    h2b_path = output / "H2b_benefit_retention_corrected.csv"
    h2b = pd.read_csv(h2b_path)
    h2b["original_effect_direction"] = np.where(
        h2b["mean_retention_diff"] > 0,
        "positive",
        "zero_or_negative",
    )
    ordered = [
        "teacher", "n_datasets", "n_seed_cells", "mean_retention_diff",
        "ci_low", "ci_high", "wilcoxon_p_one_sided", "benefit_min_relative",
        "original_effect_direction", "inference_status",
    ]
    h2b[ordered].to_csv(h2b_path, index=False)

    h4_path = output / "H4_corrected.csv"
    h4 = pd.read_csv(h4_path)
    h4["interpretation"] = np.where(
        h4["n_favorable_datasets"] == h4["n_datasets"],
        "DIRECTION_GENERALIZES",
        "DIRECTION_NOT_UNIVERSAL",
    )
    h4.to_csv(h4_path, index=False)

    q = wide[wide.teacher.isin(QUANTUM_TEACHERS)].copy()
    q["naive_outcome"] = np.select(
        [
            q["regret__NaiveKD"] > BENEFIT_THRESHOLD,
            q["regret__NaiveKD"] < -BENEFIT_THRESHOLD,
        ],
        ["harmful", "beneficial"],
        default="neutral",
    )

    harm = q[q.naive_outcome == "harmful"]
    benefit = q[q.naive_outcome == "beneficial"]
    neutral = q[q.naive_outcome == "neutral"]
    harm_avoided = harm["regret__SoftSR"] <= BENEFIT_THRESHOLD
    false_accept = (
        (harm["gate__SoftSR"] > 0)
        & (harm["regret__SoftSR"] > BENEFIT_THRESHOLD)
    )
    benefit_retained = benefit["regret__SoftSR"] < -BENEFIT_THRESHOLD
    active = q["gate__SoftSR"] > 0
    intermediate = active & (q["gate__SoftSR"] < 1)

    overall = pd.DataFrame([{
        "family": "quantum",
        "n_cells": len(q),
        "harmful_naive_n": len(harm),
        "harmful_naive_rate": len(harm) / len(q),
        "harm_avoided_n": int(harm_avoided.sum()),
        "harm_avoidance_rate": float(harm_avoided.mean()),
        "false_acceptance_n": int(false_accept.sum()),
        "false_acceptance_rate": float(false_accept.mean()),
        "beneficial_naive_n": len(benefit),
        "beneficial_naive_rate": len(benefit) / len(q),
        "benefit_retained_n": int(benefit_retained.sum()),
        "benefit_retention_rate": float(benefit_retained.mean()),
        "missed_benefit_n": int((~benefit_retained).sum()),
        "missed_benefit_rate": float((~benefit_retained).mean()),
        "neutral_naive_n": len(neutral),
        "active_gate_n": int(active.sum()),
        "active_gate_rate": float(active.mean()),
        "intermediate_gate_n": int(intermediate.sum()),
        "intermediate_gate_rate": float(intermediate.mean()),
    }])
    overall.to_csv(output / "secondary_harm_benefit_quantum_overall.csv", index=False)

    by_teacher = []
    for teacher in QUANTUM_TEACHERS:
        wt = q[q.teacher == teacher]
        wh = wt[wt.naive_outcome == "harmful"]
        wb = wt[wt.naive_outcome == "beneficial"]
        ha = wh["regret__SoftSR"] <= BENEFIT_THRESHOLD
        br = wb["regret__SoftSR"] < -BENEFIT_THRESHOLD
        by_teacher.append({
            "teacher": teacher,
            "n_cells": len(wt),
            "harmful_naive_n": len(wh),
            "harm_avoidance_n": int(ha.sum()),
            "harm_avoidance_rate": float(ha.mean()),
            "false_acceptance_n": int((
                (wh["gate__SoftSR"] > 0)
                & (wh["regret__SoftSR"] > BENEFIT_THRESHOLD)
            ).sum()),
            "beneficial_naive_n": len(wb),
            "benefit_retained_n": int(br.sum()),
            "benefit_retention_rate": float(br.mean()),
            "missed_benefit_n": int((~br).sum()),
            "active_gate_n": int((wt["gate__SoftSR"] > 0).sum()),
            "intermediate_gate_n": int((
                (wt["gate__SoftSR"] > 0)
                & (wt["gate__SoftSR"] < 1)
            ).sum()),
        })
    by_teacher = pd.DataFrame(by_teacher).sort_values("teacher")
    by_teacher.to_csv(output / "secondary_harm_benefit_by_teacher.csv", index=False)

    failure_cols = [
        "dataset", "seed", "teacher", "eval_MASE__Standalone",
        "eval_MASE__NaiveKD", "eval_MASE__SoftSR", "regret__NaiveKD",
        "regret__SoftSR", "gate__SoftSR",
    ]
    failures = q[
        (q.naive_outcome == "harmful")
        & (q["regret__SoftSR"] > BENEFIT_THRESHOLD)
    ].sort_values(["dataset", "seed", "teacher"])
    failures[failure_cols].to_csv(
        output / "secondary_false_acceptance_cases.csv", index=False
    )

    attenuation_cols = [
        "dataset", "seed", "teacher", "eval_MASE__Standalone",
        "eval_MASE__NaiveKD", "eval_MASE__HardRejectSR",
        "eval_MASE__SoftSR", "regret__NaiveKD", "regret__HardRejectSR",
        "regret__SoftSR", "gate__SoftSR",
    ]
    attenuation = q[
        (q["gate__SoftSR"] > 0)
        & (q["gate__SoftSR"] < 1)
    ].sort_values(["dataset", "seed", "teacher"])
    attenuation[attenuation_cols].to_csv(
        output / "secondary_intermediate_attenuation_cases.csv", index=False
    )

    gate_activity = (
        q.groupby("teacher")
        .agg(
            n_cells=("gate__SoftSR", "size"),
            gate_zero_n=("gate__SoftSR", lambda x: int((x == 0).sum())),
            gate_one_n=("gate__SoftSR", lambda x: int((x == 1).sum())),
            gate_intermediate_n=(
                "gate__SoftSR", lambda x: int(((x > 0) & (x < 1)).sum())
            ),
        )
        .reset_index()
        .sort_values("teacher")
    )
    gate_activity.to_csv(output / "secondary_gate_activity_quantum.csv", index=False)

    active_counts = (
        q.assign(active=(q["gate__SoftSR"] > 0).astype(int))
        .pivot_table(index="dataset", columns="teacher", values="active", aggfunc="sum", fill_value=0)
        .reindex(columns=list(QUANTUM_TEACHERS), fill_value=0)
        .reset_index()
    )
    active_counts.to_csv(
        output / "secondary_teacher_dataset_active_gate_counts.csv", index=False
    )

    standalone = (
        df[df.method == "Standalone"]
        .drop_duplicates(["dataset", "seed"])
    )
    standalone_metrics = (
        standalone.groupby("dataset")
        .agg(
            MASE_mean=("eval_MASE", "mean"),
            MASE_median=("eval_MASE", "median"),
            MASE_std=("eval_MASE", "std"),
            MAE_mean=("eval_MAE", "mean"),
            RMSE_mean=("eval_RMSE", "mean"),
            sMAPE_mean=("eval_sMAPE", "mean"),
        )
        .reset_index()
    )
    standalone_metrics.to_csv(output / "standalone_metrics_by_dataset.csv", index=False)

    h1 = pd.read_csv(output / "H1_corrected.csv")
    h2a = pd.read_csv(output / "H2a_noninferiority_corrected.csv")
    h3 = pd.read_csv(output / "H3_corrected.csv")
    conclusion = pd.DataFrame({"teacher": QUANTUM_TEACHERS})
    conclusion["H1a_vs_NaiveKD"] = [
        h1[(h1.teacher == t) & (h1.subhypothesis == "H1a")].status.iloc[0]
        for t in QUANTUM_TEACHERS
    ]
    conclusion["H1b_vs_PersistenceGateKD"] = [
        h1[(h1.teacher == t) & (h1.subhypothesis == "H1b")].status.iloc[0]
        for t in QUANTUM_TEACHERS
    ]
    conclusion["H2a_noninferiority_vs_HardReject"] = [
        h2a[h2a.teacher == t].status.iloc[0] for t in QUANTUM_TEACHERS
    ]
    conclusion["H2b_benefit_retention"] = [
        h2b[h2b.teacher == t].inference_status.iloc[0] for t in QUANTUM_TEACHERS
    ]
    conclusion["H3_accepted_cell_noninferiority"] = [
        h3[h3.teacher == t].inference_status.iloc[0] for t in QUANTUM_TEACHERS
    ]
    conclusion.to_csv(output / "confirmatory_conclusion_matrix.csv", index=False)

    # Figure 1: dataset-level relative regret.
    methods = ["NaiveKD", "PersistenceGateKD", "HardRejectSR", "SoftSR"]
    qd = dataset_level[dataset_level.teacher.isin(QUANTUM_TEACHERS)]
    values = [qd[f"regret__{m}"].to_numpy() for m in methods]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.boxplot(values, tick_labels=methods, showmeans=True)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Dataset-level relative regret vs Standalone")
    ax.set_title(
        "Quantum teachers: relative regret across dataset–teacher blocks\n"
        "(seeds averaged within dataset)"
    )
    fig.tight_layout()
    fig.savefig(output / "figure_relative_regret_quantum.pdf", bbox_inches="tight")
    fig.savefig(output / "figure_relative_regret_quantum.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: descriptive harm/benefit counts.
    row = overall.iloc[0]
    labels = [
        "Harmful\nNaiveKD", "Harm avoided\nby SoftSR",
        "Beneficial\nNaiveKD", "Benefit retained\nby SoftSR",
    ]
    counts = [
        int(row.harmful_naive_n), int(row.harm_avoided_n),
        int(row.beneficial_naive_n), int(row.benefit_retained_n),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.5, str(value), ha="center")
    ax.set_ylabel("Number of quantum teacher seed-cells")
    ax.set_title("Post-confirmatory descriptive analysis: harm avoidance vs benefit retention")
    fig.tight_layout()
    fig.savefig(output / "figure_harm_benefit_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(output / "figure_harm_benefit_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: SoftSR gate states.
    ga = gate_activity.set_index("teacher").loc[list(sorted(QUANTUM_TEACHERS))]
    x = np.arange(len(ga))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, ga.gate_zero_n, label="g = 0")
    ax.bar(x, ga.gate_intermediate_n, bottom=ga.gate_zero_n, label="0 < g < 1")
    ax.bar(
        x, ga.gate_one_n,
        bottom=ga.gate_zero_n + ga.gate_intermediate_n,
        label="g = 1",
    )
    ax.set_xticks(x, ga.index)
    ax.set_ylabel("Number of seed-cells")
    ax.set_title("SoftSR gate activity across quantum teachers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "figure_softsr_gate_activity.pdf", bbox_inches="tight")
    fig.savefig(output / "figure_softsr_gate_activity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(results_path: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_path)
    validate(df, results_path)
    wide = build_wide(df)
    wide.to_csv(output / "wide_seed_level_recomputed.csv", index=False)

    regrets = [
        "regret__NaiveKD",
        "regret__PersistenceGateKD",
        "regret__HardRejectSR",
        "regret__SoftSR",
    ]
    dataset_level = (
        wide.groupby(["dataset","teacher","teacher_family"], as_index=False)[regrets]
        .mean()
    )
    dataset_level.to_csv(output / "dataset_level_relative_regret.csv", index=False)

    # H1
    h1 = []
    for teacher in QUANTUM_TEACHERS:
        wt = dataset_level[dataset_level.teacher == teacher]
        rows = []
        for comparator, sub in (
            ("NaiveKD","H1a"),
            ("PersistenceGateKD","H1b"),
        ):
            d = (
                wt["regret__SoftSR"].to_numpy()
                - wt[f"regret__{comparator}"].to_numpy()
            )
            stat, p = wilcoxon_less(d)
            mean, lo, hi = bootstrap_mean_ci(
                d, RNG_SEED + sum(map(ord, teacher + comparator))
            )
            rows.append({
                "teacher": teacher,
                "subhypothesis": sub,
                "comparator": comparator,
                "n_datasets": len(d),
                "mean_diff_regret": mean,
                "median_diff_regret": float(np.median(d)),
                "ci_low": lo,
                "ci_high": hi,
                "n_favorable_datasets": int((d < 0).sum()),
                "fraction_favorable": float((d < 0).mean()),
                "wilcoxon_stat": stat,
                "p_raw": p,
            })
        adj = holm_two(rows[0]["p_raw"], rows[1]["p_raw"])
        for r, pa in zip(rows, adj):
            r["p_holm_within_teacher"] = float(pa)
            r["status"] = "SUPPORTED" if pa < 0.05 else "NOT_SUPPORTED"
            h1.append(r)
    H1 = pd.DataFrame(h1)
    H1.to_csv(output / "H1_corrected.csv", index=False)

    # H2a
    h2a = []
    for teacher in QUANTUM_TEACHERS:
        wt = dataset_level[dataset_level.teacher == teacher]
        d = wt["regret__SoftSR"].to_numpy() - wt["regret__HardRejectSR"].to_numpy()
        mean, lo, hi = bootstrap_mean_ci(
            d, RNG_SEED + sum(map(ord, teacher + "H2a"))
        )
        h2a.append({
            "teacher": teacher,
            "n_datasets": len(d),
            "mean_diff_soft_minus_hard": mean,
            "ci_low": lo,
            "ci_high": hi,
            "delta_NI": DELTA_NI,
            "non_inferior": bool(hi < DELTA_NI),
            "status": "SUPPORTED" if hi < DELTA_NI else "NOT_SUPPORTED",
        })
    H2a = pd.DataFrame(h2a)
    H2a.to_csv(output / "H2a_noninferiority_corrected.csv", index=False)

    q = wide[wide.teacher.isin(QUANTUM_TEACHERS)].copy()

    # H2b
    h2b = []
    for teacher in QUANTUM_TEACHERS:
        wt = q[q.teacher == teacher].copy()
        eligible = wt[
            (-wt["regret__NaiveKD"] >= BENEFIT_THRESHOLD)
            & (wt["gate__SoftSR"] > 0)
        ].copy()

        if eligible.empty:
            h2b.append({
                "teacher": teacher,
                "n_datasets": 0,
                "n_seed_cells": 0,
                "mean_retention_diff": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "wilcoxon_p_one_sided": np.nan,
                "benefit_min_relative": BENEFIT_THRESHOLD,
                "inference_status": "INSUFFICIENT_EVIDENCE",
            })
            continue

        eligible["naive_benefit"] = -eligible["regret__NaiveKD"]
        eligible["soft_benefit"] = -eligible["regret__SoftSR"]
        eligible["hard_benefit"] = -eligible["regret__HardRejectSR"]
        eligible["retention_diff"] = (
            eligible["soft_benefit"] / eligible["naive_benefit"]
            - eligible["hard_benefit"] / eligible["naive_benefit"]
        )

        per_dataset = eligible.groupby("dataset")["retention_diff"].mean().to_numpy()
        mean, lo, hi = bootstrap_mean_ci(
            per_dataset, RNG_SEED + sum(map(ord, teacher + "H2b"))
        )
        _, p = wilcoxon_greater(per_dataset)

        if len(per_dataset) < 2 or not np.isfinite(p):
            status = "INSUFFICIENT_EVIDENCE"
        elif p < 0.05 and mean > 0:
            status = "SUPPORTED"
        else:
            status = "NOT_SUPPORTED"

        h2b.append({
            "teacher": teacher,
            "n_datasets": len(per_dataset),
            "n_seed_cells": len(eligible),
            "mean_retention_diff": mean,
            "ci_low": lo,
            "ci_high": hi,
            "wilcoxon_p_one_sided": p,
            "benefit_min_relative": BENEFIT_THRESHOLD,
            "inference_status": status,
        })

    H2b = pd.DataFrame(h2b)
    H2b.to_csv(output / "H2b_benefit_retention_corrected.csv", index=False)

    H2s = H2a[["teacher","non_inferior"]].rename(
        columns={"non_inferior":"H2a_non_inferior"}
    ).merge(H2b[["teacher","inference_status"]], on="teacher")
    H2s = H2s.rename(columns={"inference_status":"H2b_status"})
    H2s["H2_overall_status"] = np.where(
        H2s.H2a_non_inferior & (H2s.H2b_status == "SUPPORTED"),
        "SUPPORTED",
        np.where(
            H2s.H2a_non_inferior,
            "PARTIAL_SUPPORT_H2A_ONLY",
            "NOT_SUPPORTED",
        ),
    )
    H2s.to_csv(output / "H2_summary_corrected.csv", index=False)

    # H3
    h3 = []
    for teacher in QUANTUM_TEACHERS:
        accepted = q[
            (q.teacher == teacher)
            & (q["gate__SoftSR"] > 0)
        ].copy()

        if accepted.empty:
            h3.append({
                "teacher": teacher,
                "n_datasets_with_acceptance": 0,
                "n_seed_cells_accepted": 0,
                "mean_regret": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "delta_NI": DELTA_NI,
                "numerical_non_inferior": False,
                "inference_status": "INSUFFICIENT_EVIDENCE",
            })
            continue

        best = np.minimum(
            accepted["eval_MASE__Standalone"].to_numpy(),
            accepted["eval_MASE__NaiveKD"].to_numpy(),
        )
        accepted["accepted_regret"] = (
            accepted["eval_MASE__SoftSR"].to_numpy() - best
        ) / np.maximum(best, 1e-12)

        values = accepted.groupby("dataset")["accepted_regret"].mean().to_numpy()
        mean, lo, hi = bootstrap_mean_ci(
            values, RNG_SEED + sum(map(ord, teacher + "H3"))
        )
        numeric_ni = hi < DELTA_NI

        status = (
            "INSUFFICIENT_EVIDENCE"
            if len(values) < 2
            else ("SUPPORTED" if numeric_ni else "NOT_SUPPORTED")
        )

        h3.append({
            "teacher": teacher,
            "n_datasets_with_acceptance": len(values),
            "n_seed_cells_accepted": len(accepted),
            "mean_regret": mean,
            "ci_low": lo,
            "ci_high": hi,
            "delta_NI": DELTA_NI,
            "numerical_non_inferior": numeric_ni,
            "inference_status": status,
        })
    pd.DataFrame(h3).to_csv(output / "H3_corrected.csv", index=False)

    # H4
    h4 = []
    for teacher in QUANTUM_TEACHERS:
        wt = dataset_level[dataset_level.teacher == teacher]
        for comparator in ("NaiveKD","PersistenceGateKD"):
            d = (
                wt["regret__SoftSR"].to_numpy()
                - wt[f"regret__{comparator}"].to_numpy()
            )
            h4.append({
                "teacher": teacher,
                "comparator": comparator,
                "n_datasets": len(d),
                "n_favorable_datasets": int((d < 0).sum()),
                "fraction_favorable": float((d < 0).mean()),
            })
    pd.DataFrame(h4).to_csv(output / "H4_corrected.csv", index=False)

    # Secondary post-confirmatory harm/benefit analysis
    q["naive_outcome"] = np.select(
        [
            q["regret__NaiveKD"] > BENEFIT_THRESHOLD,
            q["regret__NaiveKD"] < -BENEFIT_THRESHOLD,
        ],
        ["harmful","beneficial"],
        default="neutral",
    )
    harm = q[q.naive_outcome == "harmful"]
    benefit = q[q.naive_outcome == "beneficial"]

    summary = pd.DataFrame([{
        "family":"quantum",
        "n_cells":len(q),
        "harmful_naive_n":len(harm),
        "harm_avoided_n":int((harm["regret__SoftSR"] <= BENEFIT_THRESHOLD).sum()),
        "harm_avoidance_rate":float(
            (harm["regret__SoftSR"] <= BENEFIT_THRESHOLD).mean()
        ),
        "false_acceptance_n":int(
            (
                (harm["gate__SoftSR"] > 0)
                & (harm["regret__SoftSR"] > BENEFIT_THRESHOLD)
            ).sum()
        ),
        "beneficial_naive_n":len(benefit),
        "benefit_retained_n":int(
            (benefit["regret__SoftSR"] < -BENEFIT_THRESHOLD).sum()
        ),
        "benefit_retention_rate":float(
            (benefit["regret__SoftSR"] < -BENEFIT_THRESHOLD).mean()
        ),
        "missed_benefit_n":int(
            (benefit["regret__SoftSR"] >= -BENEFIT_THRESHOLD).sum()
        ),
        "active_gate_n":int((q["gate__SoftSR"] > 0).sum()),
        "intermediate_gate_n":int(
            ((q["gate__SoftSR"] > 0) & (q["gate__SoftSR"] < 1)).sum()
        ),
    }])
    summary.to_csv(
        output / "secondary_harm_benefit_quantum_overall.csv", index=False
    )

    failures = q[
        (q.naive_outcome == "harmful")
        & (q["regret__SoftSR"] > BENEFIT_THRESHOLD)
    ]
    failures.to_csv(output / "secondary_false_acceptance_cases.csv", index=False)

    attenuation = q[
        (q["gate__SoftSR"] > 0)
        & (q["gate__SoftSR"] < 1)
    ]
    attenuation.to_csv(
        output / "secondary_intermediate_attenuation_cases.csv", index=False
    )

    generate_extended_outputs(df, wide, dataset_level, output)

    print("Corrected analysis complete.")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.results, args.output)
