#!/usr/bin/env python3
"""Linear export of the canonical notebook.

The notebook in notebooks/ is the authoritative executable artifact.
This file is provided for code review and text search.
"""
try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)


# %% [markdown] cell 0
# 
# # Quantum-to-CeNN Selective Knowledge Distillation
# ## Reproducible confirmatory benchmark — V3 with sealed final holdout
# 
# This notebook is the paper-facing reproducibility artifact for the final confirmatory study.
# 
# ### Critical evaluation policy
# 
# The chronological data are split into four disjoint target partitions:
# 
# 1. **Train** — model fitting only.
# 2. **Calibration** — gate construction only.
# 3. **Development holdout** — SMOKE and PROFILE diagnostics only.
# 4. **Final test holdout** — accessed only when `RUN_MODE == "FULL"`.
# 
# The final test arrays are **not materialized** during SMOKE or PROFILE. The FULL branch is the first code path that is allowed to construct and evaluate the final-test windows.
# 
# ### Frozen scientific invariants
# 
# - chronological split and train-only scaling;
# - deterministic temporal grids whose exact indices are archived;
# - leakage-free seasonal persistence;
# - moving-block bootstrap for overlapping calibration windows;
# - same student initialization and minibatch schedule within paired conditions;
# - exact reuse of Standalone for `g=0` and NaiveKD for `g=1`;
# - seeds are stochastic replications, not independent inferential units;
# - dataset-level confirmatory inference;
# - H1 restricted to the pre-specified quantum-teacher family;
# - RFF and ESN retained as secondary classical controls;
# - H2 split into H2a non-inferiority and H2b benefit retention;
# - all caches and outputs namespaced by a fingerprint that includes configuration, code identity, dataset hashes and preprocessing/statistical protocol versions.
# 
# > SMOKE and PROFILE results are diagnostic only and must never be merged into the FULL confirmatory result file.

# %% cell 1
# 1. Environment, provenance and deterministic execution

import os
import sys
import json
import math
import time
import copy
import random
import hashlib
import platform
import socket
import subprocess
import shutil
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMPORTANT: bump this string whenever any experiment-defining code changes.
PIPELINE_CODE_VERSION = "PAPER_T4_5H_V3_FINAL_HOLDOUT_2026-07-20"
PREPROCESSING_VERSION = "CHRONO_SORT_DEDUP_GRID_PERSISTENCE_DEV_FINAL_V3"
STATISTICAL_PROTOCOL_VERSION = "SEALED_FINAL_TEST_DATASET_LEVEL_H1_H2A_H2B_V3"

ROOT = Path(".")
DATA_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "results_confirmatory"
CACHE_DIR = ROOT / "cache_confirmatory"
MANIFEST_DIR = RESULTS_DIR / "manifests"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

for p in (DATA_DIR, RESULTS_DIR, CACHE_DIR, MANIFEST_DIR, ANALYSIS_DIR):
    p.mkdir(parents=True, exist_ok=True)

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def stable_seed(*parts) -> int:
    token = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:8], 16)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None

def environment_manifest():
    return {
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "git_commit": git_commit(),
        "pipeline_code_version": PIPELINE_CODE_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "statistical_protocol_version": STATISTICAL_PROTOCOL_VERSION,
    }

ENVIRONMENT = environment_manifest()

with (MANIFEST_DIR / "environment_manifest.json").open("w", encoding="utf-8") as f:
    json.dump(ENVIRONMENT, f, indent=2)

try:
    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    (MANIFEST_DIR / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
except Exception as exc:
    (MANIFEST_DIR / "pip_freeze.txt").write_text(
        f"pip freeze unavailable: {exc}\n", encoding="utf-8"
    )

seed_everything(0)

print(json.dumps(ENVIRONMENT, indent=2))
print("DEVICE:", DEVICE)

# %% cell 2

# 2. Frozen compute-bounded confirmatory configuration

@dataclass(frozen=True)
class LossWeights:
    pred: float = 1.0
    distill: float = 0.35
    obs: float = 0.05
    spec: float = 0.05
    acf: float = 0.05
    smooth: float = 0.01
    stab: float = 0.01

@dataclass(frozen=True)
class ConfirmatoryConfig:
    profile_name: str = "PAPER_T4_5H_V3_FINAL_HOLDOUT"

    # Forecast geometry
    lookback: int = 96
    horizon: int = 24

    # Four chronological target partitions.
    train_fraction: float = 0.70
    calibration_fraction: float = 0.15
    development_fraction: float = 0.075
    # final_test_fraction is the remainder = 0.075

    # Deterministic compute bounding
    max_train_windows: int = 12000
    max_cal_windows: int = 4000
    max_development_windows: int = 4000
    max_final_test_windows: int = 4000

    # Student
    cenn_steps: int = 4
    hidden_dim: int = 128
    emul_dim: int = 32
    epochs: int = 12
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    rho: float = 12.0
    acf_lag: int = 8

    # Stochastic replications: never independent inferential units
    seeds: Tuple[int, ...] = (11, 23, 37, 51, 79)

    # Gate calibration
    bootstrap_gate_samples: int = 1000
    gate_ci: float = 0.95
    soft_reject_advantage: float = -0.02
    soft_accept_advantage: float = 0.05
    hard_min_advantage: float = 0.00

    # Confirmatory inference
    delta_ni: float = 0.02
    bootstrap_stat_samples: int = 5000

    # H2b: meaningful NaiveKD benefit threshold
    benefit_min_relative: float = 0.01

    # Quantum teachers
    n_qubits: int = 6
    q_layers: int = 3
    qkrr_max_train: int = 400
    teacher_ridge: float = 1e-3
    vqc_epochs: int = 5

    # Reproducibility / runtime
    include_classical_controls: bool = True
    cache_enabled: bool = True
    wallclock_guard_minutes: int = 285
    invariance_tolerance: float = 1e-6

    weights: LossWeights = field(default_factory=LossWeights)

CFG = ConfirmatoryConfig()

_fraction_sum = (
    CFG.train_fraction
    + CFG.calibration_fraction
    + CFG.development_fraction
)
if not (0 < _fraction_sum < 1):
    raise ValueError("Train + calibration + development fractions must be < 1.")

FINAL_TEST_FRACTION = 1.0 - _fraction_sum

CFG_DICT = asdict(CFG)
CFG_DICT["final_test_fraction"] = FINAL_TEST_FRACTION
CFG_JSON = json.dumps(CFG_DICT, sort_keys=True, separators=(",", ":"))
CFG_SHA256 = hashlib.sha256(CFG_JSON.encode("utf-8")).hexdigest()

CODE_IDENTITY_PAYLOAD = "|".join([
    PIPELINE_CODE_VERSION,
    PREPROCESSING_VERSION,
    STATISTICAL_PROTOCOL_VERSION,
    ENVIRONMENT.get("git_commit") or "NO_GIT_COMMIT",
])
CODE_FINGERPRINT = hashlib.sha256(
    CODE_IDENTITY_PAYLOAD.encode("utf-8")
).hexdigest()

CONFIG_RECORD = {
    "sha256": CFG_SHA256,
    "code_fingerprint": CODE_FINGERPRINT,
    "created_utc": utc_now(),
    "config": CFG_DICT,
    "environment": ENVIRONMENT,
}

with (MANIFEST_DIR / "confirmatory_config.json").open("w", encoding="utf-8") as f:
    json.dump(CONFIG_RECORD, f, indent=2)

print("PROFILE:", CFG.profile_name)
print("CONFIG SHA-256:", CFG_SHA256)
print("CODE FINGERPRINT:", CODE_FINGERPRINT)
print("FINAL TEST FRACTION:", FINAL_TEST_FRACTION)

# %% [markdown] cell 3
# 
# ### Dataset provenance, chronological canonicalization and sealed final holdout
# 
# For timestamped datasets, preprocessing parses timestamps, removes unparseable timestamps, sorts ascending, and deterministically aggregates duplicated timestamps before any split.
# 
# The target timeline is divided chronologically into:
# 
# - 70% Train
# - 15% Calibration
# - 7.5% Development holdout
# - 7.5% Final test holdout
# 
# SMOKE and PROFILE may construct only Train, Calibration and Development arrays. The Final test target arrays remain unmaterialized until `RUN_MODE == "FULL"`.
# 
# The raw file SHA-256, processed numeric matrix SHA-256, run fingerprint, split boundaries and exact deterministic window-start indices are archived.

# %% cell 4
# 3. Dataset registry, download manifest, raw hashes and run identity

import io
import gzip
import zipfile
import urllib.request

DATASETS = {
    "ETTh1": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "compression": None,
        "target": "OT",
        "seasonality": 24,
        "read_kwargs": {},
        "timestamp_candidates": ("date",),
        "dayfirst": False,
    },
    "ETTm2": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
        "compression": None,
        "target": "OT",
        "seasonality": 96,
        "read_kwargs": {},
        "timestamp_candidates": ("date",),
        "dayfirst": False,
    },
    "Energy": {
        "url": "https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/PJME_hourly.csv",
        "compression": None,
        "target": "PJME_MW",
        "seasonality": 24,
        "read_kwargs": {},
        "timestamp_candidates": ("Datetime", "datetime", "date"),
        "dayfirst": False,
    },
    "Exchange": {
        "url": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz",
        "compression": "gz",
        "target": None,
        "seasonality": 5,
        "read_kwargs": {"header": None},
        "timestamp_candidates": (),
        "dayfirst": False,
    },
    "Jena": {
        "url": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
        "compression": "zip",
        "target": "T (degC)",
        "seasonality": 144,
        "read_kwargs": {},
        "timestamp_candidates": ("Date Time", "datetime", "date"),
        "dayfirst": True,
    },
    "AAPL": {
        "url": "https://raw.githubusercontent.com/matplotlib/sample_data/master/aapl.csv",
        "compression": None,
        "target": "Close",
        "seasonality": 5,
        "read_kwargs": {},
        "timestamp_candidates": ("Date", "date"),
        "dayfirst": False,
    },
}

def fetch_bytes(url, compression):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=120).read()

    if compression == "gz":
        return gzip.decompress(raw)

    if compression == "zip":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not members:
                raise RuntimeError(f"No CSV in archive: {url}")
            return zf.read(members[0])

    return raw

def download_datasets(force=False):
    rows = []
    print(f"Preparing {len(DATASETS)} datasets...")

    for name, spec in DATASETS.items():
        path = DATA_DIR / f"{name}.csv"

        if force or not path.exists() or path.stat().st_size < 128:
            print(f"  Downloading {name} ...", flush=True)
            path.write_bytes(fetch_bytes(spec["url"], spec["compression"]))
        else:
            print(f"  Using cached {name}: {path}", flush=True)

        rows.append({
            "dataset": name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "url": spec["url"],
        })

    manifest = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    manifest.to_csv(MANIFEST_DIR / "dataset_raw_manifest.csv", index=False)
    return manifest

dataset_manifest = download_datasets(force=False)

DATASET_FINGERPRINT_PAYLOAD = "|".join(
    f"{r.dataset}:{r.sha256}"
    for r in dataset_manifest.itertuples(index=False)
)
DATASET_FINGERPRINT = hashlib.sha256(
    DATASET_FINGERPRINT_PAYLOAD.encode("utf-8")
).hexdigest()

RUN_FINGERPRINT_PAYLOAD = "|".join([
    CFG_SHA256,
    CODE_FINGERPRINT,
    DATASET_FINGERPRINT,
    PREPROCESSING_VERSION,
    STATISTICAL_PROTOCOL_VERSION,
])
RUN_FINGERPRINT = hashlib.sha256(
    RUN_FINGERPRINT_PAYLOAD.encode("utf-8")
).hexdigest()
RUN_TAG = RUN_FINGERPRINT[:12]

RUN_IDENTITY = {
    "created_utc": utc_now(),
    "profile": CFG.profile_name,
    "config_sha256": CFG_SHA256,
    "code_fingerprint": CODE_FINGERPRINT,
    "dataset_fingerprint": DATASET_FINGERPRINT,
    "preprocessing_version": PREPROCESSING_VERSION,
    "statistical_protocol_version": STATISTICAL_PROTOCOL_VERSION,
    "run_fingerprint": RUN_FINGERPRINT,
    "run_tag": RUN_TAG,
}

with (MANIFEST_DIR / f"run_identity_{RUN_TAG}.json").open("w", encoding="utf-8") as f:
    json.dump(RUN_IDENTITY, f, indent=2)

print("RUN TAG:", RUN_TAG)
print("RUN FINGERPRINT:", RUN_FINGERPRINT)
display(dataset_manifest)

# %% cell 5

# 4. Strict chronological preprocessing, deterministic temporal grids,
#    true multi-step windows and sealed final test

@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

@dataclass
class DataBundle:
    name: str

    X_train: torch.Tensor
    Y_train: torch.Tensor

    X_cal: torch.Tensor
    Y_cal: torch.Tensor

    X_dev: torch.Tensor
    Y_dev: torch.Tensor

    # These remain None unless RUN_MODE == "FULL".
    X_final: Optional[torch.Tensor]
    Y_final: Optional[torch.Tensor]

    y_train_raw_series: np.ndarray

    y_cal_raw: np.ndarray
    y_dev_raw: np.ndarray
    y_final_raw: Optional[np.ndarray]

    y_cal_persistence_raw: np.ndarray
    y_dev_persistence_raw: np.ndarray
    y_final_persistence_raw: Optional[np.ndarray]

    feature_scaler: Standardizer
    target_scaler: Standardizer
    target_column: str
    seasonality: int

    # train_end, cal_end, dev_end, final_end
    split_indices: Tuple[int, int, int, int]

    train_starts: np.ndarray
    cal_starts: np.ndarray
    dev_starts: np.ndarray
    final_starts: np.ndarray

    final_test_materialized: bool
    preprocessing_metadata: Dict[str, Any]

    def inverse_y(self, y_scaled):
        return self.target_scaler.inverse(y_scaled)


def ensure_final_test_access_allowed():
    if globals().get("RUN_MODE", None) != "FULL":
        raise RuntimeError(
            "FINAL TEST IS SEALED. "
            "It may only be materialized when RUN_MODE == 'FULL'."
        )


def deterministic_temporal_grid(starts: np.ndarray, cap: Optional[int]) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int64)

    if cap is None or len(starts) <= cap:
        return starts

    pos = np.round(np.linspace(0, len(starts) - 1, cap)).astype(np.int64)
    pos = np.unique(pos)

    if len(pos) < cap:
        missing = np.setdiff1d(np.arange(len(starts)), pos, assume_unique=False)
        pos = np.sort(np.concatenate([pos, missing[:cap - len(pos)]]))

    return starts[pos[:cap]]


def canonicalize_dataset(dataset_name):
    spec = DATASETS[dataset_name]
    path = DATA_DIR / f"{dataset_name}.csv"

    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run download_datasets() first.")

    df = pd.read_csv(path, **spec["read_kwargs"])
    raw_rows = len(df)

    timestamp_col = next(
        (c for c in spec["timestamp_candidates"] if c in df.columns),
        None,
    )

    duplicate_timestamps = 0
    dropped_bad_timestamps = 0

    if timestamp_col is not None:
        ts = pd.to_datetime(
            df[timestamp_col],
            errors="coerce",
            dayfirst=spec.get("dayfirst", False),
        )

        valid = ts.notna()
        dropped_bad_timestamps = int((~valid).sum())
        df = df.loc[valid].copy()
        df["_timestamp"] = ts.loc[valid].values
        df = df.sort_values("_timestamp", kind="mergesort")

        duplicate_timestamps = int(df["_timestamp"].duplicated(keep=False).sum())

        for c in df.columns:
            if c in (timestamp_col, "_timestamp"):
                continue
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")

        numeric_cols = [
            c for c in df.columns
            if c not in (timestamp_col, "_timestamp")
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        if duplicate_timestamps:
            df = (
                df[["_timestamp"] + numeric_cols]
                .groupby("_timestamp", as_index=False, sort=True)
                .mean(numeric_only=True)
            )

        numeric = df.select_dtypes(include=[np.number]).copy()

    else:
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                coerced = pd.to_numeric(df[c], errors="coerce")
                if coerced.notna().sum() >= 0.95 * len(df):
                    df[c] = coerced
        numeric = df.select_dtypes(include=[np.number]).copy()

    numeric = numeric.interpolate(method="linear").bfill().ffill()

    if numeric.empty:
        raise ValueError(f"{dataset_name}: no usable numeric columns.")

    target = spec["target"] if spec["target"] is not None else numeric.columns[-1]
    if target not in numeric.columns:
        raise ValueError(
            f"{dataset_name}: target {target!r} absent. "
            f"Numeric columns={list(numeric.columns)}"
        )

    processed_hash = sha256_bytes(
        np.ascontiguousarray(numeric.to_numpy(dtype=np.float32)).tobytes()
    )

    meta = {
        "dataset": dataset_name,
        "raw_rows": int(raw_rows),
        "processed_rows": int(len(numeric)),
        "timestamp_column": timestamp_col,
        "dropped_bad_timestamps": dropped_bad_timestamps,
        "duplicate_timestamp_rows_detected": duplicate_timestamps,
        "processed_numeric_sha256": processed_hash,
        "target": str(target),
    }

    return numeric, target, meta


def seasonal_persistence(target_raw, target_start, horizon, seasonality):
    """Leakage-free recursive seasonal-naive forecast."""
    m = max(1, int(seasonality))
    cycle_start = max(0, target_start - m)

    observed_cycle = np.asarray(
        target_raw[cycle_start:target_start],
        dtype=np.float32,
    )

    if len(observed_cycle) == 0:
        raise ValueError("Persistence forecast has no observed history.")

    return np.asarray(
        [observed_cycle[h % len(observed_cycle)] for h in range(horizon)],
        dtype=np.float32,
    )


def make_windows(
    features,
    target_scaled,
    target_raw,
    starts,
    lookback,
    horizon,
    seasonality,
):
    X = np.stack([
        features[s:s + lookback]
        for s in starts
    ]).astype(np.float32)

    Y = np.stack([
        target_scaled[s + lookback:s + lookback + horizon]
        for s in starts
    ]).astype(np.float32)

    YR = np.stack([
        target_raw[s + lookback:s + lookback + horizon]
        for s in starts
    ]).astype(np.float32)

    P = np.stack([
        seasonal_persistence(
            target_raw,
            s + lookback,
            horizon,
            seasonality,
        )
        for s in starts
    ]).astype(np.float32)

    return X, Y, YR, P


def build_data_bundle(
    dataset_name,
    cfg=CFG,
    include_final_test=False,
):
    if include_final_test:
        ensure_final_test_access_allowed()

    numeric, target_col, prep_meta = canonicalize_dataset(dataset_name)

    features_raw = numeric.to_numpy(dtype=np.float32)
    target_raw = numeric[target_col].to_numpy(dtype=np.float32)
    n = len(numeric)

    train_end = int(n * cfg.train_fraction)
    cal_end = int(
        n * (
            cfg.train_fraction
            + cfg.calibration_fraction
        )
    )
    dev_end = int(
        n * (
            cfg.train_fraction
            + cfg.calibration_fraction
            + cfg.development_fraction
        )
    )
    final_end = n

    if not (
        cfg.lookback + cfg.horizon
        < train_end
        < cal_end
        < dev_end
        < final_end
    ):
        raise ValueError(
            f"{dataset_name}: invalid train/cal/dev/final split geometry."
        )

    # Fit scaling only on raw training points.
    fmean = features_raw[:train_end].mean(axis=0, keepdims=True)
    fstd = features_raw[:train_end].std(axis=0, keepdims=True)
    fstd = np.where(fstd < 1e-8, 1.0, fstd)

    ymean = np.array(
        [[target_raw[:train_end].mean()]],
        dtype=np.float32,
    )
    ystd = np.array(
        [[target_raw[:train_end].std()]],
        dtype=np.float32,
    )
    ystd = np.where(ystd < 1e-8, 1.0, ystd)

    fs = Standardizer(fmean, fstd)
    ys = Standardizer(ymean, ystd)

    features = fs.transform(features_raw)
    target_scaled = ys.transform(
        target_raw.reshape(-1, 1)
    ).reshape(-1)

    # Every forecast horizon lies fully inside its target partition.
    # Input lookback may use chronologically earlier data.
    train_all = np.arange(
        0,
        train_end - cfg.lookback - cfg.horizon + 1,
        dtype=np.int64,
    )

    cal_all = np.arange(
        train_end - cfg.lookback,
        cal_end - cfg.lookback - cfg.horizon + 1,
        dtype=np.int64,
    )

    dev_all = np.arange(
        cal_end - cfg.lookback,
        dev_end - cfg.lookback - cfg.horizon + 1,
        dtype=np.int64,
    )

    final_all = np.arange(
        dev_end - cfg.lookback,
        final_end - cfg.lookback - cfg.horizon + 1,
        dtype=np.int64,
    )

    train_starts = deterministic_temporal_grid(
        train_all,
        cfg.max_train_windows,
    )
    cal_starts = deterministic_temporal_grid(
        cal_all,
        cfg.max_cal_windows,
    )
    dev_starts = deterministic_temporal_grid(
        dev_all,
        cfg.max_development_windows,
    )
    final_starts = deterministic_temporal_grid(
        final_all,
        cfg.max_final_test_windows,
    )

    if min(
        len(train_starts),
        len(cal_starts),
        len(dev_starts),
        len(final_starts),
    ) == 0:
        raise ValueError(
            f"{dataset_name}: at least one chronological partition has no windows."
        )

    Xtr, Ytr, _, _ = make_windows(
        features,
        target_scaled,
        target_raw,
        train_starts,
        cfg.lookback,
        cfg.horizon,
        DATASETS[dataset_name]["seasonality"],
    )

    Xcal, Ycal, YRcal, Pcal = make_windows(
        features,
        target_scaled,
        target_raw,
        cal_starts,
        cfg.lookback,
        cfg.horizon,
        DATASETS[dataset_name]["seasonality"],
    )

    Xdev, Ydev, YRdev, Pdev = make_windows(
        features,
        target_scaled,
        target_raw,
        dev_starts,
        cfg.lookback,
        cfg.horizon,
        DATASETS[dataset_name]["seasonality"],
    )

    # Final test arrays are not even constructed outside FULL.
    Xfinal = None
    Yfinal = None
    YRfinal = None
    Pfinal = None

    if include_final_test:
        Xfinal_np, Yfinal_np, YRfinal, Pfinal = make_windows(
            features,
            target_scaled,
            target_raw,
            final_starts,
            cfg.lookback,
            cfg.horizon,
            DATASETS[dataset_name]["seasonality"],
        )
        Xfinal = torch.from_numpy(Xfinal_np)
        Yfinal = torch.from_numpy(Yfinal_np)

    index_path = (
        MANIFEST_DIR
        / f"window_starts_{dataset_name}_{RUN_TAG}.npz"
    )

    # Indices may be archived before final evaluation; targets are not.
    np.savez_compressed(
        index_path,
        train_starts=train_starts,
        cal_starts=cal_starts,
        development_starts=dev_starts,
        final_test_starts=final_starts,
    )

    prep_meta.update({
        "train_end": int(train_end),
        "cal_end": int(cal_end),
        "development_end": int(dev_end),
        "final_test_end": int(final_end),
        "all_train_windows": int(len(train_all)),
        "all_cal_windows": int(len(cal_all)),
        "all_development_windows": int(len(dev_all)),
        "all_final_test_windows": int(len(final_all)),
        "selected_train_windows": int(len(train_starts)),
        "selected_cal_windows": int(len(cal_starts)),
        "selected_development_windows": int(len(dev_starts)),
        "selected_final_test_windows": int(len(final_starts)),
        "final_test_materialized": bool(include_final_test),
        "window_indices_file": str(index_path),
        "window_indices_sha256": sha256_file(index_path),
        "config_sha256": CFG_SHA256,
        "run_fingerprint": RUN_FINGERPRINT,
        "run_tag": RUN_TAG,
        "preprocessing_version": PREPROCESSING_VERSION,
    })

    manifest_suffix = (
        "full"
        if include_final_test
        else "development_only"
    )

    with (
        MANIFEST_DIR
        / f"preprocessing_{dataset_name}_{RUN_TAG}_{manifest_suffix}.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(prep_meta, f, indent=2)

    bundle = DataBundle(
        name=dataset_name,
        X_train=torch.from_numpy(Xtr),
        Y_train=torch.from_numpy(Ytr),
        X_cal=torch.from_numpy(Xcal),
        Y_cal=torch.from_numpy(Ycal),
        X_dev=torch.from_numpy(Xdev),
        Y_dev=torch.from_numpy(Ydev),
        X_final=Xfinal,
        Y_final=Yfinal,
        y_train_raw_series=target_raw[:train_end].copy(),
        y_cal_raw=YRcal,
        y_dev_raw=YRdev,
        y_final_raw=YRfinal,
        y_cal_persistence_raw=Pcal,
        y_dev_persistence_raw=Pdev,
        y_final_persistence_raw=Pfinal,
        feature_scaler=fs,
        target_scaler=ys,
        target_column=str(target_col),
        seasonality=DATASETS[dataset_name]["seasonality"],
        split_indices=(
            train_end,
            cal_end,
            dev_end,
            final_end,
        ),
        train_starts=train_starts,
        cal_starts=cal_starts,
        dev_starts=dev_starts,
        final_starts=final_starts,
        final_test_materialized=bool(include_final_test),
        preprocessing_metadata=prep_meta,
    )

    assert bundle.Y_train.shape[1] == cfg.horizon > 1
    assert bundle.Y_cal.shape[1] == cfg.horizon
    assert bundle.Y_dev.shape[1] == cfg.horizon

    if include_final_test:
        assert bundle.Y_final is not None
        assert bundle.Y_final.shape[1] == cfg.horizon
        assert bundle.y_final_raw is not None

    return bundle


def evaluation_view(bundle, evaluation_split):
    if evaluation_split == "development":
        return (
            bundle.X_dev,
            bundle.y_dev_raw,
            bundle.y_dev_persistence_raw,
            bundle.dev_starts,
        )

    if evaluation_split == "final_test":
        ensure_final_test_access_allowed()

        if (
            not bundle.final_test_materialized
            or bundle.X_final is None
            or bundle.y_final_raw is None
            or bundle.y_final_persistence_raw is None
        ):
            raise RuntimeError(
                "Final test requested but was not materialized by the FULL path."
            )

        return (
            bundle.X_final,
            bundle.y_final_raw,
            bundle.y_final_persistence_raw,
            bundle.final_starts,
        )

    raise ValueError(
        "evaluation_split must be 'development' or 'final_test'."
    )

# %% cell 6
# 5. Raw-scale metrics and seasonal MASE

def mase_denominator(train_raw, seasonality):
    y=np.asarray(train_raw,dtype=np.float64).reshape(-1)
    m=max(1,int(seasonality))
    if len(y)<=m: m=1
    return float(max(np.mean(np.abs(y[m:]-y[:-m])),1e-12))

def metric_dict(y_true_raw,y_pred_raw,train_raw,seasonality):
    yt=np.asarray(y_true_raw,dtype=np.float64)
    yp=np.asarray(y_pred_raw,dtype=np.float64)
    e=yp-yt
    mse=float(np.mean(e**2)); mae=float(np.mean(np.abs(e)))
    return {
        'MSE':mse,
        'RMSE':float(np.sqrt(mse)),
        'MAE':mae,
        'MASE':float(mae/mase_denominator(train_raw,seasonality)),
        'sMAPE':float(200*np.mean(np.abs(e)/(np.abs(yt)+np.abs(yp)+1e-8)))
    }

def per_sample_scaled_absolute_error(y_true_raw,y_pred_raw,train_raw,seasonality):
    d=mase_denominator(train_raw,seasonality)
    return np.mean(np.abs(y_pred_raw-y_true_raw),axis=1)/d

# %% cell 7
# 6. Exact batched statevector simulator
C64=torch.complex64
class StatevectorSim:
    def __init__(self,n_qubits,device=DEVICE):
        self.n=n_qubits; self.dim=2**n_qubits; self.device=device
        idx=torch.arange(self.dim,device=device)
        self.z_signs=torch.stack([1.-2.*((idx>>(self.n-1-q))&1).float() for q in range(self.n)])
        self.x_flips=torch.stack([idx^(1<<(self.n-1-q)) for q in range(self.n)])
        self.cnot_perms=[]
        for c in range(self.n):
            t=(c+1)%self.n
            cbit=(idx>>(self.n-1-c))&1
            self.cnot_perms.append(torch.where(cbit.bool(),idx^(1<<(self.n-1-t)),idx))
    def init_state(self,b):
        psi=torch.zeros(b,self.dim,dtype=C64,device=self.device); psi[:,0]=1.; return psi
    def _gate(self,psi,q,a,b,c,d):
        B=psi.shape[0]; L,R=2**q,2**(self.n-1-q)
        psi=psi.view(B,L,2,R); s0,s1=psi[:,:,0,:],psi[:,:,1,:]
        a,b,c,d=(v.view(B,1,1) for v in (a,b,c,d))
        return torch.stack((a*s0+b*s1,c*s0+d*s1),dim=2).reshape(B,self.dim)
    def ry(self,psi,q,theta):
        h=theta.to(psi.real.dtype)/2; co=torch.cos(h).to(C64); si=torch.sin(h).to(C64)
        return self._gate(psi,q,co,-si,si,co)
    def rz(self,psi,q,theta):
        h=theta.to(psi.real.dtype)/2; em=torch.exp(-1j*h.to(C64)); ep=torch.exp(1j*h.to(C64)); z=torch.zeros_like(em)
        return self._gate(psi,q,em,z,z,ep)
    def cnot_ring(self,psi):
        for perm in self.cnot_perms: psi=psi[:,perm]
        return psi
    def observables(self,psi):
        prob=psi.real.square()+psi.imag.square()
        z=prob@self.z_signs.T
        zz=torch.stack([prob@(self.z_signs[q]*self.z_signs[(q+1)%self.n]) for q in range(self.n)],dim=1)
        x=torch.stack([(psi.conj()*psi[:,self.x_flips[q]]).real.sum(1) for q in range(self.n)],dim=1)
        return torch.cat([z,zz,x],dim=1)

def apply_layers(sim,angles,rz):
    psi=sim.init_state(angles.shape[0]); obs=[]
    for l in range(angles.shape[1]):
        for q in range(sim.n):
            psi=sim.ry(psi,q,angles[:,l,q]); psi=sim.rz(psi,q,rz[l,q].expand(len(psi)))
        psi=sim.cnot_ring(psi); obs.append(sim.observables(psi))
    return psi,torch.cat(obs,dim=1)

# %% cell 8
# 7. Quantum teachers plus matched classical control teachers

class QuantumWindowEncoder:
    def __init__(self, in_dim, layers, nq, seed, device=DEVICE):
        rng = np.random.default_rng(seed)
        P = rng.normal(
            0,
            1 / np.sqrt(in_dim),
            size=(in_dim, layers * nq),
        ).astype(np.float32)
        self.P = torch.tensor(P, device=device)
        self.layers = layers
        self.nq = nq
        self.device = device

    def encode(self, x):
        xt = x.to(self.device)
        flat = xt.reshape(len(xt), -1)
        return (
            np.pi * torch.tanh(flat @ self.P)
        ).view(len(xt), self.layers, self.nq)

class QELMTeacher:
    name = "QELM"
    family = "quantum"

    def __init__(self, seed, nq=6, layers=3, ridge=1e-3, batch=512):
        self.seed = seed
        self.nq = nq
        self.layers = layers
        self.ridge = ridge
        self.batch = batch
        self.sim = StatevectorSim(nq)

    def _features(self, x):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), self.batch):
                _, obs = apply_layers(
                    self.sim,
                    self.encoder.encode(x[i:i + self.batch]),
                    self.rz,
                )
                out.append(obs.cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def fit(self, x, y):
        rng = np.random.default_rng(self.seed)
        self.encoder = QuantumWindowEncoder(
            x.shape[1] * x.shape[2],
            self.layers,
            self.nq,
            self.seed,
        )
        self.rz = torch.tensor(
            rng.uniform(0, 2 * np.pi, size=(self.layers, self.nq)),
            dtype=torch.float32,
            device=DEVICE,
        )
        H = self._features(x)
        Y = y.numpy().astype(np.float32)
        A = H.T @ H + self.ridge * np.eye(H.shape[1], dtype=np.float32)
        self.beta = np.linalg.solve(A, H.T @ Y).astype(np.float32)
        self._fit_obs = H
        self._fit_pred = H @ self.beta
        return self

    def training_artifacts(self):
        return self._fit_pred, self._fit_obs

    def predict(self, x):
        return self._features(x) @ self.beta

class VQCTeacher:
    name = "VQC"
    family = "quantum"

    def __init__(self, seed, horizon, nq=6, layers=3, epochs=5, lr=3e-2, batch=256):
        self.seed = seed
        self.horizon = horizon
        self.nq = nq
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.sim = StatevectorSim(nq)

    def _features(self, x):
        _, obs = apply_layers(
            self.sim,
            self.encoder.encode(x),
            self.rz,
        )
        return obs

    def fit(self, x, y):
        seed_everything(self.seed)
        rng = np.random.default_rng(self.seed)

        self.encoder = QuantumWindowEncoder(
            x.shape[1] * x.shape[2],
            self.layers,
            self.nq,
            self.seed,
        )
        self.rz = torch.tensor(
            rng.uniform(0, 2 * np.pi, size=(self.layers, self.nq)),
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True,
        )
        self.head = nn.Linear(
            3 * self.nq * self.layers,
            self.horizon,
        ).to(DEVICE)

        opt = torch.optim.Adam(
            [self.rz] + list(self.head.parameters()),
            lr=self.lr,
        )
        yd = y.to(DEVICE)
        gen = torch.Generator().manual_seed(self.seed + 1000)

        for epoch in range(self.epochs):
            order = torch.randperm(len(x), generator=gen)
            for i in range(0, len(x), self.batch):
                idx = order[i:i + self.batch]
                pred = self.head(self._features(x[idx]))
                loss = F.mse_loss(pred, yd[idx.to(DEVICE)])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.head.parameters()) + [self.rz],
                    1.0,
                )
                opt.step()

        self.rz = self.rz.detach()
        self.head.eval()
        return self

    def artifacts(self, x):
        preds, obs = [], []
        with torch.no_grad():
            for i in range(0, len(x), self.batch):
                f = self._features(x[i:i + self.batch])
                obs.append(f.cpu().numpy())
                preds.append(self.head(f).cpu().numpy())
        return (
            np.concatenate(preds).astype(np.float32),
            np.concatenate(obs).astype(np.float32),
        )

    def training_artifacts(self):
        return None

    def predict(self, x):
        return self.artifacts(x)[0]

class QRCTeacher:
    """
    Sequential temporal injection with persistent state.
    """
    name = "QRC"
    family = "quantum"

    def __init__(self, seed, horizon, nq=6, ridge=1e-3, leak=0.2, batch=256):
        self.seed = seed
        self.horizon = horizon
        self.nq = nq
        self.ridge = ridge
        self.leak = leak
        self.batch = batch
        self.sim = StatevectorSim(nq)

    def _features(self, x):
        out = []
        with torch.no_grad():
            for start in range(0, len(x), self.batch):
                xb = x[start:start + self.batch].to(DEVICE)
                psi = self.sim.init_state(len(xb))
                pooled = torch.zeros(
                    len(xb),
                    3 * self.nq,
                    device=DEVICE,
                )

                for t in range(xb.shape[1]):
                    ang = np.pi * torch.tanh(xb[:, t, :] @ self.P)

                    for q in range(self.nq):
                        psi = self.sim.ry(psi, q, ang[:, q])
                        psi = self.sim.rz(
                            psi,
                            q,
                            self.rz[q].expand(len(xb)),
                        )

                    psi = self.sim.cnot_ring(psi)
                    obs = self.sim.observables(psi)
                    pooled = self.leak * pooled + (1 - self.leak) * obs

                out.append(pooled.cpu().numpy())

        return np.concatenate(out).astype(np.float32)

    def fit(self, x, y):
        rng = np.random.default_rng(self.seed)
        feature_dim = x.shape[2]

        self.P = torch.tensor(
            rng.normal(
                0,
                1 / np.sqrt(feature_dim),
                size=(feature_dim, self.nq),
            ).astype(np.float32),
            device=DEVICE,
        )
        self.rz = torch.tensor(
            rng.uniform(0, 2 * np.pi, size=(self.nq,)),
            dtype=torch.float32,
            device=DEVICE,
        )

        H = self._features(x)
        Y = y.numpy().astype(np.float32)
        A = H.T @ H + self.ridge * np.eye(H.shape[1], dtype=np.float32)
        self.beta = np.linalg.solve(A, H.T @ Y).astype(np.float32)
        self._fit_obs = H
        self._fit_pred = H @ self.beta
        return self

    def training_artifacts(self):
        return self._fit_pred, self._fit_obs

    def predict(self, x):
        return self._features(x) @ self.beta

class QKRRTeacher:
    name = "QKRR"
    family = "quantum"

    def __init__(self, seed, horizon, nq=6, layers=2, ridge=1e-2, max_train=400, batch=256):
        self.seed = seed
        self.horizon = horizon
        self.nq = nq
        self.layers = layers
        self.ridge = ridge
        self.max_train = max_train
        self.batch = batch
        self.sim = StatevectorSim(nq)

    def _states(self, x):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), self.batch):
                ang = self.encoder.encode(x[i:i + self.batch])
                psi = self.sim.init_state(len(ang))

                for layer in range(self.layers):
                    for q in range(self.nq):
                        psi = self.sim.ry(psi, q, ang[:, layer, q])
                        psi = self.sim.rz(
                            psi,
                            q,
                            self.rz[layer, q].expand(len(ang)),
                        )
                    psi = self.sim.cnot_ring(psi)

                out.append(psi)

        return torch.cat(out)

    def fit(self, x, y):
        rng = np.random.default_rng(self.seed)

        self.encoder = QuantumWindowEncoder(
            x.shape[1] * x.shape[2],
            self.layers,
            self.nq,
            self.seed,
        )
        self.rz = torch.tensor(
            rng.uniform(
                0,
                2 * np.pi,
                size=(self.layers, self.nq),
            ),
            dtype=torch.float32,
            device=DEVICE,
        )

        if len(x) > self.max_train:
            idx = np.round(
                np.linspace(0, len(x) - 1, self.max_train)
            ).astype(int)
            xr = x[idx]
            yr = y[idx]
        else:
            xr, yr = x, y

        self.ref = self._states(xr)
        K = (
            torch.abs(self.ref @ self.ref.T.conj())
            .square()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        self.alpha = np.linalg.solve(
            K + self.ridge * np.eye(len(K), dtype=np.float32),
            yr.numpy().astype(np.float32),
        )
        return self

    def predict(self, x):
        states = self._states(x)
        K = (
            torch.abs(states @ self.ref.T.conj())
            .square()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        return K @ self.alpha

    def training_artifacts(self):
        return None

    def observables(self, x):
        states = self._states(x)
        return self.sim.observables(states).cpu().numpy().astype(np.float32)

class RFFTeacher:
    """
    Classical random-feature ridge control.
    Feature dimension matches QELM observable dimension: 3 * nq * q_layers.
    """
    name = "RFF"
    family = "classical_control"

    def __init__(self, seed, horizon, feature_dim, ridge=1e-3):
        self.seed = seed
        self.horizon = horizon
        self.feature_dim = feature_dim
        self.ridge = ridge

    def _features(self, x):
        flat = x.numpy().reshape(len(x), -1).astype(np.float32)
        return np.sqrt(2.0 / self.feature_dim) * np.cos(
            flat @ self.W + self.b
        )

    def fit(self, x, y):
        rng = np.random.default_rng(self.seed)
        in_dim = x.shape[1] * x.shape[2]
        self.W = rng.normal(
            0,
            1 / np.sqrt(in_dim),
            size=(in_dim, self.feature_dim),
        ).astype(np.float32)
        self.b = rng.uniform(
            0,
            2 * np.pi,
            size=(self.feature_dim,),
        ).astype(np.float32)

        H = self._features(x).astype(np.float32)
        Y = y.numpy().astype(np.float32)
        A = H.T @ H + self.ridge * np.eye(H.shape[1], dtype=np.float32)
        self.beta = np.linalg.solve(A, H.T @ Y).astype(np.float32)
        self._fit_obs = H
        self._fit_pred = H @ self.beta
        return self

    def training_artifacts(self):
        return self._fit_pred, self._fit_obs

    def predict(self, x):
        return self._features(x) @ self.beta

class ESNTeacher:
    """
    Classical sequential reservoir control with the same readout feature dimension
    as QRC (3 * nq).
    """
    name = "ESN"
    family = "classical_control"

    def __init__(self, seed, horizon, reservoir_dim, ridge=1e-3, leak=0.2, batch=512):
        self.seed = seed
        self.horizon = horizon
        self.reservoir_dim = reservoir_dim
        self.ridge = ridge
        self.leak = leak
        self.batch = batch

    def _features(self, x):
        out = []

        with torch.no_grad():
            for start in range(0, len(x), self.batch):
                xb = x[start:start + self.batch].to(DEVICE)
                state = torch.zeros(
                    len(xb),
                    self.reservoir_dim,
                    device=DEVICE,
                )

                for t in range(xb.shape[1]):
                    candidate = torch.tanh(
                        xb[:, t, :] @ self.Win
                        + state @ self.Wres
                        + self.bias
                    )
                    state = (
                        self.leak * state
                        + (1 - self.leak) * candidate
                    )

                out.append(state.cpu().numpy())

        return np.concatenate(out).astype(np.float32)

    def fit(self, x, y):
        rng = np.random.default_rng(self.seed)
        fdim = x.shape[2]

        Win = rng.normal(
            0,
            1 / np.sqrt(fdim),
            size=(fdim, self.reservoir_dim),
        ).astype(np.float32)

        Wres = rng.normal(
            0,
            1 / np.sqrt(self.reservoir_dim),
            size=(self.reservoir_dim, self.reservoir_dim),
        ).astype(np.float32)

        # Deterministic spectral-radius normalization.
        eig = np.linalg.eigvals(Wres.astype(np.float64))
        radius = max(float(np.max(np.abs(eig))), 1e-8)
        Wres = (0.9 / radius * Wres).astype(np.float32)

        self.Win = torch.tensor(Win, device=DEVICE)
        self.Wres = torch.tensor(Wres, device=DEVICE)
        self.bias = torch.tensor(
            rng.normal(0, 0.01, size=(self.reservoir_dim,)).astype(np.float32),
            device=DEVICE,
        )

        H = self._features(x)
        Y = y.numpy().astype(np.float32)
        A = H.T @ H + self.ridge * np.eye(H.shape[1], dtype=np.float32)
        self.beta = np.linalg.solve(A, H.T @ Y).astype(np.float32)
        self._fit_obs = H
        self._fit_pred = H @ self.beta
        return self

    def training_artifacts(self):
        return self._fit_pred, self._fit_obs

    def predict(self, x):
        return self._features(x) @ self.beta

QUANTUM_TEACHERS = ("QELM", "VQC", "QRC", "QKRR")
CONTROL_TEACHERS = ("RFF", "ESN")
TEACHER_NAMES = (
    QUANTUM_TEACHERS + CONTROL_TEACHERS
    if CFG.include_classical_controls
    else QUANTUM_TEACHERS
)

def teacher_family(name):
    return "quantum" if name in QUANTUM_TEACHERS else "classical_control"

def make_teacher(name, seed, horizon):
    if name == "QELM":
        return QELMTeacher(
            seed,
            nq=CFG.n_qubits,
            layers=CFG.q_layers,
            ridge=CFG.teacher_ridge,
        )
    if name == "VQC":
        return VQCTeacher(
            seed,
            horizon,
            nq=CFG.n_qubits,
            layers=CFG.q_layers,
            epochs=CFG.vqc_epochs,
        )
    if name == "QRC":
        return QRCTeacher(
            seed,
            horizon,
            nq=CFG.n_qubits,
            ridge=CFG.teacher_ridge,
        )
    if name == "QKRR":
        return QKRRTeacher(
            seed,
            horizon,
            nq=CFG.n_qubits,
            max_train=CFG.qkrr_max_train,
        )
    if name == "RFF":
        return RFFTeacher(
            seed,
            horizon,
            feature_dim=3 * CFG.n_qubits * CFG.q_layers,
            ridge=CFG.teacher_ridge,
        )
    if name == "ESN":
        return ESNTeacher(
            seed,
            horizon,
            reservoir_dim=3 * CFG.n_qubits,
            ridge=CFG.teacher_ridge,
        )
    raise ValueError(name)

# %% cell 9
# 8. Fixed-dimensional teacher-observable projection

class ObservableProjector:
    def __init__(self, in_dim, out_dim, seed):
        rng = np.random.default_rng(seed)
        self.P = rng.normal(
            0,
            1 / np.sqrt(max(1, in_dim)),
            size=(in_dim, out_dim),
        ).astype(np.float32)

    def transform(self, obs):
        return np.asarray(obs, dtype=np.float32) @ self.P

# %% cell 10
# 9. Temporal CeNN student: lattice aligned with the lookback axis
class TemporalCeNNCore(nn.Module):
    def __init__(self,n_features,lookback,steps=4):
        super().__init__(); self.lookback=lookback; self.steps=steps
        self.input_proj=nn.Linear(n_features,1)
        self.kernel=nn.Parameter(torch.tensor([0.15,0.70,0.15],dtype=torch.float32).view(1,1,3))
        self.self_gain=nn.Parameter(torch.tensor(0.65,dtype=torch.float32))
        self.bias=nn.Parameter(torch.zeros(lookback))
    def forward(self,x,return_states=False):
        s=torch.tanh(self.input_proj(x).squeeze(-1)); states=[s]
        for _ in range(self.steps):
            z=F.pad(s.unsqueeze(1),(1,1),mode='replicate'); neighbor=F.conv1d(z,self.kernel).squeeze(1)
            s=torch.tanh(self.self_gain*s+neighbor+self.bias); states.append(s)
        return (s,states) if return_states else s

class CeNNForecaster(nn.Module):
    def __init__(self,n_features,lookback,horizon,hidden_dim,emul_dim,steps):
        super().__init__(); self.core=TemporalCeNNCore(n_features,lookback,steps)
        self.forecast_head=nn.Sequential(nn.Linear(lookback,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,horizon))
        self.emulation_head=nn.Sequential(nn.Linear(lookback,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,emul_dim))
    def forward(self,x,return_aux=False):
        s,states=self.core(x,True); y=self.forecast_head(s); e=self.emulation_head(s)
        return (y,e,states) if return_aux else y

def make_student(bundle):
    return CeNNForecaster(bundle.X_train.shape[2],bundle.X_train.shape[1],bundle.Y_train.shape[1],CFG.hidden_dim,CFG.emul_dim,CFG.cenn_steps)

# %% cell 11
# 10. Composite objective: all teacher-derived terms share the same gate

def autocorrelation(seq,max_lag):
    B,H=seq.shape; L=min(max_lag,H-1)
    if L<=0: return seq.new_zeros((B,1))
    c=seq-seq.mean(dim=-1,keepdim=True); d=c.square().sum(dim=-1).clamp_min(1e-8)
    return torch.stack([(c[:,lag:]*c[:,:-lag]).sum(dim=-1)/d for lag in range(1,L+1)],dim=-1)

def composite_loss(pred,emul,states,y,tp,to,gate):
    w=CFG.weights; total=w.pred*F.mse_loss(pred,y)
    if gate>0:
        total=total+gate*w.distill*F.mse_loss(pred,tp)
        total=total+gate*w.obs*F.mse_loss(emul,to)
        if pred.shape[-1]>1:
            total=total+gate*w.spec*F.mse_loss(torch.abs(torch.fft.rfft(pred,dim=-1)),torch.abs(torch.fft.rfft(tp,dim=-1)))
            total=total+gate*w.acf*F.mse_loss(autocorrelation(pred,CFG.acf_lag),autocorrelation(tp,CFG.acf_lag))
    smooth=torch.stack([(states[i]-states[i-1]).square().mean() for i in range(1,len(states))]).mean()
    stab=torch.stack([torch.clamp(torch.linalg.norm(s,dim=-1)-CFG.rho,min=0).square().mean() for s in states]).mean()
    return total+w.smooth*smooth+w.stab*stab

# %% cell 12
# 11. Calibration-only gates with moving-block bootstrap uncertainty

@dataclass
class GateDecision:
    gate: float
    action: str
    mean_advantage: float
    ci_low: float
    ci_high: float


def infer_temporal_block_length(starts, n_samples, lookback, horizon):
    """Translate overlap in raw time indices into a conservative block length.

    Sequential calibration-window scores are not IID. A block spans approximately
    one full input+forecast dependence range, adjusted for the median spacing of
    the deterministic temporal grid.
    """
    starts = np.asarray(starts, dtype=np.int64)
    n_samples = int(n_samples)
    if n_samples <= 3:
        return 1

    if len(starts) > 1:
        stride = max(1, int(round(float(np.median(np.diff(starts))))))
    else:
        stride = 1

    dependence_span = max(1, int(lookback + horizon - 1))
    raw_block = max(2, int(math.ceil(dependence_span / stride)))

    # Keep at least four effective blocks when possible.
    upper = max(2, n_samples // 4)
    return int(min(raw_block, upper))


def moving_block_bootstrap_mean_ci(values, confidence, n_boot, seed, block_length):
    """Efficient moving-block bootstrap for the mean of an ordered sequence.

    Uses precomputed block sums rather than materializing concatenated bootstrap
    samples, preserving the moving-block bootstrap while keeping FULL runtime bounded.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    n = len(x)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty sequence.")

    L = int(max(1, min(block_length, n)))
    rng = np.random.default_rng(seed)

    if L == 1:
        idx = rng.integers(0, n, size=(n_boot, n))
        means = x[idx].mean(axis=1)
    else:
        csum = np.concatenate(([0.0], np.cumsum(x)))
        block_sums = csum[L:] - csum[:-L]
        n_full = n // L
        remainder = n % L

        full_starts = rng.integers(
            0,
            len(block_sums),
            size=(n_boot, n_full),
        )
        totals = block_sums[full_starts].sum(axis=1)

        if remainder:
            partial_sums = csum[remainder:] - csum[:-remainder]
            partial_starts = rng.integers(
                0,
                len(partial_sums),
                size=n_boot,
            )
            totals = totals + partial_sums[partial_starts]

        means = totals / n

    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def teacher_advantage_samples(y_true_raw, ref_pred_raw, teacher_pred_raw, train_raw, seasonality):
    ref_err = per_sample_scaled_absolute_error(
        y_true_raw, ref_pred_raw, train_raw, seasonality
    )
    teacher_err = per_sample_scaled_absolute_error(
        y_true_raw, teacher_pred_raw, train_raw, seasonality
    )
    # Positive means teacher is better than the reference.
    return (ref_err - teacher_err) / np.maximum(ref_err, 1e-8)


def gate_from_advantage(samples, seed, mode, block_length):
    mean_adv = float(np.mean(samples))
    low, high = moving_block_bootstrap_mean_ci(
        samples,
        CFG.gate_ci,
        CFG.bootstrap_gate_samples,
        seed,
        block_length,
    )

    if mode == "hard":
        g = 1.0 if low >= CFG.hard_min_advantage else 0.0
    elif mode == "soft":
        lo, hi = CFG.soft_reject_advantage, CFG.soft_accept_advantage
        g = float(np.clip((low - lo) / max(hi - lo, 1e-12), 0.0, 1.0))
    else:
        raise ValueError(mode)

    action = "reject" if g == 0.0 else ("accept" if g == 1.0 else "attenuate")
    return GateDecision(g, action, mean_adv, low, high)

# %% cell 13
# 12. Paired student training, prediction and deterministic condition caches

def make_epoch_orders(n, epochs, seed):
    gen = torch.Generator().manual_seed(seed + 424242)
    return [
        torch.randperm(n, generator=gen)
        for _ in range(epochs)
    ]

def initial_student_state(bundle, seed):
    seed_everything(seed)
    return copy.deepcopy(make_student(bundle).state_dict())

def gpu_status():
    if not torch.cuda.is_available():
        return "CPU"
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return (
        f"{torch.cuda.get_device_name(0)} | "
        f"allocated={alloc:.2f} GB | reserved={reserved:.2f} GB"
    )

def train_student_paired(
    bundle,
    init_state,
    orders,
    tp_train=None,
    to_train=None,
    gate=0.0,
    label="student",
    verbose=False,
):
    model = make_student(bundle).to(DEVICE)
    model.load_state_dict(copy.deepcopy(init_state), strict=True)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
    )

    n = len(bundle.X_train)

    if tp_train is None:
        tp_train = np.zeros((n, CFG.horizon), dtype=np.float32)
    if to_train is None:
        to_train = np.zeros((n, CFG.emul_dim), dtype=np.float32)

    TP = torch.from_numpy(np.asarray(tp_train, dtype=np.float32))
    TO = torch.from_numpy(np.asarray(to_train, dtype=np.float32))

    if verbose:
        print(
            f"    [{label}] start | n={n} | epochs={len(orders)} "
            f"| gate={gate:.6f} | {gpu_status()}",
            flush=True,
        )

    t0 = time.time()
    model.train()

    for epoch, order in enumerate(orders, start=1):
        epoch_loss = 0.0
        seen = 0

        for start in range(0, n, CFG.batch_size):
            idx = order[start:start + CFG.batch_size]

            xb = bundle.X_train[idx].to(
                DEVICE,
                non_blocking=True,
            )
            yb = bundle.Y_train[idx].to(
                DEVICE,
                non_blocking=True,
            )

            pred, emul, states = model(xb, True)

            loss = composite_loss(
                pred,
                emul,
                states,
                yb,
                TP[idx].to(DEVICE, non_blocking=True),
                TO[idx].to(DEVICE, non_blocking=True),
                float(gate),
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss: {label}, epoch={epoch}"
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                CFG.grad_clip,
            )
            opt.step()

            bs = len(idx)
            epoch_loss += float(loss.detach().cpu()) * bs
            seen += bs

        if verbose:
            print(
                f"    [{label}] epoch {epoch:02d}/{len(orders)} "
                f"loss={epoch_loss / max(seen, 1):.6f}",
                flush=True,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if verbose:
        print(
            f"    [{label}] DONE in {(time.time() - t0) / 60:.2f} min",
            flush=True,
        )

    model.eval()
    return model

def predict_scaled(model, X, batch=1024):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(
                model(
                    X[i:i + batch].to(
                        DEVICE,
                        non_blocking=True,
                    )
                ).cpu().numpy()
            )
    return np.concatenate(out).astype(np.float32)

def evaluate_student(model, X, Y_raw, bundle):
    pred_raw = bundle.inverse_y(predict_scaled(model, X))
    return (
        metric_dict(
            Y_raw,
            pred_raw,
            bundle.y_train_raw_series,
            bundle.seasonality,
        ),
        pred_raw,
    )

def cache_path(kind, *parts):
    safe = [
        str(p).replace("/", "_").replace(" ", "_")
        for p in parts
    ]
    folder = CACHE_DIR / RUN_TAG / kind
    folder.mkdir(parents=True, exist_ok=True)
    return folder / ("__".join(safe) + ".npz")

def save_npz_atomic(path: Path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)

def load_npz_dict(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}

def condition_gate_key(gate):
    return f"{float(gate):.8f}"

# %% cell 14

# 13. Protocol tests

def assert_state_dict_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if not torch.equal(a[k], b[k]):
            raise AssertionError(
                f"Initialization mismatch: {k}"
            )


def protocol_static_tests(bundle):
    assert bundle.Y_train.shape[1] == CFG.horizon > 1
    assert bundle.Y_cal.shape[1] == CFG.horizon
    assert bundle.Y_dev.shape[1] == CFG.horizon

    assert len(bundle.train_starts) <= CFG.max_train_windows
    assert len(bundle.cal_starts) <= CFG.max_cal_windows
    assert len(bundle.dev_starts) <= CFG.max_development_windows
    assert len(bundle.final_starts) <= CFG.max_final_test_windows

    train_end, cal_end, dev_end, final_end = bundle.split_indices

    # Entire forecast target horizon remains within its target partition.
    assert np.max(
        bundle.train_starts
        + CFG.lookback
        + CFG.horizon
    ) <= train_end

    assert np.min(
        bundle.cal_starts
        + CFG.lookback
    ) >= train_end

    assert np.max(
        bundle.cal_starts
        + CFG.lookback
        + CFG.horizon
    ) <= cal_end

    assert np.min(
        bundle.dev_starts
        + CFG.lookback
    ) >= cal_end

    assert np.max(
        bundle.dev_starts
        + CFG.lookback
        + CFG.horizon
    ) <= dev_end

    assert np.min(
        bundle.final_starts
        + CFG.lookback
    ) >= dev_end

    assert np.max(
        bundle.final_starts
        + CFG.lookback
        + CFG.horizon
    ) <= final_end

    # Outside FULL, final-test target arrays must not exist.
    assert bundle.final_test_materialized is False
    assert bundle.X_final is None
    assert bundle.Y_final is None
    assert bundle.y_final_raw is None
    assert bundle.y_final_persistence_raw is None

    # Direct final-test access is blocked outside FULL.
    try:
        ensure_final_test_access_allowed()
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "Final-test access lock failed outside FULL."
        )

    # Same seed -> identical initialization and minibatch schedule.
    s1 = initial_student_state(bundle, 123)
    s2 = initial_student_state(bundle, 123)
    assert_state_dict_equal(s1, s2)

    o1 = make_epoch_orders(100, 3, 123)
    o2 = make_epoch_orders(100, 3, 123)
    assert all(
        torch.equal(a, b)
        for a, b in zip(o1, o2)
    )

    # Gate monotonicity under same bootstrap geometry.
    bad = np.full(100, -0.1)
    mid = np.full(100, 0.01)
    good = np.full(100, 0.1)
    L = 10

    assert (
        gate_from_advantage(
            bad, 1, "soft", L
        ).gate
        <= gate_from_advantage(
            mid, 1, "soft", L
        ).gate
        <= gate_from_advantage(
            good, 1, "soft", L
        ).gate
    )

    # Persistence leakage audit.
    synthetic = np.arange(
        20,
        dtype=np.float32,
    )
    target_start = 10

    pred = seasonal_persistence(
        synthetic,
        target_start,
        horizon=12,
        seasonality=5,
    )

    expected_cycle = synthetic[5:10]
    expected = np.asarray([
        expected_cycle[h % 5]
        for h in range(12)
    ], dtype=np.float32)

    assert np.array_equal(
        pred,
        expected,
    )
    assert float(np.max(pred)) < target_start

    assert RUN_TAG and len(RUN_TAG) == 12
    assert RUN_FINGERPRINT != CFG_SHA256

    print("Static protocol tests: PASS")
    print("Leakage-free seasonal persistence test: PASS")
    print("Run fingerprint namespace test: PASS")
    print("Final-test materialization lock: PASS")


def dynamic_g0_invariance_audit(bundle, seed=11):
    """
    Explicit retraining audit performed in SMOKE on DEVELOPMENT only.
    The final test remains sealed.
    """
    init = initial_student_state(
        bundle,
        seed,
    )

    orders = make_epoch_orders(
        len(bundle.X_train),
        CFG.epochs,
        seed,
    )

    standalone = train_student_paired(
        bundle,
        init,
        orders,
        gate=0.0,
        label="audit_standalone",
        verbose=False,
    )

    duplicate_g0 = train_student_paired(
        bundle,
        init,
        orders,
        tp_train=np.ones(
            (
                len(bundle.X_train),
                CFG.horizon,
            ),
            dtype=np.float32,
        ),
        to_train=np.ones(
            (
                len(bundle.X_train),
                CFG.emul_dim,
            ),
            dtype=np.float32,
        ),
        gate=0.0,
        label="audit_g0",
        verbose=False,
    )

    p1 = predict_scaled(
        standalone,
        bundle.X_dev,
    )
    p2 = predict_scaled(
        duplicate_g0,
        bundle.X_dev,
    )

    max_abs = float(
        np.max(
            np.abs(
                p1 - p2
            )
        )
    )

    print(
        f"Dynamic g=0 DEVELOPMENT invariance max |Δ| = {max_abs:.3e}"
    )

    if max_abs > CFG.invariance_tolerance:
        raise AssertionError(
            f"g=0 invariance failed: {max_abs:.3e}"
        )

    return max_abs


# This protocol bundle is DEVELOPMENT-ONLY by construction.
_protocol_bundle = build_data_bundle(
    "ETTh1",
    include_final_test=False,
)

print(
    "Protocol ETTh1 shapes:",
    "train", tuple(_protocol_bundle.X_train.shape),
    "cal", tuple(_protocol_bundle.X_cal.shape),
    "development", tuple(_protocol_bundle.X_dev.shape),
    "final_materialized", _protocol_bundle.final_test_materialized,
)

protocol_static_tests(
    _protocol_bundle
)

# %% cell 15

# 14. Efficient paired benchmark block with split-aware caches.
# Development diagnostics and Final-test confirmation are never mixed.

METHODS = (
    "Standalone",
    "NaiveKD",
    "PersistenceGateKD",
    "HardRejectSR",
    "SoftSR",
)


def get_or_train_standalone(
    bundle,
    dataset_name,
    seed,
    evaluation_split,
    verbose=False,
):
    path = cache_path(
        "standalone",
        evaluation_split,
        dataset_name,
        seed,
    )

    if CFG.cache_enabled and path.exists():
        d = load_npz_dict(path)

        return {
            "cal_pred_raw": d["cal_pred_raw"],
            "eval_pred_raw": d["eval_pred_raw"],
        }

    X_eval, y_eval_raw, _, _ = evaluation_view(
        bundle,
        evaluation_split,
    )

    init = initial_student_state(
        bundle,
        seed,
    )

    orders = make_epoch_orders(
        len(bundle.X_train),
        CFG.epochs,
        seed,
    )

    model = train_student_paired(
        bundle,
        init,
        orders,
        gate=0.0,
        label=(
            f"Standalone/"
            f"{dataset_name}/"
            f"{seed}/"
            f"{evaluation_split}"
        ),
        verbose=verbose,
    )

    _, cal_pred_raw = evaluate_student(
        model,
        bundle.X_cal,
        bundle.y_cal_raw,
        bundle,
    )

    _, eval_pred_raw = evaluate_student(
        model,
        X_eval,
        y_eval_raw,
        bundle,
    )

    save_npz_atomic(
        path,
        cal_pred_raw=cal_pred_raw,
        eval_pred_raw=eval_pred_raw,
    )

    return {
        "cal_pred_raw": cal_pred_raw,
        "eval_pred_raw": eval_pred_raw,
    }


def get_or_train_teacher_artifacts(
    bundle,
    dataset_name,
    seed,
    teacher_name,
    evaluation_split,
    verbose=False,
):
    path = cache_path(
        "teacher",
        evaluation_split,
        dataset_name,
        seed,
        teacher_name,
    )

    if CFG.cache_enabled and path.exists():
        return load_npz_dict(path)

    X_eval, _, _, _ = evaluation_view(
        bundle,
        evaluation_split,
    )

    teacher_seed = stable_seed(
        CFG_SHA256,
        dataset_name,
        teacher_name,
        seed,
    )

    if verbose:
        print(
            f"  Fitting {teacher_name} "
            f"({teacher_family(teacher_name)}) "
            f"for {evaluation_split}...",
            flush=True,
        )

    teacher = make_teacher(
        teacher_name,
        teacher_seed,
        CFG.horizon,
    )

    t0 = time.time()

    teacher.fit(
        bundle.X_train,
        bundle.Y_train,
    )

    train_artifacts = (
        teacher.training_artifacts()
    )

    if train_artifacts is None:
        if teacher_name == "VQC":
            tp_train, raw_obs = teacher.artifacts(
                bundle.X_train
            )

        elif teacher_name == "QKRR":
            tp_train = teacher.predict(
                bundle.X_train
            )
            raw_obs = teacher.observables(
                bundle.X_train
            )

        else:
            raise RuntimeError(
                f"No training-artifact path for {teacher_name}"
            )

    else:
        tp_train, raw_obs = train_artifacts

    projector = ObservableProjector(
        raw_obs.shape[1],
        CFG.emul_dim,
        stable_seed(
            CFG_SHA256,
            dataset_name,
            teacher_name,
            seed,
            "observable_projector",
        ),
    )

    to_train = projector.transform(
        raw_obs
    )

    tp_cal = teacher.predict(
        bundle.X_cal
    )

    tp_eval = teacher.predict(
        X_eval
    )

    if not all(
        np.isfinite(a).all()
        for a in (
            tp_train,
            to_train,
            tp_cal,
            tp_eval,
        )
    ):
        raise FloatingPointError(
            f"{dataset_name}/"
            f"{seed}/"
            f"{teacher_name}/"
            f"{evaluation_split}: "
            "non-finite teacher artifacts."
        )

    artifact = {
        "tp_train": np.asarray(
            tp_train,
            np.float32,
        ),
        "to_train": np.asarray(
            to_train,
            np.float32,
        ),
        "tp_cal": np.asarray(
            tp_cal,
            np.float32,
        ),
        "tp_eval": np.asarray(
            tp_eval,
            np.float32,
        ),
        "fit_seconds": np.asarray(
            [time.time() - t0],
            dtype=np.float64,
        ),
    }

    save_npz_atomic(
        path,
        **artifact,
    )

    return artifact


def get_or_train_gate_condition(
    bundle,
    dataset_name,
    seed,
    teacher_name,
    evaluation_split,
    gate,
    tp_train,
    to_train,
    standalone_eval_raw,
    naive_eval_raw=None,
    verbose=False,
):
    gate = float(gate)

    if gate == 0.0:
        return (
            standalone_eval_raw,
            "reused_standalone",
        )

    if (
        gate == 1.0
        and naive_eval_raw is not None
    ):
        return (
            naive_eval_raw,
            "reused_naive",
        )

    path = cache_path(
        "student_condition",
        evaluation_split,
        dataset_name,
        seed,
        teacher_name,
        condition_gate_key(gate),
    )

    if CFG.cache_enabled and path.exists():
        d = load_npz_dict(path)
        return (
            d["eval_pred_raw"],
            "cache",
        )

    X_eval, y_eval_raw, _, _ = evaluation_view(
        bundle,
        evaluation_split,
    )

    init = initial_student_state(
        bundle,
        seed,
    )

    orders = make_epoch_orders(
        len(bundle.X_train),
        CFG.epochs,
        seed,
    )

    model = train_student_paired(
        bundle,
        init,
        orders,
        tp_train=tp_train,
        to_train=to_train,
        gate=gate,
        label=(
            f"{dataset_name}/"
            f"{seed}/"
            f"{teacher_name}/"
            f"{evaluation_split}/"
            f"g={gate:.4f}"
        ),
        verbose=verbose,
    )

    _, eval_pred_raw = evaluate_student(
        model,
        X_eval,
        y_eval_raw,
        bundle,
    )

    save_npz_atomic(
        path,
        eval_pred_raw=eval_pred_raw,
    )

    return (
        eval_pred_raw,
        "trained",
    )


def benchmark_teacher_block(
    bundle,
    dataset_name,
    seed,
    teacher_name,
    standalone_artifacts,
    evaluation_split,
    verbose=False,
):
    X_eval, y_eval_raw, _, _ = evaluation_view(
        bundle,
        evaluation_split,
    )

    ns_cal_raw = standalone_artifacts[
        "cal_pred_raw"
    ]

    ns_eval_raw = standalone_artifacts[
        "eval_pred_raw"
    ]

    ns_cal_metrics = metric_dict(
        bundle.y_cal_raw,
        ns_cal_raw,
        bundle.y_train_raw_series,
        bundle.seasonality,
    )

    ns_eval_metrics = metric_dict(
        y_eval_raw,
        ns_eval_raw,
        bundle.y_train_raw_series,
        bundle.seasonality,
    )

    teacher_art = get_or_train_teacher_artifacts(
        bundle,
        dataset_name,
        seed,
        teacher_name,
        evaluation_split,
        verbose=verbose,
    )

    tp_train = teacher_art[
        "tp_train"
    ]

    to_train = teacher_art[
        "to_train"
    ]

    tp_cal_raw = bundle.inverse_y(
        teacher_art[
            "tp_cal"
        ]
    )

    tp_eval_raw = bundle.inverse_y(
        teacher_art[
            "tp_eval"
        ]
    )

    teacher_cal_metrics = metric_dict(
        bundle.y_cal_raw,
        tp_cal_raw,
        bundle.y_train_raw_series,
        bundle.seasonality,
    )

    teacher_eval_metrics = metric_dict(
        y_eval_raw,
        tp_eval_raw,
        bundle.y_train_raw_series,
        bundle.seasonality,
    )

    # Gates are based ONLY on calibration.
    sr = teacher_advantage_samples(
        bundle.y_cal_raw,
        ns_cal_raw,
        tp_cal_raw,
        bundle.y_train_raw_series,
        bundle.seasonality,
    )

    pr = teacher_advantage_samples(
        bundle.y_cal_raw,
        bundle.y_cal_persistence_raw,
        tp_cal_raw,
        bundle.y_train_raw_series,
        bundle.seasonality,
    )

    gate_block_length = infer_temporal_block_length(
        bundle.cal_starts,
        len(sr),
        CFG.lookback,
        CFG.horizon,
    )

    decisions = {
        "NaiveKD": GateDecision(
            1.0,
            "accept",
            float(np.mean(sr)),
            np.nan,
            np.nan,
        ),

        "PersistenceGateKD": gate_from_advantage(
            pr,
            stable_seed(
                CFG_SHA256,
                dataset_name,
                teacher_name,
                seed,
                "persistence_gate",
            ),
            "soft",
            gate_block_length,
        ),

        "HardRejectSR": gate_from_advantage(
            sr,
            stable_seed(
                CFG_SHA256,
                dataset_name,
                teacher_name,
                seed,
                "hard_gate",
            ),
            "hard",
            gate_block_length,
        ),

        "SoftSR": gate_from_advantage(
            sr,
            stable_seed(
                CFG_SHA256,
                dataset_name,
                teacher_name,
                seed,
                "soft_gate",
            ),
            "soft",
            gate_block_length,
        ),
    }

    common = {
        "dataset": dataset_name,
        "seed": seed,
        "teacher": teacher_name,
        "teacher_family": teacher_family(
            teacher_name
        ),
        "evaluation_split": evaluation_split,
        "run_tag": RUN_TAG,
        "run_fingerprint": RUN_FINGERPRINT,
        "gate_block_length": gate_block_length,
        "cal_teacher_MASE": teacher_cal_metrics[
            "MASE"
        ],
        "cal_standalone_MASE": ns_cal_metrics[
            "MASE"
        ],
        "teacher_eval_MASE": teacher_eval_metrics[
            "MASE"
        ],
    }

    rows = [{
        **common,
        "method": "Standalone",
        "gate": 0.0,
        "gate_action": "standalone",
        "condition_source": "shared_standalone",
        **{
            f"eval_{k}": v
            for k, v in ns_eval_metrics.items()
        },
    }]

    # Train NaiveKD once.
    naive_eval_raw, naive_source = (
        get_or_train_gate_condition(
            bundle,
            dataset_name,
            seed,
            teacher_name,
            evaluation_split,
            1.0,
            tp_train,
            to_train,
            ns_eval_raw,
            naive_eval_raw=None,
            verbose=verbose,
        )
    )

    naive_metrics = metric_dict(
        y_eval_raw,
        naive_eval_raw,
        bundle.y_train_raw_series,
        bundle.seasonality,
    )

    naive_decision = decisions[
        "NaiveKD"
    ]

    rows.append({
        **common,
        "method": "NaiveKD",
        "gate": 1.0,
        "gate_action": "accept",
        "condition_source": naive_source,
        "gate_mean_advantage": naive_decision.mean_advantage,
        "gate_ci_low": np.nan,
        "gate_ci_high": np.nan,
        **{
            f"eval_{k}": v
            for k, v in naive_metrics.items()
        },
    })

    trained_gate_predictions = {
        0.0: ns_eval_raw,
        1.0: naive_eval_raw,
    }

    for method in (
        "PersistenceGateKD",
        "HardRejectSR",
        "SoftSR",
    ):
        decision = decisions[
            method
        ]

        g = float(
            decision.gate
        )

        if g in trained_gate_predictions:
            pred_raw = trained_gate_predictions[
                g
            ]

            source = (
                "reused_standalone"
                if g == 0.0
                else "reused_naive"
            )

        else:
            pred_raw, source = (
                get_or_train_gate_condition(
                    bundle,
                    dataset_name,
                    seed,
                    teacher_name,
                    evaluation_split,
                    g,
                    tp_train,
                    to_train,
                    ns_eval_raw,
                    naive_eval_raw=naive_eval_raw,
                    verbose=verbose,
                )
            )

            trained_gate_predictions[
                g
            ] = pred_raw

        metrics = metric_dict(
            y_eval_raw,
            pred_raw,
            bundle.y_train_raw_series,
            bundle.seasonality,
        )

        if g == 0.0:
            max_abs = float(
                np.max(
                    np.abs(
                        pred_raw
                        - ns_eval_raw
                    )
                )
            )

            if max_abs > CFG.invariance_tolerance:
                raise AssertionError(
                    f"g=0 reuse invariance failed: "
                    f"{max_abs:.3e}"
                )

        rows.append({
            **common,
            "method": method,
            "gate": g,
            "gate_action": decision.action,
            "condition_source": source,
            "gate_mean_advantage": decision.mean_advantage,
            "gate_ci_low": decision.ci_low,
            "gate_ci_high": decision.ci_high,
            **{
                f"eval_{k}": v
                for k, v in metrics.items()
            },
        })

    return rows

# %% cell 16

# 15. Resumable runner with a sealed Final-test FULL route

def completed_teacher_blocks(
    results_df,
    evaluation_split,
):
    if results_df.empty:
        return set()

    if "run_fingerprint" in results_df.columns:
        results_df = results_df[
            results_df[
                "run_fingerprint"
            ] == RUN_FINGERPRINT
        ]

    if "evaluation_split" in results_df.columns:
        results_df = results_df[
            results_df[
                "evaluation_split"
            ] == evaluation_split
        ]

    counts = (
        results_df
        .groupby(
            [
                "dataset",
                "seed",
                "teacher",
            ]
        )["method"]
        .nunique()
    )

    return set(
        counts[
            counts >= len(METHODS)
        ].index.tolist()
    )


def atomic_save_results(
    rows,
    output_csv,
):
    out = pd.DataFrame(
        rows
    )

    tmp = output_csv.with_suffix(
        ".tmp.csv"
    )

    out.to_csv(
        tmp,
        index=False,
    )

    os.replace(
        tmp,
        output_csv,
    )


def result_path(
    kind,
    evaluation_split,
):
    return (
        RESULTS_DIR
        / f"{kind}_{evaluation_split}_{RUN_TAG}.csv"
    )


def run_confirmatory(
    evaluation_split,
    datasets=tuple(DATASETS.keys()),
    teachers=TEACHER_NAMES,
    seeds=CFG.seeds,
    output_csv=None,
    verbose=False,
    wallclock_guard=True,
):
    if evaluation_split not in (
        "development",
        "final_test",
    ):
        raise ValueError(
            "evaluation_split must be "
            "'development' or 'final_test'."
        )

    if evaluation_split == "final_test":
        ensure_final_test_access_allowed()

    if output_csv is None:
        output_csv = result_path(
            "confirmatory_results",
            evaluation_split,
        )

    session_start = time.time()

    existing = (
        pd.read_csv(output_csv)
        if output_csv.exists()
        else pd.DataFrame()
    )

    if not existing.empty:
        if "run_fingerprint" not in existing.columns:
            raise RuntimeError(
                "Existing result file lacks run fingerprint."
            )

        foreign = existing[
            existing[
                "run_fingerprint"
            ] != RUN_FINGERPRINT
        ]

        if len(foreign):
            raise RuntimeError(
                "Result file contains a different run fingerprint."
            )

        if "evaluation_split" not in existing.columns:
            raise RuntimeError(
                "Existing result file lacks evaluation_split."
            )

        wrong_split = existing[
            existing[
                "evaluation_split"
            ] != evaluation_split
        ]

        if len(wrong_split):
            raise RuntimeError(
                "Refusing to mix Development and Final-test rows."
            )

    rows = (
        []
        if existing.empty
        else existing.to_dict(
            "records"
        )
    )

    completed = completed_teacher_blocks(
        existing,
        evaluation_split,
    )

    total = (
        len(datasets)
        * len(seeds)
        * len(teachers)
    )

    print(
        f"Runner profile={CFG.profile_name} | "
        f"run_tag={RUN_TAG} | "
        f"evaluation_split={evaluation_split} | "
        f"completed={len(completed)}/{total}",
        flush=True,
    )

    print(
        f"Device: {gpu_status()}",
        flush=True,
    )

    print(
        f"Run fingerprint: {RUN_FINGERPRINT}",
        flush=True,
    )

    if evaluation_split == "final_test":
        print(
            "FINAL TEST ACCESS: UNSEALED BY FULL MODE",
            flush=True,
        )
    else:
        print(
            "FINAL TEST ACCESS: SEALED",
            flush=True,
        )

    block_counter = 0

    for dataset_name in datasets:
        print(
            f"\n=== DATASET {dataset_name} ===",
            flush=True,
        )

        bundle = build_data_bundle(
            dataset_name,
            include_final_test=(
                evaluation_split
                == "final_test"
            ),
        )

        if (
            evaluation_split
            == "development"
            and bundle.final_test_materialized
        ):
            raise AssertionError(
                "Development run unexpectedly materialized final test."
            )

        for seed in seeds:
            standalone = get_or_train_standalone(
                bundle,
                dataset_name,
                seed,
                evaluation_split,
                verbose=verbose,
            )

            for teacher_name in teachers:
                block_counter += 1

                key = (
                    dataset_name,
                    seed,
                    teacher_name,
                )

                if key in completed:
                    print(
                        f"[{block_counter}/{total}] "
                        f"SKIP {key}",
                        flush=True,
                    )
                    continue

                elapsed_min = (
                    time.time()
                    - session_start
                ) / 60

                if (
                    wallclock_guard
                    and elapsed_min
                    >= CFG.wallclock_guard_minutes
                ):
                    atomic_save_results(
                        rows,
                        output_csv,
                    )

                    print(
                        "\nWALL-CLOCK GUARD reached. "
                        "Progress saved safely. "
                        "Rerun FULL to resume the same "
                        "run fingerprint and Final-test file.",
                        flush=True,
                    )

                    return pd.DataFrame(
                        rows
                    )

                print(
                    f"[{block_counter}/{total}] "
                    f"RUN {key}",
                    flush=True,
                )

                t0 = time.time()

                block_rows = (
                    benchmark_teacher_block(
                        bundle,
                        dataset_name,
                        seed,
                        teacher_name,
                        standalone,
                        evaluation_split,
                        verbose=verbose,
                    )
                )

                rows.extend(
                    block_rows
                )

                atomic_save_results(
                    rows,
                    output_csv,
                )

                print(
                    f"  saved in "
                    f"{(time.time() - t0) / 60:.2f} min "
                    f"| session="
                    f"{(time.time() - session_start) / 60:.1f} min",
                    flush=True,
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(
        "ALL REQUESTED BLOCKS COMPLETE.",
        flush=True,
    )

    return pd.DataFrame(
        rows
    )


# -------------------------------------------------------------------
# EXECUTION CONTROL
# -------------------------------------------------------------------
# NONE:
#   definitions only
#
# SMOKE:
#   ETTh1 + QELM + one seed, DEVELOPMENT only
#
# PROFILE:
#   ETTh1 + all teachers + one seed, DEVELOPMENT only
#
# FULL:
#   all datasets + all seeds + all teachers, FINAL TEST only
#
# The Final-test arrays cannot be materialized outside FULL.
RUN_MODE = "SMOKE"

print("=" * 72)
print("RUN_MODE:", RUN_MODE)
print("RUN_TAG:", RUN_TAG)
print("TEACHERS:", TEACHER_NAMES)
print("GPU:", gpu_status())
print("=" * 72)

if RUN_MODE == "SMOKE":
    smoke_bundle = build_data_bundle(
        "ETTh1",
        include_final_test=False,
    )

    dynamic_g0_invariance_audit(
        smoke_bundle,
        seed=CFG.seeds[0],
    )

    results_smoke = run_confirmatory(
        evaluation_split="development",
        datasets=("ETTh1",),
        teachers=("QELM",),
        seeds=(CFG.seeds[0],),
        output_csv=result_path(
            "smoke_results",
            "development",
        ),
        verbose=True,
        wallclock_guard=False,
    )

    if set(
        results_smoke[
            "evaluation_split"
        ].unique()
    ) != {"development"}:
        raise AssertionError(
            "SMOKE produced non-development rows."
        )

    display(
        results_smoke
    )

elif RUN_MODE == "PROFILE":
    profile_start = time.time()

    results_profile = run_confirmatory(
        evaluation_split="development",
        datasets=("ETTh1",),
        teachers=TEACHER_NAMES,
        seeds=(CFG.seeds[0],),
        output_csv=result_path(
            "profile_results",
            "development",
        ),
        verbose=False,
        wallclock_guard=False,
    )

    if set(
        results_profile[
            "evaluation_split"
        ].unique()
    ) != {"development"}:
        raise AssertionError(
            "PROFILE produced non-development rows."
        )

    profile_minutes = (
        time.time()
        - profile_start
    ) / 60

    print(
        f"PROFILE completed in "
        f"{profile_minutes:.2f} min.",
        flush=True,
    )

    print(
        "Final test remained sealed during PROFILE.",
        flush=True,
    )

    display(
        results_profile
    )

elif RUN_MODE == "FULL":
    # This is the ONLY branch that can materialize the final test.
    results = run_confirmatory(
        evaluation_split="final_test",
        output_csv=result_path(
            "confirmatory_results",
            "final_test",
        ),
        verbose=False,
        wallclock_guard=True,
    )

    if (
        not results.empty
        and set(
            results[
                "evaluation_split"
            ].unique()
        ) != {"final_test"}
    ):
        raise AssertionError(
            "FULL result file contains non-final-test rows."
        )

    print(
        f"Final-test rows currently available: "
        f"{len(results)}",
        flush=True,
    )

    display(
        results.tail()
    )

elif RUN_MODE == "NONE":
    print(
        "No experiment launched.",
        flush=True,
    )

else:
    raise ValueError(
        "RUN_MODE must be NONE, SMOKE, PROFILE or FULL."
    )

# %% [markdown] cell 17
# 
# ## Confirmatory statistics — FINAL TEST ONLY
# 
# The confirmatory analysis functions refuse any dataframe that contains Development rows.
# 
# The top-level inferential unit is the **dataset**. Seeds are repeated stochastic realizations and are averaged within dataset before confirmatory inference.
# 
# - **H1:** SoftSR relative regret versus NaiveKD and PersistenceGateKD, for the four quantum teachers only; one-sided paired Wilcoxon across datasets; Holm correction across the two comparisons within each pre-specified teacher family.
# - **H2a:** SoftSR non-inferiority to HardRejectSR on dataset-mean relative regret.
# - **H2b:** benefit retention relative to HardRejectSR on SoftSR-accepted cells with at least 1% NaiveKD benefit.
# - **H3:** non-inferiority of SoftSR to the better of Standalone and NaiveKD on accepted cells.
# - **H4:** directional generality across datasets for the quantum-teacher family.
# 
# RFF and ESN remain secondary classical controls and are not merged into the confirmatory H1–H4 family.
# 
# Only rows with `evaluation_split == "final_test"` may enter these analyses.

# %% cell 18
# 16. Dataset-level confirmatory analysis with corrected H1/H2 families

CONFIRMATORY_TEACHERS = QUANTUM_TEACHERS
SECONDARY_CONTROL_TEACHERS = CONTROL_TEACHERS


def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def bootstrap_dataset_mean_ci(dataset_values, n_boot, confidence, seed):
    x = np.asarray(dataset_values, dtype=np.float64)
    if len(x) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        stats[b] = x[idx].mean()

    alpha = (1 - confidence) / 2
    return {
        "mean": float(x.mean()),
        "ci_low": float(np.quantile(stats, alpha)),
        "ci_high": float(np.quantile(stats, 1 - alpha)),
    }


def prepare_wide(results):
    required = {
        "dataset", "seed", "teacher", "teacher_family",
        "method", "eval_MASE", "gate", "run_fingerprint", "evaluation_split",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")

    unique_runs = set(results["run_fingerprint"].dropna().astype(str))
    if unique_runs != {RUN_FINGERPRINT}:
        raise ValueError(
            f"Analysis expected run {RUN_FINGERPRINT}, found {sorted(unique_runs)}"
        )

    unique_splits = set(results["evaluation_split"].dropna().astype(str))
    if unique_splits != {"final_test"}:
        raise ValueError(
            "Confirmatory analysis accepts FINAL TEST rows only. "
            f"Found evaluation splits: {sorted(unique_splits)}"
        )

    wide = results.pivot_table(
        index=["dataset", "seed", "teacher", "teacher_family"],
        columns="method",
        values=["eval_MASE", "gate"],
        aggfunc="first",
    )
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    needed = [
        "eval_MASE__Standalone",
        "eval_MASE__NaiveKD",
        "eval_MASE__HardRejectSR",
        "eval_MASE__SoftSR",
    ]
    wide = wide.dropna(subset=needed)

    ns = wide["eval_MASE__Standalone"].to_numpy()
    for method in ("NaiveKD", "PersistenceGateKD", "HardRejectSR", "SoftSR"):
        col = f"eval_MASE__{method}"
        if col in wide:
            wide[f"regret__{method}"] = (
                wide[col] - ns
            ) / np.maximum(ns, 1e-12)

    return wide


def dataset_mean_regrets(wide):
    regret_cols = [c for c in wide.columns if c.startswith("regret__")]
    return (
        wide
        .groupby(["dataset", "teacher", "teacher_family"], as_index=False)[regret_cols]
        .mean()
    )


def analyze_confirmatory(results):
    wide = prepare_wide(results)
    dataset_level = dataset_mean_regrets(wide)

    # ---------------------------------------------------------------
    # H1: quantum teachers only; Holm within each teacher's two tests.
    # ---------------------------------------------------------------
    h1_rows = []
    for teacher in CONFIRMATORY_TEACHERS:
        w = dataset_level[dataset_level.teacher == teacher]
        if w.empty:
            continue

        teacher_rows = []
        for comparator in ("NaiveKD", "PersistenceGateKD"):
            diff = (
                w["regret__SoftSR"].to_numpy()
                - w[f"regret__{comparator}"].to_numpy()
            )
            if np.allclose(diff, 0):
                stat, p = 0.0, 1.0
            else:
                stat, p = wilcoxon(
                    diff,
                    alternative="less",
                    zero_method="wilcox",
                    method="auto",
                )

            ci = bootstrap_dataset_mean_ci(
                diff,
                CFG.bootstrap_stat_samples,
                0.95,
                stable_seed(RUN_FINGERPRINT, "H1", teacher, comparator),
            )
            teacher_rows.append({
                "teacher": teacher,
                "comparator": comparator,
                "family": f"H1_{teacher}_two_comparators",
                "n_datasets": len(diff),
                "mean_diff_regret": ci["mean"],
                "median_diff_regret": float(np.median(diff)),
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "wilcoxon_stat": float(stat),
                "p_raw": float(p),
            })

        adjusted = holm_adjust([r["p_raw"] for r in teacher_rows])
        for r, p_adj in zip(teacher_rows, adjusted):
            r["p_holm_within_teacher"] = p_adj
            h1_rows.append(r)

    H1 = pd.DataFrame(h1_rows)

    # ---------------------------------------------------------------
    # H2a: non-inferiority SoftSR vs HardRejectSR at dataset level.
    # ---------------------------------------------------------------
    h2a_rows = []
    for teacher in CONFIRMATORY_TEACHERS:
        w = dataset_level[dataset_level.teacher == teacher]
        if w.empty:
            continue
        diff = (
            w["regret__SoftSR"].to_numpy()
            - w["regret__HardRejectSR"].to_numpy()
        )
        ci = bootstrap_dataset_mean_ci(
            diff,
            CFG.bootstrap_stat_samples,
            0.95,
            stable_seed(RUN_FINGERPRINT, "H2a", teacher),
        )
        h2a_rows.append({
            "teacher": teacher,
            "n_datasets": len(diff),
            "mean_diff_soft_minus_hard": ci["mean"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "delta_NI": CFG.delta_ni,
            "non_inferior": bool(ci["ci_high"] < CFG.delta_ni),
        })
    H2a = pd.DataFrame(h2a_rows)

    # ---------------------------------------------------------------
    # H2b: benefit retention on SoftSR-accepted, meaningfully beneficial cells.
    # ---------------------------------------------------------------
    h2b_rows = []
    for teacher in CONFIRMATORY_TEACHERS:
        w = wide[wide.teacher == teacher].copy()
        if w.empty:
            continue

        ns = w["eval_MASE__Standalone"].to_numpy()
        naive = w["eval_MASE__NaiveKD"].to_numpy()
        soft = w["eval_MASE__SoftSR"].to_numpy()
        hard = w["eval_MASE__HardRejectSR"].to_numpy()

        relative_naive_benefit = (ns - naive) / np.maximum(ns, 1e-12)
        eligible_mask = (
            (w["gate__SoftSR"].to_numpy() > 0)
            & (relative_naive_benefit >= CFG.benefit_min_relative)
        )

        eligible = w.loc[eligible_mask, ["dataset", "seed"]].copy()
        if eligible.empty:
            h2b_rows.append({
                "teacher": teacher,
                "n_datasets": 0,
                "n_seed_cells": 0,
                "mean_retention_diff": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "wilcoxon_p_one_sided": np.nan,
                "benefit_min_relative": CFG.benefit_min_relative,
                "superior_retention": False,
            })
            continue

        idx = np.where(eligible_mask)[0]
        available = ns[idx] - naive[idx]
        retention_soft = (ns[idx] - soft[idx]) / np.maximum(available, 1e-12)
        retention_hard = (ns[idx] - hard[idx]) / np.maximum(available, 1e-12)
        eligible["retention_diff"] = retention_soft - retention_hard

        per_dataset = eligible.groupby("dataset")["retention_diff"].mean()
        vals = per_dataset.to_numpy()
        ci = bootstrap_dataset_mean_ci(
            vals,
            CFG.bootstrap_stat_samples,
            0.95,
            stable_seed(RUN_FINGERPRINT, "H2b", teacher),
        )

        if len(vals) >= 2 and not np.allclose(vals, 0):
            _, p_ret = wilcoxon(vals, alternative="greater", zero_method="wilcox", method="auto")
            p_ret = float(p_ret)
        else:
            p_ret = np.nan

        h2b_rows.append({
            "teacher": teacher,
            "n_datasets": len(vals),
            "n_seed_cells": len(eligible),
            "mean_retention_diff": ci["mean"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "wilcoxon_p_one_sided": p_ret,
            "benefit_min_relative": CFG.benefit_min_relative,
            "superior_retention": bool(ci["ci_low"] > 0),
        })
    H2b = pd.DataFrame(h2b_rows)

    # Combined H2 decision requires both components when H2b is estimable.
    H2_summary_rows = []
    for teacher in CONFIRMATORY_TEACHERS:
        a = H2a[H2a.teacher == teacher]
        b = H2b[H2b.teacher == teacher]
        if a.empty:
            continue
        ni = bool(a.iloc[0]["non_inferior"])
        b_estimable = (not b.empty) and int(b.iloc[0]["n_datasets"]) > 0
        retention = bool(b.iloc[0]["superior_retention"]) if b_estimable else False
        H2_summary_rows.append({
            "teacher": teacher,
            "H2a_non_inferior": ni,
            "H2b_estimable": b_estimable,
            "H2b_superior_retention": retention,
            "H2_supported": bool(ni and b_estimable and retention),
        })
    H2_summary = pd.DataFrame(H2_summary_rows)

    # ---------------------------------------------------------------
    # H3: accepted SoftSR cells vs better of Standalone and NaiveKD.
    # ---------------------------------------------------------------
    h3_rows = []
    for teacher in CONFIRMATORY_TEACHERS:
        w = wide[wide.teacher == teacher].copy()
        accepted = w[w["gate__SoftSR"] > 0].copy()

        if accepted.empty:
            h3_rows.append({
                "teacher": teacher,
                "n_datasets_with_acceptance": 0,
                "n_seed_cells_accepted": 0,
                "mean_regret": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "delta_NI": CFG.delta_ni,
                "non_inferior": False,
            })
            continue

        best = np.minimum(
            accepted["eval_MASE__Standalone"].to_numpy(),
            accepted["eval_MASE__NaiveKD"].to_numpy(),
        )
        accepted["accepted_regret"] = (
            accepted["eval_MASE__SoftSR"].to_numpy() - best
        ) / np.maximum(best, 1e-12)

        per_dataset = accepted.groupby("dataset")["accepted_regret"].mean().to_numpy()
        ci = bootstrap_dataset_mean_ci(
            per_dataset,
            CFG.bootstrap_stat_samples,
            0.95,
            stable_seed(RUN_FINGERPRINT, "H3", teacher),
        )
        h3_rows.append({
            "teacher": teacher,
            "n_datasets_with_acceptance": len(per_dataset),
            "n_seed_cells_accepted": len(accepted),
            "mean_regret": ci["mean"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "delta_NI": CFG.delta_ni,
            "non_inferior": bool(ci["ci_high"] < CFG.delta_ni),
        })
    H3 = pd.DataFrame(h3_rows)

    # ---------------------------------------------------------------
    # H4: directional generality for confirmatory quantum teachers only.
    # ---------------------------------------------------------------
    h4_rows = []
    for teacher in CONFIRMATORY_TEACHERS:
        w = dataset_level[dataset_level.teacher == teacher]
        for comparator in ("NaiveKD", "PersistenceGateKD"):
            diff = (
                w["regret__SoftSR"].to_numpy()
                - w[f"regret__{comparator}"].to_numpy()
            )
            h4_rows.append({
                "teacher": teacher,
                "comparator": comparator,
                "n_datasets": len(diff),
                "n_favorable_datasets": int(np.sum(diff < 0)),
                "fraction_favorable": float(np.mean(diff < 0)) if len(diff) else np.nan,
            })
    H4 = pd.DataFrame(h4_rows)

    # Secondary controls: reported descriptively, not merged into confirmatory H1-H4.
    controls_dataset_level = dataset_level[
        dataset_level.teacher.isin(SECONDARY_CONTROL_TEACHERS)
    ].copy()

    seed_variability = (
        wide
        .groupby(["dataset", "teacher", "teacher_family"])
        .agg(
            n_seeds=("seed", "nunique"),
            soft_regret_mean=("regret__SoftSR", "mean"),
            soft_regret_std=("regret__SoftSR", "std"),
        )
        .reset_index()
    )

    return {
        "wide_seed_level": wide,
        "dataset_level": dataset_level,
        "H1": H1,
        "H2a_noninferiority": H2a,
        "H2b_benefit_retention": H2b,
        "H2_summary": H2_summary,
        "H3": H3,
        "H4": H4,
        "secondary_controls_dataset_level": controls_dataset_level,
        "seed_variability": seed_variability,
    }

# %% cell 19
# 17. Save confirmatory analysis, figures and reproducibility manifests


def save_analysis(results):
    analyses = analyze_confirmatory(results)

    run_analysis_dir = ANALYSIS_DIR / RUN_TAG
    run_analysis_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        results
        .groupby(["teacher_family", "teacher", "method"], dropna=False)
        .agg(
            n_seed_runs=("eval_MASE", "size"),
            n_datasets=("dataset", "nunique"),
            mean_MASE=("eval_MASE", "mean"),
            median_MASE=("eval_MASE", "median"),
            std_MASE=("eval_MASE", "std"),
            mean_gate=("gate", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(run_analysis_dir / "summary_by_teacher_method.csv", index=False)

    for name, df in analyses.items():
        df.to_csv(run_analysis_dir / f"{name}.csv", index=False)

    display(summary)
    for name in (
        "H1",
        "H2a_noninferiority",
        "H2b_benefit_retention",
        "H2_summary",
        "H3",
        "H4",
    ):
        print("\n", name)
        display(analyses[name])

    return analyses


def plot_test_mase(results):
    run_analysis_dir = ANALYSIS_DIR / RUN_TAG
    run_analysis_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = (
        results
        .groupby(["dataset", "teacher", "method"])["eval_MASE"]
        .mean()
        .reset_index()
    )
    global_summary = (
        dataset_summary
        .groupby(["teacher", "method"])["eval_MASE"]
        .mean()
        .unstack("method")
    )

    ax = global_summary.plot(kind="bar", figsize=(11, 4.5))
    ax.set_ylabel("Mean dataset-level test MASE")
    ax.set_xlabel("Teacher")
    ax.set_title("Confirmatory performance (seeds averaged within dataset)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(run_analysis_dir / "confirmatory_test_mase.pdf", bbox_inches="tight")
    plt.savefig(run_analysis_dir / "confirmatory_test_mase.png", dpi=300, bbox_inches="tight")
    plt.show()


def create_artifact_manifest(root=RESULTS_DIR):
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(RESULTS_DIR / f"artifact_manifest_sha256_{RUN_TAG}.csv", index=False)
    return manifest


def write_run_manifest(status):
    full_path = result_path("confirmatory_results", "final_test")
    manifest = {
        "status": status,
        "updated_utc": utc_now(),
        "profile": CFG.profile_name,
        "run_tag": RUN_TAG,
        "run_fingerprint": RUN_FINGERPRINT,
        "config_sha256": CFG_SHA256,
        "code_fingerprint": CODE_FINGERPRINT,
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "environment": ENVIRONMENT,
        "raw_dataset_manifest": str(MANIFEST_DIR / "dataset_raw_manifest.csv"),
        "run_identity": str(MANIFEST_DIR / f"run_identity_{RUN_TAG}.json"),
        "results_path": str(full_path),
        "analysis_dir": str(ANALYSIS_DIR / RUN_TAG),
    }
    with (MANIFEST_DIR / f"run_manifest_{RUN_TAG}.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


_full_results_path = result_path("confirmatory_results", "final_test")

if _full_results_path.exists():
    full_results = pd.read_csv(_full_results_path)
    complete_blocks = completed_teacher_blocks(full_results, "final_test")
    expected_blocks = len(DATASETS) * len(CFG.seeds) * len(TEACHER_NAMES)

    if len(complete_blocks) == expected_blocks:
        unique_splits = set(
            full_results["evaluation_split"].dropna().astype(str)
        )
        if unique_splits != {"final_test"}:
            raise RuntimeError(
                f"FULL file is contaminated by non-final rows: {sorted(unique_splits)}"
            )

        print("Full confirmatory FINAL TEST run is complete for run tag", RUN_TAG)
        analyses = save_analysis(full_results)
        plot_test_mase(full_results)
        write_run_manifest("complete")
    else:
        print(
            f"Partial results for {RUN_TAG}: "
            f"{len(complete_blocks)}/{expected_blocks} teacher blocks complete. "
            "Rerun FULL to resume."
        )
        write_run_manifest("partial")
else:
    print("No full confirmatory result file yet for run tag", RUN_TAG)
    write_run_manifest("not_started")

artifact_manifest = create_artifact_manifest()
display(artifact_manifest.tail())

# %% [markdown] cell 20
# 
# ## Reviewer-facing checklist before FULL and before GitHub / Zenodo freeze
# 
# ### Before FULL
# 
# - [ ] Run `SMOKE` and confirm `Final-test materialization lock: PASS`.
# - [ ] Confirm SMOKE output contains only `evaluation_split = development`.
# - [ ] Confirm dynamic `g=0` DEVELOPMENT invariance is within `1e-6`.
# - [ ] Run `PROFILE`; confirm it reports `Final test remained sealed during PROFILE`.
# - [ ] Confirm PROFILE output contains only `evaluation_split = development`.
# - [ ] Do not inspect or manually construct final-test labels.
# - [ ] Freeze all thresholds, loss weights, NI margin, benefit threshold, bootstrap rule, seeds and teacher definitions before setting `RUN_MODE = "FULL"`.
# 
# ### FULL
# 
# - [ ] Change only `RUN_MODE = "FULL"`.
# - [ ] Confirm console prints `FINAL TEST ACCESS: UNSEALED BY FULL MODE`.
# - [ ] FULL must write only `confirmatory_results_final_test_<RUN_TAG>.csv`.
# - [ ] Never merge SMOKE or PROFILE rows into the FULL CSV.
# - [ ] If the wall-clock guard stops the run, rerun the same FULL; it resumes the same run fingerprint.
# 
# ### Before publication freeze
# 
# - [ ] Confirm every row in the confirmatory CSV has `evaluation_split = final_test`.
# - [ ] Confirm confirmatory analysis refuses Development rows.
# - [ ] Archive raw dataset SHA-256 hashes and dataset fingerprint.
# - [ ] Archive processed numeric matrix hashes.
# - [ ] Archive exact train/calibration/development/final-test window indices and hashes.
# - [ ] Archive `run_identity_<RUN_TAG>.json`.
# - [ ] Archive configuration, environment manifest and `pip_freeze.txt`.
# - [ ] Archive final-test results and all run-tagged analysis CSVs.
# - [ ] Report seeds as stochastic replications and datasets as inferential units.
# - [ ] State H1 Holm correction exactly as implemented.
# - [ ] Report H2a and H2b separately.
# - [ ] Report moving-block bootstrap rules and inferred block lengths.
# - [ ] Keep RFF/ESN as secondary controls.
# - [ ] Do not claim intrinsic quantum advantage solely from selective-distillation performance.
# - [ ] Cite the exact GitHub release and Zenodo DOI corresponding to this run fingerprint.
# 
# ### Recommended manuscript statement
# 
# > Hyperparameters, gating thresholds and statistical procedures were frozen before the confirmatory run. Development diagnostics used a dedicated chronological development holdout. The final test partition was not materialized or evaluated during debugging, smoke testing or computational profiling and was accessed only by the final confirmatory execution path.

# %% cell 21
# 18. Optional: package completed reproducibility outputs for archival


def package_reproducibility_outputs():
    archive_base = ROOT / (
        f"quantum_cenn_reproducibility_{CFG.profile_name}_{RUN_TAG}"
    )
    archive_path = shutil.make_archive(
        str(archive_base),
        "zip",
        root_dir=str(RESULTS_DIR),
    )
    print("Created reproducibility archive:", archive_path)
    print("SHA-256:", sha256_file(Path(archive_path)))
    print("Run fingerprint:", RUN_FINGERPRINT)
    return Path(archive_path)

# Run only after FULL is complete:
# reproducibility_zip = package_reproducibility_outputs()
