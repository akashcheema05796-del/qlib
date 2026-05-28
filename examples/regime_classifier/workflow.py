# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
End-to-end example: HMM Regime Classifier + Regime-Gated Strategy.

This script demonstrates how to integrate the regime classifier into a Qlib
workflow:

  1. Build a RegimeDataHandler with OHLCV-derived features.
  2. Fit an HMMRegimeModel (BIC-based state selection + LightGBM transition
     classifier) on the training segment.
  3. Predict regimes on the test segment and inspect the output.
  4. Run a RegimeGatedStrategy backtest that modulates portfolio exposure based
     on the detected regime and transition probability.

Usage
-----
::

    # Download sample data first (CSI 300, daily bars)
    python workflow.py download_data

    # Run the full regime classification + backtest
    python workflow.py run

    # Inspect regime predictions only (no backtest)
    python workflow.py predict_only

Dependencies
------------
    pip install qlib hmmlearn lightgbm scikit-learn
"""

import os
from pathlib import Path

import fire
import pandas as pd

DIRNAME = Path(__file__).absolute().resolve().parent


# ---------------------------------------------------------------------------
# Data download helper
# ---------------------------------------------------------------------------

def download_data(provider_uri: str = "~/.qlib/qlib_data/cn_data"):
    """Download Qlib sample data (CSI 300, daily)."""
    from qlib.tests.data import GetData

    GetData().qlib_data(target_dir=provider_uri, exists_skip=True)
    print(f"Data written to {provider_uri}")


# ---------------------------------------------------------------------------
# Shared initialisation
# ---------------------------------------------------------------------------

def _init_qlib(provider_uri: str = "~/.qlib/qlib_data/cn_data"):
    import qlib
    from qlib.config import REG_CN

    qlib.init(provider_uri=provider_uri, region=REG_CN)


# ---------------------------------------------------------------------------
# Regime prediction demo
# ---------------------------------------------------------------------------

def predict_only(
    provider_uri: str = "~/.qlib/qlib_data/cn_data",
    instruments: str = "csi300",
    train_start: str = "2010-01-01",
    train_end: str = "2019-12-31",
    test_start: str = "2020-01-01",
    test_end: str = "2022-12-31",
    n_states: str = "auto",
    max_states: int = 5,
):
    """Fit the HMM regime model and print per-date regime labels."""
    _init_qlib(provider_uri)

    from qlib.data.dataset import DatasetH
    from qlib.contrib.data.handler_regime import RegimeDataHandler
    from qlib.contrib.model.hmm_regime import HMMRegimeModel

    # --- Build handler ---
    handler = RegimeDataHandler(
        instruments=instruments,
        start_time=train_start,
        end_time=test_end,
        fit_start_time=train_start,
        fit_end_time=train_end,
    )

    # --- Build dataset with train / test segments ---
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (train_start, train_end),
            "test": (test_start, test_end),
        },
    )

    # --- Fit model ---
    model = HMMRegimeModel(
        n_states=n_states,
        max_states=max_states,
        n_seeds=5,
        transition_horizon=5,
    )
    model.fit(dataset)

    print("\nRegime map:", model.regime_map)

    # --- Predict ---
    regime_df = model.predict(dataset, segment="test")
    print("\nSample regime predictions (first 10 dates):")
    print(regime_df.groupby(level="datetime").first().head(10))

    # --- Regime distribution ---
    dist = regime_df["regime"].value_counts()
    print("\nRegime distribution (test set):")
    print(dist.to_string())

    return model, regime_df


# ---------------------------------------------------------------------------
# Full backtest
# ---------------------------------------------------------------------------

def run(
    provider_uri: str = "~/.qlib/qlib_data/cn_data",
    instruments: str = "csi300",
    train_start: str = "2010-01-01",
    train_end: str = "2019-12-31",
    test_start: str = "2020-01-01",
    test_end: str = "2022-12-31",
    topk: int = 50,
    n_drop: int = 5,
    n_states: str = "auto",
    max_states: int = 5,
):
    """Run regime-gated TopK backtest against a plain TopK baseline."""
    _init_qlib(provider_uri)

    from qlib.data.dataset import DatasetH
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.data.handler_regime import RegimeDataHandler
    from qlib.contrib.model.hmm_regime import HMMRegimeModel
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    from qlib.contrib.strategy.regime_gated import RegimeGatedStrategy
    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.report import analysis_model, analysis_position

    # ---- Alpha signal dataset (Alpha158 features, LightGBM alpha model) ----
    alpha_handler = Alpha158(
        instruments=instruments,
        start_time=train_start,
        end_time=test_end,
        fit_start_time=train_start,
        fit_end_time=train_end,
    )
    alpha_dataset = DatasetH(
        handler=alpha_handler,
        segments={
            "train": (train_start, train_end),
            "valid": (train_end, test_start),
            "test": (test_start, test_end),
        },
    )
    alpha_model = LGBModel()
    alpha_model.fit(alpha_dataset)

    # ---- Regime dataset + model ----
    regime_handler = RegimeDataHandler(
        instruments=instruments,
        start_time=train_start,
        end_time=test_end,
        fit_start_time=train_start,
        fit_end_time=train_end,
    )
    regime_dataset = DatasetH(
        handler=regime_handler,
        segments={
            "train": (train_start, train_end),
            "test": (test_start, test_end),
        },
    )
    regime_model = HMMRegimeModel(
        n_states=n_states,
        max_states=max_states,
        n_seeds=5,
        transition_horizon=5,
    )
    regime_model.fit(regime_dataset)
    regime_df = regime_model.predict(regime_dataset, segment="test")

    print("\nRegime map:", regime_model.regime_map)
    print("\nRegime distribution (test):")
    print(regime_df["regime"].value_counts().to_string())

    # ---- Regime-gated strategy ----
    regime_strategy = RegimeGatedStrategy(
        signal=(alpha_model, alpha_dataset),
        regime_signal=regime_df,
        trans_prob_thresh=0.40,
        base_risk_degree=0.80,
    )

    # ---- Baseline: plain TopK strategy ----
    baseline_strategy = TopkDropoutStrategy(
        signal=(alpha_model, alpha_dataset),
        topk=topk,
        n_drop=n_drop,
    )

    # ---- Backtest both strategies ----
    print("\nRunning regime-gated backtest...")
    report_regime, positions_regime = backtest_daily(
        start_time=test_start,
        end_time=test_end,
        strategy=regime_strategy,
    )

    print("Running baseline backtest...")
    report_baseline, positions_baseline = backtest_daily(
        start_time=test_start,
        end_time=test_end,
        strategy=baseline_strategy,
    )

    # ---- Print summary ----
    print("\n=== Regime-Gated Strategy ===")
    print(report_regime)

    print("\n=== Baseline TopK Strategy ===")
    print(report_baseline)

    return report_regime, report_baseline


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fire.Fire(
        {
            "download_data": download_data,
            "predict_only": predict_only,
            "run": run,
        }
    )
