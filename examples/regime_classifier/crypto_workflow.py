# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
End-to-end crypto regime classification + derivatives backtesting workflow.

This script ties together the full pipeline described in
``docs/REGIME_PROJECT_PLAN.md``:

  Step 1 — Download data
      OHLCV from Binance, funding rates from Binance futures,
      DVOL (implied vol index) from Deribit.

  Step 2 — Build feature handler
      Stationary OHLCV features (vol as percentile rank) via
      RegimeDataHandler.  Funding and DVOL merged in afterward.

  Step 3 — Fit HMM regime model (K=3, forward-filtered decode)
      BIC is run as a diagnostic; K is fixed at 3 to avoid overfitting.
      Hysteresis suppresses state chatter.

  Step 4 — Predict regimes on a test slice (no walk-forward)
      Quick preview of state distribution and transition probabilities.

  Step 5 — Run full walk-forward validation
      W1=18mo fit / W2=6mo select / W3=3mo OOS, rolled quarterly.
      Baselines: buy-and-hold, constant FundingCarry, vol-targeting.
      Reports aggregate OOS Sharpe and hit rate vs baselines.

  Step 6 — (Sealed holdout — run manually after research phase)
      Evaluate the final frozen pipeline on the last 15 months of data.
      Touch exactly once.

Usage
-----
::

    # 1. Download all data
    python crypto_workflow.py download_data

    # 2. Quick regime preview (no walk-forward)
    python crypto_workflow.py predict_only

    # 3. Full walk-forward research run
    python crypto_workflow.py run

    # 4. Sealed holdout (run once, after research is finalised)
    python crypto_workflow.py holdout

Dependencies
------------
    pip install qlib hmmlearn lightgbm scikit-learn scipy requests fire
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import fire
import numpy as np
import pandas as pd

DIRNAME = Path(__file__).absolute().resolve().parent
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Configuration (single source of truth for all magic numbers)
# ---------------------------------------------------------------------------

CFG = dict(
    # --- Data paths ---
    ohlcv_dir="~/.qlib/qlib_data/crypto_binance/1d",
    funding_dir="~/.qlib/qlib_data/crypto_binance/funding",
    dvol_dir="~/.qlib/qlib_data/crypto_deribit",
    # --- Universe ---
    instruments=["btcusdt", "ethusdt"],
    # --- Period ---
    train_start="2018-01-01",
    holdout_start="2024-10-01",   # sealed; change only if extending dataset
    data_end="2025-06-30",         # update as new data arrives
    # --- HMM ---
    hmm_n_states=3,
    hmm_n_seeds=15,
    hmm_n_iter=300,
    hmm_hysteresis_prob=0.70,
    hmm_hysteresis_bars=3,
    # --- Walk-forward windows (months) ---
    wf_fit_months=18,
    wf_select_months=6,
    wf_oos_months=3,
    # --- Strategy selection ---
    selector_metric="sharpe",
    selector_min_obs=80,
    selector_margin=0.30,
    selector_bootstrap_n=500,
    selector_bootstrap_hit_rate=0.60,
    fallback_strategy="FundingCarry",
    trans_prob_thresh=0.45,
    # --- Options (only available from 2021-07 onward) ---
    options_start="2021-07-01",
    # --- Annualisation ---
    annualization=365,
)


# ---------------------------------------------------------------------------
# Step 1 — Data download
# ---------------------------------------------------------------------------

def download_data(
    ohlcv_dir: str = CFG["ohlcv_dir"],
    funding_dir: str = CFG["funding_dir"],
    dvol_dir: str = CFG["dvol_dir"],
    start_date: str = CFG["train_start"],
):
    """Download OHLCV, funding rates, and DVOL from public APIs."""
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "data_collector"))

    # OHLCV (Binance spot)
    logger.info("=== Downloading OHLCV ===")
    from crypto_binance.collector import collect as collect_ohlcv
    collect_ohlcv(
        symbols=["BTCUSDT", "ETHUSDT"],
        freq="1d",
        start_date=start_date,
        output_dir=ohlcv_dir,
    )

    # Funding rates (Binance futures)
    logger.info("=== Downloading funding rates ===")
    from crypto_binance.funding_collector import collect as collect_funding
    collect_funding(
        symbols=["BTCUSDT", "ETHUSDT"],
        start_date="2019-09-01",
        output_dir=funding_dir,
    )

    # DVOL (Deribit)
    logger.info("=== Downloading Deribit DVOL ===")
    from crypto_deribit.dvol_collector import collect as collect_dvol
    collect_dvol(
        currencies=["BTC", "ETH"],
        start_date="2021-03-01",
        output_dir=dvol_dir,
    )

    logger.info("Data download complete.")


# ---------------------------------------------------------------------------
# Qlib initialisation
# ---------------------------------------------------------------------------

def _init_qlib(provider_uri: str = CFG["ohlcv_dir"]):
    import qlib
    qlib.init(provider_uri=provider_uri, region="us")


# ---------------------------------------------------------------------------
# Feature / data helpers
# ---------------------------------------------------------------------------

def _load_funding(funding_dir: str, instruments) -> Optional[pd.DataFrame]:
    """Load funding CSV files and return a DataFrame aligned to instruments."""
    dfs = []
    for inst in instruments:
        path = Path(funding_dir).expanduser() / "features" / inst / "funding.csv"
        if not path.exists():
            logger.warning("Funding data not found: %s  (run download_data first)", path)
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "datetime"
        df["instrument"] = inst
        dfs.append(df)
    combined = pd.concat(dfs)
    return combined.reset_index().set_index(["datetime", "instrument"])


def _load_dvol(dvol_dir: str, instruments) -> Optional[pd.DataFrame]:
    """Load DVOL CSV files and return a DataFrame aligned to instruments."""
    dfs = []
    for inst in instruments:
        path = Path(dvol_dir).expanduser() / "features" / inst / "dvol.csv"
        if not path.exists():
            logger.warning("DVOL data not found: %s  (run download_data first)", path)
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "datetime"
        df["instrument"] = inst
        dfs.append(df)
    combined = pd.concat(dfs)
    return combined.reset_index().set_index(["datetime", "instrument"])


# ---------------------------------------------------------------------------
# Step 2–3 — Quick predict-only preview
# ---------------------------------------------------------------------------

def predict_only(
    provider_uri: str = CFG["ohlcv_dir"],
    train_start: str = CFG["train_start"],
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
    test_end: str = "2023-12-31",
    n_states: int = CFG["hmm_n_states"],
):
    """Fit HMM on the training slice and print regime predictions on test."""
    _init_qlib(provider_uri)

    from qlib.data.dataset import DatasetH
    from qlib.contrib.data.handler_regime import RegimeDataHandler
    from qlib.contrib.model.hmm_regime import HMMRegimeModel

    handler = RegimeDataHandler(
        instruments=CFG["instruments"],
        start_time=train_start,
        end_time=test_end,
        fit_start_time=train_start,
        fit_end_time=train_end,
        rank_window=CFG["annualization"],
    )
    dataset = DatasetH(
        handler=handler,
        segments={"train": (train_start, train_end), "test": (test_start, test_end)},
    )

    model = HMMRegimeModel(
        n_states=n_states,
        n_seeds=CFG["hmm_n_seeds"],
        n_iter=CFG["hmm_n_iter"],
        hysteresis_prob=CFG["hmm_hysteresis_prob"],
        hysteresis_bars=CFG["hmm_hysteresis_bars"],
    )
    logger.info("Fitting HMM regime model (K=%d)...", n_states)
    model.fit(dataset)

    regime_df = model.predict(dataset, segment="test", decode="filtered")

    print("\n=== Sample predictions (first 10 dates) ===")
    print(regime_df.groupby(level="datetime").first().head(10))
    print("\n=== State distribution (test) ===")
    print(regime_df["state"].value_counts().sort_index())
    print("\n=== Transition probability stats ===")
    print(regime_df.groupby("state")["trans_prob"].describe())

    return model, regime_df


# ---------------------------------------------------------------------------
# Step 4–5 — Full walk-forward research run
# ---------------------------------------------------------------------------

def run(
    provider_uri: str = CFG["ohlcv_dir"],
    funding_dir: str = CFG["funding_dir"],
    dvol_dir: str = CFG["dvol_dir"],
    train_start: str = CFG["train_start"],
    research_end: str = "2024-09-30",  # keep holdout sealed
):
    """Run the full walk-forward regime backtesting pipeline."""
    _init_qlib(provider_uri)

    from qlib.contrib.strategy.crypto_payoff import PerpSimulator, OptionSimulator
    from qlib.contrib.workflow.regime_walkforward import RegimeWalkForward

    # Load supplementary data
    funding_df = _load_funding(funding_dir, CFG["instruments"])
    dvol_df = _load_dvol(dvol_dir, CFG["instruments"])

    # Build per-instrument daily price and funding series
    prices_btc, prices_eth, funding_btc, funding_eth, dvol_btc, dvol_eth = \
        _extract_series(funding_df, dvol_df, provider_uri, research_end)

    perp = PerpSimulator()
    opt = OptionSimulator()

    def strategy_returns_factory(start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.Series]:
        """Return dict of strategy daily PnL for a given date range."""
        idx = pd.date_range(start, end, freq="D")
        p = prices_btc.reindex(idx).ffill()
        f = funding_btc.reindex(idx).fillna(0)
        d = dvol_btc.reindex(idx)

        returns = {
            "LongPerp":     perp.long_perp(p, f),
            "ShortPerp":    perp.short_perp(p, f),
            "FundingCarry": perp.funding_carry(p, f),
            "Flat":         perp.flat(idx),
        }
        # Options only available from 2021-07
        if start >= pd.Timestamp(CFG["options_start"]) and d.notna().any():
            returns["ShortStraddle"] = opt.short_straddle(p, d)
            returns["IronCondor"]    = opt.iron_condor(p, d)
            returns["BullPutSpread"] = opt.bull_put_spread(p, d)

        return returns

    wf = RegimeWalkForward(
        fit_months=CFG["wf_fit_months"],
        select_months=CFG["wf_select_months"],
        oos_months=CFG["wf_oos_months"],
        hmm_n_states=CFG["hmm_n_states"],
        hmm_n_seeds=CFG["hmm_n_seeds"],
        hmm_n_iter=CFG["hmm_n_iter"],
        hmm_hysteresis_prob=CFG["hmm_hysteresis_prob"],
        hmm_hysteresis_bars=CFG["hmm_hysteresis_bars"],
        selector_metric=CFG["selector_metric"],
        selector_min_obs=CFG["selector_min_obs"],
        selector_margin=CFG["selector_margin"],
        selector_bootstrap_n=CFG["selector_bootstrap_n"],
        selector_bootstrap_hit_rate=CFG["selector_bootstrap_hit_rate"],
        fallback_strategy=CFG["fallback_strategy"],
        trans_prob_thresh=CFG["trans_prob_thresh"],
    )

    # Load features DataFrame
    features_df = _load_features(provider_uri, train_start, research_end)

    # Baselines (cover the full potential OOS period)
    oos_start = pd.Timestamp(train_start) + pd.DateOffset(
        months=CFG["wf_fit_months"] + CFG["wf_select_months"]
    )
    oos_idx = pd.date_range(oos_start, research_end, freq="D")
    p_full = prices_btc.reindex(oos_idx).ffill()
    f_full = funding_btc.reindex(oos_idx).fillna(0)
    rv20 = p_full.pct_change().rolling(20).std() * np.sqrt(CFG["annualization"])
    vol_target_exposure = np.minimum(0.40 / (rv20 + 1e-8), 2.0)  # cap 2x
    baselines = {
        "BuyAndHold":     p_full.pct_change().rename("BuyAndHold"),
        "FundingCarry":   PerpSimulator().funding_carry(p_full, f_full),
        "VolTargeting":   (p_full.pct_change() * vol_target_exposure.shift(1)).rename("VolTargeting"),
    }

    logger.info("Starting walk-forward run: %s → %s", train_start, research_end)
    result = wf.run(features_df, strategy_returns_factory, baselines=baselines)

    # Print summary
    print("\n" + "=" * 60)
    print("WALK-FORWARD RESULTS")
    print("=" * 60)
    print(result.summary().to_string())
    print(f"\nAggregate OOS Sharpe:  {result.agg_sharpe:.3f} ± {result.agg_sharpe_std:.3f}")
    print(f"Sharpe t-stat:         {result.sharpe_tstat:.2f}")
    print(f"OOS windows:           {len(result.windows)}")
    print(f"Worst window Sharpe:   {result.agg_sharpe_min:.3f}")
    print(f"Aggregate Calmar:      {result.agg_calmar:.3f}")
    print(f"Worst drawdown:        {result.agg_max_dd:.1%}")
    print(f"Total strategy switches: {result.total_switch_count}")
    print("\nHit rate vs baselines:")
    for name, rate in result.hit_rate_vs_baseline.items():
        print(f"  vs {name:20s}: {rate:.1%}")

    # Log trial for multiple-testing ledger
    RegimeWalkForward.log_trial(wf.__dict__, result.agg_sharpe)

    # Plot if matplotlib available
    try:
        result.plot_oos_pnl()
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Step 6 — Sealed holdout (one shot)
# ---------------------------------------------------------------------------

def holdout(
    provider_uri: str = CFG["ohlcv_dir"],
    funding_dir: str = CFG["funding_dir"],
    dvol_dir: str = CFG["dvol_dir"],
):
    """Evaluate the frozen pipeline on the sealed holdout period.

    Run exactly ONCE after the research phase is complete and all
    hyperparameter choices are frozen.  The result is reported regardless
    of outcome.
    """
    print("=" * 60)
    print("SEALED HOLDOUT EVALUATION")
    print(f"Period: {CFG['holdout_start']} → {CFG['data_end']}")
    print("This will be run exactly once.  Are you sure? [y/N] ", end="")
    answer = input().strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    logger.info("Running sealed holdout: %s → %s", CFG["holdout_start"], CFG["data_end"])
    # Re-use the run() function but set the period to the holdout slice.
    # The walk-forward harness will fit on the immediately preceding
    # train/select windows and produce a single OOS window covering the holdout.
    result = run(
        provider_uri=provider_uri,
        funding_dir=funding_dir,
        dvol_dir=dvol_dir,
        train_start=str(pd.Timestamp(CFG["holdout_start"]) - pd.DateOffset(
            months=CFG["wf_fit_months"] + CFG["wf_select_months"]
        ))[:10],
        research_end=CFG["data_end"],
    )

    print("\n=== HOLDOUT RESULT ===")
    print(f"Sharpe (holdout): {result.agg_sharpe:.3f}")
    print(f"Max DD (holdout): {result.agg_max_dd:.1%}")
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_features(provider_uri: str, start: str, end: str) -> pd.DataFrame:
    """Load regime features as a plain date-indexed DataFrame (cross-sectional mean)."""
    _init_qlib(provider_uri)
    from qlib.data.dataset import DatasetH
    from qlib.contrib.data.handler_regime import RegimeDataHandler
    from qlib.data.dataset.handler import DataHandlerLP

    handler = RegimeDataHandler(
        instruments=CFG["instruments"],
        start_time=start,
        end_time=end,
        fit_start_time=start,
        fit_end_time=str(pd.Timestamp(start) + pd.DateOffset(months=CFG["wf_fit_months"]))[:10],
        rank_window=CFG["annualization"],
    )
    dataset = DatasetH(
        handler=handler,
        segments={"train": (start, end), "test": (start, end)},
    )
    df = dataset.prepare("train", col_set=["feature"], data_key=DataHandlerLP.DK_L)
    # Drop MultiIndex level if present; take cross-sectional mean
    if isinstance(df.columns, pd.MultiIndex):
        df = df["feature"]
    if isinstance(df.index, pd.MultiIndex):
        df = df.groupby(level="datetime").mean()
    return df.dropna()


def _extract_series(funding_df, dvol_df, provider_uri, end_date):
    """Extract per-instrument price, funding, DVOL series from loaded DataFrames."""
    import qlib
    from qlib.data import D as QD

    _init_qlib(provider_uri)
    freq = "day"
    dates = pd.date_range(CFG["train_start"], end_date, freq="D")

    def _qlib_close(inst):
        try:
            s = QD.features([inst], ["$close"], freq=freq).iloc[:, 0]
            s.index = s.index.get_level_values("datetime")
            return s.reindex(dates)
        except Exception:
            return pd.Series(np.nan, index=dates)

    prices_btc = _qlib_close("btcusdt")
    prices_eth = _qlib_close("ethusdt")

    def _funding(df, inst):
        if df is None:
            return pd.Series(0.0, index=dates)
        try:
            return df.xs(inst, level="instrument")["funding_daily"].reindex(dates).fillna(0)
        except KeyError:
            return pd.Series(0.0, index=dates)

    def _dvol(df, inst):
        if df is None:
            return pd.Series(np.nan, index=dates)
        try:
            return df.xs(inst, level="instrument")["dvol_close"].reindex(dates)
        except KeyError:
            return pd.Series(np.nan, index=dates)

    return (
        prices_btc, prices_eth,
        _funding(funding_df, "btcusdt"), _funding(funding_df, "ethusdt"),
        _dvol(dvol_df, "btcusdt"), _dvol(dvol_df, "ethusdt"),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fire.Fire({
        "download_data": download_data,
        "predict_only":  predict_only,
        "run":           run,
        "holdout":       holdout,
    })
