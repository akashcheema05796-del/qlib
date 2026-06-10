# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
RegimeWalkForward: rolling W1-fit / W2-select / W3-OOS protocol for
crypto HMM regime-based strategy allocation.

Window layout
-------------
  W1  (fit_months)    : Fit HMMRegimeModel
  W2  (select_months) : Select per-state strategies via StateStrategySelector
  W3  (oos_months)    : Trade frozen strategy map OOS; measure PnL vs baselines

The whole window rolls forward by ``oos_months`` each iteration (quarterly roll
by default).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from ...log import get_module_logger

logger = get_module_logger("RegimeWalkForward")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------

class LeakageWarning(UserWarning):
    """Raised when a shifted-features run consistently out-performs the normal run."""


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

def _compute_sharpe(pnl: pd.Series, annualization: int = 365) -> float:
    """Annualized Sharpe ratio; returns 0.0 if std == 0 or series is empty."""
    clean = pnl.dropna()
    if len(clean) < 2:
        return 0.0
    std = float(clean.std())
    if std == 0.0:
        return 0.0
    return float(clean.mean() / std * np.sqrt(annualization))


def _compute_calmar(pnl: pd.Series, annualization: int = 365) -> float:
    """Annualized return divided by max drawdown magnitude.

    Returns 0.0 when there is no drawdown or the series is too short.
    """
    clean = pnl.dropna()
    if len(clean) < 2:
        return 0.0
    ann_return = float(clean.mean() * annualization)
    max_dd = _compute_max_dd(clean)
    if max_dd == 0.0:
        return 0.0
    return ann_return / max_dd


def _compute_max_dd(pnl: pd.Series) -> float:
    """Maximum drawdown as a positive fraction of peak equity.

    Computed on cumulative sum of daily fractional PnL (not compound growth).
    Returns 0.0 if the series is empty or there is no drawdown.
    """
    clean = pnl.dropna()
    if len(clean) < 2:
        return 0.0
    cumulative = clean.cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    return float(drawdown.max())


def _block_bootstrap_sharpe(
    pnl: pd.Series,
    n_resamples: int = 1000,
    block_size: int = 15,
) -> np.ndarray:
    """Block bootstrap distribution of annualized Sharpe ratios.

    Parameters
    ----------
    pnl : pd.Series
        Daily fractional PnL.
    n_resamples : int
        Number of bootstrap samples.
    block_size : int
        Length (in days) of each contiguous block drawn during resampling.

    Returns
    -------
    np.ndarray of shape (n_resamples,)
        Bootstrapped Sharpe values.
    """
    clean = pnl.dropna().values
    n = len(clean)
    if n < block_size:
        return np.array([_compute_sharpe(pnl)] * n_resamples)

    rng = np.random.default_rng(seed=0)
    sharpes = np.empty(n_resamples)
    for i in range(n_resamples):
        # Number of blocks needed to cover n observations
        n_blocks = int(np.ceil(n / block_size))
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([clean[s : s + block_size] for s in starts])[:n]
        std = sample.std()
        sharpes[i] = (sample.mean() / std * np.sqrt(365)) if std > 0 else 0.0
    return sharpes


# ---------------------------------------------------------------------------
# Hysteresis filter
# ---------------------------------------------------------------------------

def _apply_hysteresis(
    states: np.ndarray,
    posterior: np.ndarray,
    prob_thresh: float = 0.70,
    n_bars: int = 3,
) -> np.ndarray:
    """Post-process raw argmax states with a two-level hysteresis gate.

    A state transition from state ``s`` to state ``s'`` is only accepted when:
    1. ``posterior[t, s'] >= prob_thresh`` for ``n_bars`` consecutive bars.

    This reduces whipsawing in uncertain regimes.

    Parameters
    ----------
    states : np.ndarray, shape (T,)
        Raw integer state sequence (argmax of posterior).
    posterior : np.ndarray, shape (T, K)
        State posterior probabilities at each time step.
    prob_thresh : float
        Minimum posterior probability to trigger a confirmed transition.
    n_bars : int
        Number of consecutive bars above threshold required.

    Returns
    -------
    np.ndarray of same shape as ``states``
        Hysteresis-filtered state sequence.
    """
    T = len(states)
    filtered = states.copy()
    current_state = states[0]
    candidate_state = -1
    candidate_count = 0

    for t in range(1, T):
        proposed = states[t]
        if proposed == current_state:
            candidate_state = -1
            candidate_count = 0
            filtered[t] = current_state
        else:
            # Proposed differs from current
            if proposed == candidate_state:
                if posterior[t, proposed] >= prob_thresh:
                    candidate_count += 1
                else:
                    candidate_count = 0
            else:
                # New candidate
                candidate_state = proposed
                candidate_count = 1 if posterior[t, proposed] >= prob_thresh else 0

            if candidate_count >= n_bars:
                current_state = candidate_state
                candidate_state = -1
                candidate_count = 0
            filtered[t] = current_state

    return filtered


# ---------------------------------------------------------------------------
# Internal: prepare feature matrix from DataFrame
# ---------------------------------------------------------------------------

def _extract_daily_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Return a date-indexed DataFrame of features suitable for HMM fitting.

    Handles both:
    - MultiIndex (datetime, instrument): computes cross-sectional mean per date.
    - Plain DatetimeIndex: uses as-is.
    """
    if isinstance(features_df.index, pd.MultiIndex):
        # Expect level names ("datetime", "instrument") or (0, 1)
        date_level = 0  # outermost level
        try:
            date_level = features_df.index.names.index("datetime")
        except ValueError:
            date_level = 0
        return features_df.groupby(level=date_level).mean()
    return features_df.copy()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    """Results for a single walk-forward window."""

    window_id: int
    w1_start: pd.Timestamp
    w3_end: pd.Timestamp
    state_strategy_map: Dict[int, str]      # {state: strategy_name}
    state_risk_map: Dict[int, float]
    oos_pnl: pd.Series                       # daily fraction of notional
    oos_sharpe: float
    oos_calmar: float
    oos_max_dd: float
    oos_win_rate: float
    switch_count: int                        # strategy changes in W3
    n_states_observed: int                   # distinct states seen in W3
    selector_report: pd.DataFrame            # from StateStrategySelector.report()


@dataclass
class WalkForwardResult:
    """Aggregate results across all walk-forward windows."""

    windows: List[WindowResult]
    config: dict                             # all hyperparameters

    # Aggregate statistics — populated in __post_init__
    agg_sharpe: float = field(init=False)
    agg_sharpe_std: float = field(init=False)
    agg_sharpe_min: float = field(init=False)
    sharpe_tstat: float = field(init=False)
    agg_calmar: float = field(init=False)
    agg_max_dd: float = field(init=False)
    hit_rate_vs_baseline: Dict[str, float] = field(init=False)
    total_switch_count: int = field(init=False)

    # Raw baseline OOS PnL for comparison (set externally if needed)
    _baseline_pnl: Dict[str, pd.Series] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        sharpes = [w.oos_sharpe for w in self.windows]
        n = len(sharpes)
        self.agg_sharpe = float(np.mean(sharpes)) if n else 0.0
        self.agg_sharpe_std = float(np.std(sharpes, ddof=1)) if n > 1 else 0.0
        self.agg_sharpe_min = float(np.min(sharpes)) if n else 0.0
        self.sharpe_tstat = (
            self.agg_sharpe / self.agg_sharpe_std * np.sqrt(n)
            if (n > 1 and self.agg_sharpe_std > 0)
            else 0.0
        )
        self.agg_calmar = float(np.mean([w.oos_calmar for w in self.windows])) if n else 0.0
        self.agg_max_dd = float(max((w.oos_max_dd for w in self.windows), default=0.0))
        self.hit_rate_vs_baseline = {}
        self.total_switch_count = sum(w.switch_count for w in self.windows)

    def _set_baseline_pnl(self, baseline_pnl: Dict[str, pd.Series]) -> None:
        """Attach full-period baseline PnL Series for use in plot_oos_pnl."""
        self._baseline_pnl = baseline_pnl

    def _compute_hit_rates(self, baselines: Dict[str, pd.Series]) -> None:
        """Compute fraction of windows where regime OOS beats each baseline."""
        if not baselines:
            return
        for bname, bpnl in baselines.items():
            hits = 0
            for w in self.windows:
                b_window = bpnl.reindex(w.oos_pnl.index)
                b_sharpe = _compute_sharpe(b_window)
                if w.oos_sharpe > b_sharpe:
                    hits += 1
            self.hit_rate_vs_baseline[bname] = hits / len(self.windows) if self.windows else 0.0

    def summary(self) -> pd.DataFrame:
        """Return per-window summary as a DataFrame."""
        rows = []
        for w in self.windows:
            rows.append(
                {
                    "window_id": w.window_id,
                    "w1_start": w.w1_start,
                    "w3_end": w.w3_end,
                    "oos_sharpe": w.oos_sharpe,
                    "oos_calmar": w.oos_calmar,
                    "oos_max_dd": w.oos_max_dd,
                    "oos_win_rate": w.oos_win_rate,
                    "switch_count": w.switch_count,
                    "n_states_observed": w.n_states_observed,
                }
            )
        return pd.DataFrame(rows).set_index("window_id")

    def plot_oos_pnl(self) -> None:
        """Concatenate OOS PnL and plot cumulative return vs baselines.

        Requires matplotlib. Silently skips if not available.
        """
        if not _MPL_AVAILABLE:
            logger.warning("matplotlib not available; cannot plot OOS PnL.")
            return

        all_oos = pd.concat([w.oos_pnl for w in self.windows]).sort_index()
        all_oos = all_oos[~all_oos.index.duplicated(keep="first")]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            all_oos.index,
            all_oos.cumsum(),
            label=f"Regime OOS (Sharpe={self.agg_sharpe:.2f})",
            linewidth=2,
        )

        for bname, bpnl in self._baseline_pnl.items():
            b_aligned = bpnl.reindex(all_oos.index).fillna(0.0)
            ax.plot(b_aligned.index, b_aligned.cumsum(), linestyle="--", label=bname)

        ax.set_title("Walk-Forward OOS Cumulative PnL")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return (fraction of notional)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RegimeWalkForward:
    """Rolling W1-fit / W2-select / W3-OOS walk-forward engine for HMM regimes.

    Parameters
    ----------
    fit_months : int
        W1 window length in calendar months.
    select_months : int
        W2 window length in calendar months.
    oos_months : int
        W3 (OOS) window length and roll step in calendar months.
    hmm_n_states : int
        Number of HMM hidden states (passed to HMMRegimeModel).
    hmm_n_seeds : int
        Multi-seed attempts per HMM fit.
    hmm_n_iter : int
        Maximum EM iterations per HMM fit.
    hmm_hysteresis_prob : float
        Posterior probability threshold for hysteresis state transitions.
    hmm_hysteresis_bars : int
        Consecutive bars above threshold required to confirm a transition.
    selector_metric : str
        Ranking metric for StateStrategySelector.
    selector_min_obs : int
        Minimum W2 observations for a strategy to be eligible for a state.
    selector_margin : float
        Minimum Sharpe margin over fallback required to select a strategy.
    selector_bootstrap_n : int
        Bootstrap resamples used for robustness check (informational).
    selector_bootstrap_hit_rate : float
        Minimum fraction of bootstrap samples with positive Sharpe to confirm.
    fallback_strategy : str
        Strategy name used when no eligible strategy is found for a state.
    trans_prob_thresh : float
        If LightGBM transition probability exceeds this value on a given day,
        multiply exposure by ``(1 - trans_prob)`` to reduce risk at transitions.
    random_seed : int
        Master random seed.
    """

    def __init__(
        self,
        fit_months: int = 18,
        select_months: int = 6,
        oos_months: int = 3,
        hmm_n_states: int = 3,
        hmm_n_seeds: int = 15,
        hmm_n_iter: int = 300,
        hmm_hysteresis_prob: float = 0.70,
        hmm_hysteresis_bars: int = 3,
        selector_metric: str = "sharpe",
        selector_min_obs: int = 80,
        selector_margin: float = 0.30,
        selector_bootstrap_n: int = 500,
        selector_bootstrap_hit_rate: float = 0.60,
        fallback_strategy: str = "FundingCarry",
        trans_prob_thresh: float = 0.45,
        random_seed: int = 42,
    ) -> None:
        self.fit_months = fit_months
        self.select_months = select_months
        self.oos_months = oos_months
        self.hmm_n_states = hmm_n_states
        self.hmm_n_seeds = hmm_n_seeds
        self.hmm_n_iter = hmm_n_iter
        self.hmm_hysteresis_prob = hmm_hysteresis_prob
        self.hmm_hysteresis_bars = hmm_hysteresis_bars
        self.selector_metric = selector_metric
        self.selector_min_obs = selector_min_obs
        self.selector_margin = selector_margin
        self.selector_bootstrap_n = selector_bootstrap_n
        self.selector_bootstrap_hit_rate = selector_bootstrap_hit_rate
        self.fallback_strategy = fallback_strategy
        self.trans_prob_thresh = trans_prob_thresh
        self.random_seed = random_seed

        # Persistent label aligner — reset at the start of each run()
        self._label_aligner = None

        # Round-trip cost charged once per strategy switch in _simulate_oos.
        # Default: taker_fee (5 bps) + slippage (0.5 bps) each side × 2 sides.
        self._switch_cost: float = 2.0 * (0.0005 + 0.00005)

    # ------------------------------------------------------------------
    # Configuration dict
    # ------------------------------------------------------------------

    def _config(self) -> dict:
        return {
            "fit_months": self.fit_months,
            "select_months": self.select_months,
            "oos_months": self.oos_months,
            "hmm_n_states": self.hmm_n_states,
            "hmm_n_seeds": self.hmm_n_seeds,
            "hmm_n_iter": self.hmm_n_iter,
            "hmm_hysteresis_prob": self.hmm_hysteresis_prob,
            "hmm_hysteresis_bars": self.hmm_hysteresis_bars,
            "selector_metric": self.selector_metric,
            "selector_min_obs": self.selector_min_obs,
            "selector_margin": self.selector_margin,
            "selector_bootstrap_n": self.selector_bootstrap_n,
            "selector_bootstrap_hit_rate": self.selector_bootstrap_hit_rate,
            "fallback_strategy": self.fallback_strategy,
            "trans_prob_thresh": self.trans_prob_thresh,
            "random_seed": self.random_seed,
        }

    # ------------------------------------------------------------------
    # Window generation
    # ------------------------------------------------------------------

    def _generate_windows(
        self, min_date: pd.Timestamp, max_date: pd.Timestamp
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate non-overlapping W1/W2/W3 window tuples.

        Each tuple is ``(w1_start, w1_end, w2_start, w2_end, w3_start, w3_end)``.
        The start pointer advances by ``oos_months`` after each iteration.
        Windows whose W3 end exceeds ``max_date`` are dropped.
        """
        windows = []
        start = min_date
        while True:
            w1_start = start
            w1_end = start + relativedelta(months=self.fit_months)
            w2_start = w1_end
            w2_end = w2_start + relativedelta(months=self.select_months)
            w3_start = w2_end
            w3_end = w3_start + relativedelta(months=self.oos_months)

            if w3_end > max_date:
                break

            windows.append((w1_start, w1_end, w2_start, w2_end, w3_start, w3_end))
            start = start + relativedelta(months=self.oos_months)

        return windows

    # ------------------------------------------------------------------
    # HMM fit + decode helpers (standalone, no Qlib dataset)
    # ------------------------------------------------------------------

    def _fit_hmm_on_features(self, daily_features: pd.DataFrame):
        """Fit HMMRegimeModel directly on a daily feature DataFrame.

        Returns the fitted model object.
        """
        try:
            from ...contrib.model.hmm_regime import HMMRegimeModel  # noqa: F401 — check import
        except ImportError as exc:
            raise ImportError(
                "qlib.contrib.model.hmm_regime is required. "
                "Ensure hmmlearn is installed."
            ) from exc

        # Import here so the module works without the full Qlib infrastructure
        from ...contrib.model.hmm_regime import (
            _fit_hmm_best_seed,
            _bounded_mask,
            _forward_filtered,
        )
        from sklearn.preprocessing import PowerTransformer, StandardScaler

        X_raw = daily_features.values.astype(np.float64)
        cols = list(daily_features.columns)

        b_mask = _bounded_mask(cols)
        power_tfm = PowerTransformer(method="yeo-johnson", standardize=False)
        scaler = StandardScaler()

        Xt = X_raw.copy()
        if b_mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                power_tfm.fit(X_raw[:, b_mask])
                Xt[:, b_mask] = power_tfm.transform(X_raw[:, b_mask])
        scaler.fit(Xt)
        X = scaler.transform(Xt)

        _, seed, hmm_model = _fit_hmm_best_seed(
            X, self.hmm_n_states, self.hmm_n_seeds, self.hmm_n_iter
        )

        # Bundle everything needed for later decoding
        return {
            "hmm_model": hmm_model,
            "power_tfm": power_tfm,
            "scaler": scaler,
            "b_mask": b_mask,
            "cols": cols,
            "seed": seed,
        }

    def _decode_segment(
        self,
        fit_bundle: dict,
        daily_features: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward-filter decode a feature DataFrame using a fitted bundle.

        Returns
        -------
        states_filtered : np.ndarray (T,)
            Hysteresis-filtered integer state labels.
        posterior : np.ndarray (T, K)
            Forward-filtered posterior probabilities.
        trans_prob : np.ndarray (T,)
            Transition probability from LightGBM (zeros if unavailable).
        """
        from ...contrib.model.hmm_regime import _forward_filtered

        hmm_model = fit_bundle["hmm_model"]
        power_tfm = fit_bundle["power_tfm"]
        scaler = fit_bundle["scaler"]
        b_mask = fit_bundle["b_mask"]
        cols = fit_bundle["cols"]

        # Align columns
        missing = [c for c in cols if c not in daily_features.columns]
        if missing:
            raise ValueError(f"Features missing in decode segment: {missing}")
        X_raw = daily_features[cols].values.astype(np.float64)

        Xt = X_raw.copy()
        if b_mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Xt[:, b_mask] = power_tfm.transform(X_raw[:, b_mask])
        X = scaler.transform(Xt)

        posterior = _forward_filtered(hmm_model, X)
        states_raw = np.argmax(posterior, axis=1)
        states_filtered = _apply_hysteresis(
            states_raw,
            posterior,
            prob_thresh=self.hmm_hysteresis_prob,
            n_bars=self.hmm_hysteresis_bars,
        )

        # Transition probabilities (via LightGBM if available)
        lgb_trans = fit_bundle.get("lgb_trans")
        if lgb_trans is not None:
            from ...contrib.model.hmm_regime import _build_lag_features

            lags = fit_bundle.get("transition_lags", [1, 2, 3, 5])
            max_lag = max(lags)
            X_lag = _build_lag_features(X, lags)
            trans_prob = np.zeros(len(X))
            trans_prob[max_lag:] = lgb_trans.predict_proba(X_lag)[:, 1]
        else:
            trans_prob = np.zeros(len(X))

        return states_filtered, posterior, trans_prob

    # ------------------------------------------------------------------
    # HMM label aligner (graceful fallback if module absent)
    # ------------------------------------------------------------------

    def _align_labels(self, fit_bundle: dict) -> np.ndarray:
        """Align HMM state labels across windows using a persistent HMMLabelAligner.

        On the first window the model is stored as the reference and the
        identity permutation is returned.  On subsequent windows the Hungarian
        algorithm maps new emission means to the stored reference.

        The aligner instance (``self._label_aligner``) is reset at the start
        of every ``run()`` call so multiple calls are independent.

        Parameters
        ----------
        fit_bundle : dict
            As returned by ``_fit_hmm_on_features``; must contain ``"hmm_model"``.

        Returns
        -------
        perm : np.ndarray, shape (K,)
            Permutation array: ``perm[raw_state] = canonical_state``.
            Apply as ``canonical = perm[raw_states_array]``.
        """
        hmm_model = fit_bundle["hmm_model"]
        K = hmm_model.n_components

        try:
            from ...contrib.model.hmm_label_aligner import HMMLabelAligner

            if self._label_aligner is None:
                # First window: store reference, return identity
                self._label_aligner = HMMLabelAligner()
                self._label_aligner.fit(hmm_model)
                return np.arange(K)
            return self._label_aligner.align(hmm_model)

        except ImportError:
            logger.debug("hmm_label_aligner not found; using identity permutation.")
            return np.arange(K)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HMMLabelAligner failed (%s); using identity permutation.", exc)
            return np.arange(K)

    # ------------------------------------------------------------------
    # Per-window simulation
    # ------------------------------------------------------------------

    def _simulate_oos(
        self,
        w3_dates: pd.DatetimeIndex,
        states_w3: np.ndarray,
        trans_prob_w3: np.ndarray,
        state_strategy_map: Dict[int, str],
        state_risk_map: Dict[int, float],
        strategy_pnl: Dict[str, pd.Series],
    ) -> Tuple[pd.Series, int]:
        """Simulate OOS PnL for W3.

        For each day t:
        1. Look up state[t] → strategy_name.
        2. Scale by risk_degree from state_risk_map.
        3. If trans_prob[t] > trans_prob_thresh: multiply by (1 - trans_prob[t]).
        4. oos_pnl[t] = strategy_pnl[strategy_name][t] * exposure.

        Returns
        -------
        oos_pnl : pd.Series indexed by w3_dates
        switch_count : int  (number of strategy changes)
        """
        pnl_values = np.zeros(len(w3_dates))
        prev_strategy: Optional[str] = None
        switch_count = 0

        for i, date in enumerate(w3_dates):
            state = int(states_w3[i])
            strategy_name = state_strategy_map.get(state, self.fallback_strategy)
            risk_degree = state_risk_map.get(state, 0.5)

            # Transition dampening
            tp = float(trans_prob_w3[i])
            if tp > self.trans_prob_thresh:
                exposure = risk_degree * (1.0 - tp)
            else:
                exposure = risk_degree

            # Look up daily PnL for the chosen strategy
            strat_series = strategy_pnl.get(strategy_name)
            if strat_series is None:
                strat_series = strategy_pnl.get(self.fallback_strategy)
            if strat_series is not None and date in strat_series.index:
                day_pnl = float(strat_series.loc[date])
            else:
                day_pnl = 0.0

            pnl_values[i] = day_pnl * exposure

            if prev_strategy is not None and strategy_name != prev_strategy:
                switch_count += 1
                # Deduct the round-trip cost of exiting the old position and
                # entering the new one.  This is the cost that PerpSimulator
                # no longer charges daily (since positions are held open).
                pnl_values[i] -= self._switch_cost
            prev_strategy = strategy_name

        oos_pnl = pd.Series(pnl_values, index=w3_dates, name="oos_pnl")
        return oos_pnl, switch_count

    # ------------------------------------------------------------------
    # Per-window runner
    # ------------------------------------------------------------------

    def _run_window(
        self,
        window_id: int,
        w1_start: pd.Timestamp,
        w1_end: pd.Timestamp,
        w2_start: pd.Timestamp,
        w2_end: pd.Timestamp,
        w3_start: pd.Timestamp,
        w3_end: pd.Timestamp,
        features_df: pd.DataFrame,
        strategy_returns_factory: Callable,
    ) -> WindowResult:
        """Execute one walk-forward window and return (WindowResult, label_map)."""
        from ...contrib.strategy.state_strategy_selector import StateStrategySelector

        daily_features = _extract_daily_features(features_df)

        # ---- W1: Fit HMM ----
        w1_mask = (daily_features.index >= w1_start) & (daily_features.index < w1_end)
        w1_features = daily_features.loc[w1_mask].dropna()
        if len(w1_features) < 30:
            raise ValueError(
                f"Window {window_id}: W1 has only {len(w1_features)} rows "
                f"(need ≥ 30). Check date range [{w1_start}, {w1_end})."
            )

        fit_bundle = self._fit_hmm_on_features(w1_features)

        # ---- Align state labels across windows ----
        perm = self._align_labels(fit_bundle)

        # ---- W2: Decode + select strategies ----
        w2_mask = (daily_features.index >= w2_start) & (daily_features.index < w2_end)
        w2_features = daily_features.loc[w2_mask].dropna()
        if len(w2_features) == 0:
            raise ValueError(
                f"Window {window_id}: W2 has 0 rows in [{w2_start}, {w2_end})."
            )

        states_w2, _, _ = self._decode_segment(fit_bundle, w2_features)
        # Apply label alignment permutation
        states_w2_aligned = perm[states_w2]
        states_w2_series = pd.Series(
            states_w2_aligned, index=w2_features.index, name="state"
        )

        w2_strategy_pnl = strategy_returns_factory(w2_start, w2_end)

        # All three guards (min_obs, margin, bootstrap) are enforced inside
        # StateStrategySelector._select_for_state() — no manual re-application here.
        selector = StateStrategySelector(
            metric=self.selector_metric,
            min_obs=self.selector_min_obs,
            annualization=365,
            margin=self.selector_margin,
            bootstrap_n=self.selector_bootstrap_n,
            bootstrap_hit_rate=self.selector_bootstrap_hit_rate,
        )
        selector.fit(
            states_w2_series,
            w2_strategy_pnl,
            fallback=self.fallback_strategy,
        )

        state_strategy_map = selector.state_strategy_map
        report = selector.report()

        state_risk_map = selector.state_risk_map()

        # ---- W3: Decode + simulate OOS ----
        w3_mask = (daily_features.index >= w3_start) & (daily_features.index < w3_end)
        w3_features = daily_features.loc[w3_mask].dropna()
        if len(w3_features) == 0:
            raise ValueError(
                f"Window {window_id}: W3 has 0 rows in [{w3_start}, {w3_end})."
            )

        states_w3, _, trans_prob_w3 = self._decode_segment(fit_bundle, w3_features)
        states_w3_aligned = perm[states_w3]
        w3_dates = w3_features.index

        w3_strategy_pnl = strategy_returns_factory(w3_start, w3_end)

        oos_pnl, switch_count = self._simulate_oos(
            w3_dates=w3_dates,
            states_w3=states_w3_aligned,
            trans_prob_w3=trans_prob_w3,
            state_strategy_map=state_strategy_map,
            state_risk_map=state_risk_map,
            strategy_pnl=w3_strategy_pnl,
        )

        oos_sharpe = _compute_sharpe(oos_pnl)
        oos_calmar = _compute_calmar(oos_pnl)
        oos_max_dd = _compute_max_dd(oos_pnl)
        clean_oos = oos_pnl.dropna()
        oos_win_rate = float((clean_oos > 0).mean()) if len(clean_oos) else 0.0
        n_states_observed = int(pd.Series(states_w3_aligned).nunique())

        result = WindowResult(
            window_id=window_id,
            w1_start=w1_start,
            w3_end=w3_dates[-1] if len(w3_dates) else w3_end,
            state_strategy_map=state_strategy_map,
            state_risk_map=state_risk_map,
            oos_pnl=oos_pnl,
            oos_sharpe=oos_sharpe,
            oos_calmar=oos_calmar,
            oos_max_dd=oos_max_dd,
            oos_win_rate=oos_win_rate,
            switch_count=switch_count,
            n_states_observed=n_states_observed,
            selector_report=report,
        )
        return result

    # ------------------------------------------------------------------
    # Public: run
    # ------------------------------------------------------------------

    def run(
        self,
        features_df: pd.DataFrame,
        strategy_returns_factory: Callable,
        baselines: Optional[Dict[str, pd.Series]] = None,
    ) -> WalkForwardResult:
        """Execute the full walk-forward protocol.

        Parameters
        ----------
        features_df : pd.DataFrame
            Daily features with MultiIndex (datetime, instrument) or
            DatetimeIndex if single-asset. Must span the full analysis period.
        strategy_returns_factory : callable
            ``f(start_date, end_date) -> Dict[str, pd.Series]``
            Returns strategy daily PnL for the specified date range.
        baselines : dict, optional
            ``{name: pd.Series}`` of baseline daily PnL covering the full OOS
            period. Used to compute ``hit_rate_vs_baseline``.

        Returns
        -------
        WalkForwardResult
        """
        if baselines is None:
            baselines = {}

        # Determine date range from features
        daily_features = _extract_daily_features(features_df)
        min_date = daily_features.index.min()
        max_date = daily_features.index.max()

        windows = self._generate_windows(min_date, max_date)
        if not windows:
            raise ValueError(
                "No complete windows fit within the available data range "
                f"[{min_date.date()}, {max_date.date()}] with "
                f"W1={self.fit_months}m + W2={self.select_months}m + "
                f"W3={self.oos_months}m. Provide more data."
            )

        logger.info(
            "RegimeWalkForward.run: %d windows | data [%s, %s] | "
            "W1=%dm W2=%dm W3=%dm",
            len(windows),
            min_date.date(),
            max_date.date(),
            self.fit_months,
            self.select_months,
            self.oos_months,
        )

        # Reset persistent aligner so each run() is independent
        self._label_aligner = None

        window_results: List[WindowResult] = []

        for idx, (w1_start, w1_end, w2_start, w2_end, w3_start, w3_end) in enumerate(windows):
            window_id = idx + 1
            try:
                wr = self._run_window(
                    window_id=window_id,
                    w1_start=w1_start,
                    w1_end=w1_end,
                    w2_start=w2_start,
                    w2_end=w2_end,
                    w3_start=w3_start,
                    w3_end=w3_end,
                    features_df=features_df,
                    strategy_returns_factory=strategy_returns_factory,
                )
                window_results.append(wr)
                logger.info(
                    "Window %d/%d [%s → %s]: OOS Sharpe=%.3f  Calmar=%.3f  "
                    "MaxDD=%.4f  WinRate=%.2f  Switches=%d  States=%d",
                    window_id,
                    len(windows),
                    w1_start.date(),
                    w3_end.date(),
                    wr.oos_sharpe,
                    wr.oos_calmar,
                    wr.oos_max_dd,
                    wr.oos_win_rate,
                    wr.switch_count,
                    wr.n_states_observed,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Window %d/%d [%s → %s] FAILED: %s",
                    window_id,
                    len(windows),
                    w1_start.date(),
                    w3_end.date(),
                    exc,
                    exc_info=True,
                )

        if not window_results:
            raise RuntimeError("All walk-forward windows failed. Check logs for details.")

        result = WalkForwardResult(windows=window_results, config=self._config())
        result._set_baseline_pnl(baselines)
        result._compute_hit_rates(baselines)

        # Record in multiple-testing ledger (guards against repeated tuning)
        RegimeWalkForward.log_trial(self._config(), result)

        return result

    # ------------------------------------------------------------------
    # Leakage check
    # ------------------------------------------------------------------

    def leakage_check(
        self,
        features_df: pd.DataFrame,
        strategy_returns_factory: Callable,
    ) -> None:
        """Run the protocol twice: once normally, once with features shifted +1 day.

        A +1-day shift injects future information. If the shifted run consistently
        improves OOS Sharpe across windows, it indicates look-ahead leakage in the
        normal run as well.

        Also asserts that no W3 date appears in any W1 or W2 window.

        Raises
        ------
        LeakageWarning
            If shifted Sharpe > normal Sharpe in ≥ 75 % of windows, implying
            the system may be leaking future information even without the shift.
        AssertionError
            If any W3 date appears in a W1 or W2 window.
        """
        logger.info("LeakageCheck: running normal protocol ...")
        normal_result = self.run(features_df, strategy_returns_factory)

        # Date non-overlap assertion
        daily_features = _extract_daily_features(features_df)
        windows = self._generate_windows(
            daily_features.index.min(), daily_features.index.max()
        )
        for w1_start, w1_end, w2_start, w2_end, w3_start, w3_end in windows:
            w3_mask = (daily_features.index >= w3_start) & (daily_features.index < w3_end)
            w3_dates = set(daily_features.index[w3_mask])

            w1_mask = (daily_features.index >= w1_start) & (daily_features.index < w1_end)
            w1_dates = set(daily_features.index[w1_mask])
            w2_mask = (daily_features.index >= w2_start) & (daily_features.index < w2_end)
            w2_dates = set(daily_features.index[w2_mask])

            overlap_w1 = w3_dates & w1_dates
            overlap_w2 = w3_dates & w2_dates
            assert not overlap_w1, (
                f"LeakageCheck: {len(overlap_w1)} W3 dates appear in W1 "
                f"[{w1_start.date()}, {w1_end.date()})"
            )
            assert not overlap_w2, (
                f"LeakageCheck: {len(overlap_w2)} W3 dates appear in W2 "
                f"[{w2_start.date()}, {w2_end.date()})"
            )

        # Shifted-features run
        logger.info("LeakageCheck: running shifted (+1 day) protocol ...")
        shifted_features = features_df.copy()
        if isinstance(shifted_features.index, pd.MultiIndex):
            # Shift date level forward by 1 business day
            level_name = "datetime"
            try:
                level_pos = shifted_features.index.names.index(level_name)
            except ValueError:
                level_pos = 0
            new_index = shifted_features.index.set_levels(
                shifted_features.index.levels[level_pos] + pd.Timedelta(days=1),
                level=level_pos,
            )
            shifted_features.index = new_index
        else:
            shifted_features.index = shifted_features.index + pd.Timedelta(days=1)

        try:
            shifted_result = self.run(shifted_features, strategy_returns_factory)
        except Exception as exc:
            logger.warning("LeakageCheck: shifted run failed (%s); skipping comparison.", exc)
            return

        n_windows = min(len(normal_result.windows), len(shifted_result.windows))
        if n_windows == 0:
            return

        shifted_wins = sum(
            1
            for nw, sw in zip(normal_result.windows, shifted_result.windows)
            if sw.oos_sharpe > nw.oos_sharpe
        )
        hit_rate = shifted_wins / n_windows
        logger.info(
            "LeakageCheck: shifted run beats normal in %d/%d windows (%.0f%%).",
            shifted_wins,
            n_windows,
            100 * hit_rate,
        )
        if hit_rate >= 0.75:
            warnings.warn(
                f"LeakageWarning: shifted features (future-injected) beat normal "
                f"in {hit_rate:.0%} of windows. This may indicate look-ahead "
                f"leakage in the pipeline. Investigate feature construction.",
                LeakageWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Multiple-testing ledger
    # ------------------------------------------------------------------

    @classmethod
    def log_trial(cls, config: dict, result: "WalkForwardResult") -> None:
        """Append a trial record to ``~/.qlib/regime_trials.csv``.

        Computes a Bonferroni-corrected p-value using the walk-forward
        t-statistic (``WalkForwardResult.sharpe_tstat``) and a t-distribution
        with ``n_windows - 1`` degrees of freedom.  This guards against
        p-value inflation from repeated hyperparameter searches.

        Parameters
        ----------
        config : dict
            Hyperparameter configuration (will be hashed for deduplication).
        result : WalkForwardResult
            Full result object; ``agg_sharpe`` and ``sharpe_tstat`` are used.
        """
        result_sharpe = result.agg_sharpe
        ledger_path = Path.home() / ".qlib" / "regime_trials.csv"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)

        config_str = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]  # noqa: S324

        import datetime as _dt

        timestamp = _dt.datetime.utcnow().isoformat(timespec="seconds")
        row = {
            "timestamp": timestamp,
            "config_hash": config_hash,
            "sharpe": round(result_sharpe, 6),
        }

        file_exists = ledger_path.exists()
        with ledger_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["timestamp", "config_hash", "sharpe"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        # Count total trials for Bonferroni correction
        with ledger_path.open("r") as fh:
            # Subtract 1 for the header row
            n_trials = max(1, sum(1 for _ in fh) - 1)

        # One-sided p-value from the walk-forward t-statistic.
        # t ~ t(df = n_windows - 1) under H0: mean Sharpe = 0.
        try:
            from scipy.stats import t as t_dist

            t_stat = result.sharpe_tstat
            n_windows = len(result.windows)
            df = max(1, n_windows - 1)
            raw_p = float(t_dist.sf(t_stat, df=df))
            bonferroni_p = min(1.0, raw_p * n_trials)
        except (ImportError, AttributeError):
            bonferroni_p = float("nan")

        logger.info(
            "Trial logged: hash=%s  sharpe=%.4f  n_trials=%d  "
            "Bonferroni-p=%.4f  ledger=%s",
            config_hash,
            result_sharpe,
            n_trials,
            bonferroni_p,
            ledger_path,
        )
