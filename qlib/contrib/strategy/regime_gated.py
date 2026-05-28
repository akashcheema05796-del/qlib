# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Regime-gated portfolio strategy for Qlib.

Wraps any weight-based signal strategy and modulates its position sizing
using a pre-computed regime DataFrame produced by HMMRegimeModel.predict().

Two gating mechanisms:
  1. **Risk-degree scaling** – each regime has a configured risk multiplier so
     "Vol_Spike" regimes automatically reduce overall exposure.
  2. **Transition-probability gate** – when the LightGBM transition classifier
     signals an imminent regime change (trans_prob > threshold), positions are
     scaled down proportionally to (1 - trans_prob).
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Union

import pandas as pd

from qlib.backtest.decision import TradeDecisionWO
from qlib.log import get_module_logger
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase

logger = get_module_logger("RegimeGatedStrategy")

# Default risk-degree per regime name.  Any unlisted regime falls back to
# base_risk_degree.  Values are fractions of total portfolio value.
DEFAULT_REGIME_RISK_MAP: Dict[str, float] = {
    "Low_Vol_Trend": 0.95,
    "Low_Vol_Range": 0.80,
    "High_Vol_Trend": 0.70,
    "Choppy": 0.50,
    "Vol_Spike": 0.20,
    # deduplicated variants
    "Low_Vol_Trend_0": 0.95,
    "Low_Vol_Range_0": 0.80,
    "High_Vol_Trend_0": 0.70,
    "Choppy_0": 0.50,
    "Vol_Spike_0": 0.20,
}


class RegimeGatedStrategy(WeightStrategyBase):
    """Portfolio strategy gated by a pre-computed market regime signal.

    Parameters
    ----------
    regime_signal : pd.DataFrame
        Output of ``HMMRegimeModel.predict()``. Must contain a ``regime``
        column and optionally ``trans_prob``. Index should be a MultiIndex
        of (datetime, instrument) or just datetime.
    regime_risk_map : dict, optional
        ``{regime_name: risk_degree}`` mapping.  Defaults to
        ``DEFAULT_REGIME_RISK_MAP``.  Any regime not listed falls back to
        ``base_risk_degree``.
    trans_prob_thresh : float
        If ``trans_prob`` exceeds this threshold, positions are scaled by
        ``(1 - trans_prob)`` to pre-empt a regime change.  Set to ``1.0``
        to disable.
    base_risk_degree : float
        Fallback risk degree used for regimes not present in
        ``regime_risk_map``.
    signal : Signal-compatible
        The base return-forecast signal (same as WeightStrategyBase).  Any
        signal type accepted by ``create_signal_from()`` is valid.

    Notes
    -----
    All other ``WeightStrategyBase`` / ``BaseSignalStrategy`` parameters
    (``topk``, ``order_generator_cls_or_obj``, etc.) are forwarded via
    ``**kwargs``.

    Example
    -------
    ::

        regime_df = regime_model.predict(dataset, segment="test")

        strategy = RegimeGatedStrategy(
            signal=(alpha_model, dataset),
            regime_signal=regime_df,
            regime_risk_map={"Low_Vol_Trend": 0.95, "Vol_Spike": 0.15},
            trans_prob_thresh=0.40,
            base_risk_degree=0.70,
        )
    """

    def __init__(
        self,
        *,
        regime_signal: pd.DataFrame,
        regime_risk_map: Optional[Dict[str, float]] = None,
        trans_prob_thresh: float = 0.40,
        base_risk_degree: float = 0.80,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._regime_signal = self._normalise_regime_signal(regime_signal)
        self._regime_risk_map = dict(DEFAULT_REGIME_RISK_MAP)
        if regime_risk_map:
            self._regime_risk_map.update(regime_risk_map)
        self._trans_prob_thresh = trans_prob_thresh
        self._base_risk_degree = base_risk_degree

        # Cached lookup for the current bar
        self._current_regime: Optional[str] = None
        self._current_trans_prob: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_regime_signal(df: pd.DataFrame) -> pd.DataFrame:
        """Accept either (datetime, instrument) MultiIndex or datetime-only index."""
        if isinstance(df.index, pd.MultiIndex):
            # Group by date and take first row (all rows for a date share the regime)
            return df.groupby(level="datetime").first()
        return df

    def _lookup_regime(self, trade_start_time) -> tuple[str, float]:
        """Return (regime_name, trans_prob) for the given bar start time."""
        try:
            row = self._regime_signal.loc[trade_start_time]
            regime = str(row["regime"]) if "regime" in row.index else "Unknown"
            trans_prob = float(row["trans_prob"]) if "trans_prob" in row.index else 0.0
        except KeyError:
            # No regime data for this date — fall back to safe defaults
            regime = "Unknown"
            trans_prob = 0.0
        return regime, trans_prob

    # ------------------------------------------------------------------
    # WeightStrategyBase overrides
    # ------------------------------------------------------------------

    def get_risk_degree(self, trade_step=None) -> float:
        """Return regime-adjusted risk degree for the current bar."""
        base = self._regime_risk_map.get(self._current_regime, self._base_risk_degree)

        # Additional scale-down when a transition is imminent
        if self._current_trans_prob > self._trans_prob_thresh:
            scale = 1.0 - self._current_trans_prob
            return base * scale

        return base

    def generate_target_weight_position(
        self,
        score,
        current,
        trade_start_time,
        trade_end_time,
    ) -> dict:
        """Compute target weights with regime-aware normalisation.

        The base weights are derived from the alpha signal (score) and
        then scaled uniformly by the regime risk degree.  Individual
        weights are not zeroed — regime gating works through the overall
        exposure level, not stock selection.
        """
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]

        # Rank-normalise so weights sum to risk_degree
        score = score.dropna()
        if score.empty:
            return {}

        # Separate long and short (support long-only by clamping to positive)
        long_scores = score[score > 0]
        if long_scores.empty:
            long_scores = score  # fall back to all if no positives

        total = long_scores.abs().sum()
        if total == 0:
            return {}

        weights = (long_scores / total).to_dict()
        return weights

    def generate_trade_decision(self, execute_result=None) -> TradeDecisionWO:
        """Override to cache regime state before delegating to parent."""
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, _ = self.trade_calendar.get_step_time(trade_step)

        self._current_regime, self._current_trans_prob = self._lookup_regime(trade_start_time)

        logger.debug(
            "Date=%s  regime=%s  trans_prob=%.3f  risk_degree=%.3f",
            trade_start_time,
            self._current_regime,
            self._current_trans_prob,
            self.get_risk_degree(trade_step),
        )

        return super().generate_trade_decision(execute_result)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def regime_summary(self) -> pd.DataFrame:
        """Return a summary of regime coverage from the loaded signal."""
        if "regime" not in self._regime_signal.columns:
            return pd.DataFrame()
        counts = self._regime_signal["regime"].value_counts()
        risk = counts.index.map(
            lambda r: self._regime_risk_map.get(r, self._base_risk_degree)
        )
        return pd.DataFrame(
            {"count": counts.values, "risk_degree": risk},
            index=counts.index,
        )
