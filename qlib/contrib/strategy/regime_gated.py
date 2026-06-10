# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
State-gated portfolio strategy for Qlib.

Wraps any weight-based signal strategy and modulates its position sizing
using a pre-computed regime DataFrame produced by HMMRegimeModel.predict().

States are raw HMM integer indices — no semantic names are assigned.
Risk degrees per state are supplied by the user (typically derived from
StateStrategySelector.state_risk_map() after empirical backtesting).

Two gating mechanisms:
  1. State-based risk degree — each HMM state has a configured multiplier.
  2. Transition-probability gate — when the LightGBM transition classifier
     signals an imminent state change (trans_prob > threshold), positions are
     scaled down by (1 - trans_prob).
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from qlib.backtest.decision import TradeDecisionWO
from qlib.log import get_module_logger
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase

logger = get_module_logger("RegimeGatedStrategy")


class RegimeGatedStrategy(WeightStrategyBase):
    """Portfolio strategy gated by HMM state index.

    Parameters
    ----------
    regime_signal : pd.DataFrame
        Output of ``HMMRegimeModel.predict()``. Must contain a ``state``
        column (int) and optionally ``trans_prob``. Index should be a
        MultiIndex of (datetime, instrument) or just datetime.
    state_risk_map : dict, optional
        ``{state_int: risk_degree}`` mapping.  Any state not listed falls
        back to ``base_risk_degree``.  Populate from
        ``StateStrategySelector.state_risk_map()`` or set manually.
    trans_prob_thresh : float
        If ``trans_prob`` exceeds this threshold, positions are scaled by
        ``(1 - trans_prob)`` to pre-empt a state change.  Set to ``1.0``
        to disable.
    base_risk_degree : float
        Fallback risk degree used for states not present in
        ``state_risk_map``.
    signal : Signal-compatible
        The base return-forecast signal (same as WeightStrategyBase).

    Example
    -------
    ::

        regime_df = regime_model.predict(dataset, segment="test")
        states = regime_df["state"].groupby(level="datetime").first()

        selector = StateStrategySelector(metric="sharpe")
        selector.fit(states, strategy_returns)

        strategy = RegimeGatedStrategy(
            signal=(alpha_model, dataset),
            regime_signal=regime_df,
            state_risk_map=selector.state_risk_map(),
            trans_prob_thresh=0.40,
        )
    """

    def __init__(
        self,
        *,
        regime_signal: pd.DataFrame,
        state_risk_map: Optional[Dict[int, float]] = None,
        trans_prob_thresh: float = 0.40,
        base_risk_degree: float = 0.80,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._regime_signal = self._normalise_regime_signal(regime_signal)
        self._state_risk_map: Dict[int, float] = dict(state_risk_map or {})
        self._trans_prob_thresh = trans_prob_thresh
        self._base_risk_degree = base_risk_degree

        self._current_state: int = -1
        self._current_trans_prob: float = 0.0

    @staticmethod
    def _normalise_regime_signal(df: pd.DataFrame) -> pd.DataFrame:
        """Accept either (datetime, instrument) MultiIndex or datetime-only index."""
        if isinstance(df.index, pd.MultiIndex):
            return df.groupby(level="datetime").first()
        return df

    def _lookup_state(self, trade_start_time) -> tuple[int, float]:
        """Return (state_int, trans_prob) for the given bar start time."""
        try:
            row = self._regime_signal.loc[trade_start_time]
            state = int(row["state"]) if "state" in row.index else -1
            trans_prob = float(row["trans_prob"]) if "trans_prob" in row.index else 0.0
        except KeyError:
            logger.warning(
                "Date %s not found in regime signal; falling back to base_risk_degree.",
                trade_start_time,
            )
            state = -1
            trans_prob = 0.0
        return state, trans_prob

    def get_risk_degree(self, trade_step=None) -> float:
        """Return state-adjusted risk degree for the current bar."""
        base = self._state_risk_map.get(self._current_state, self._base_risk_degree)
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
        """Compute target weights scaled by state risk degree."""
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]

        score = score.dropna()
        if score.empty:
            return {}

        long_scores = score[score > 0]
        if long_scores.empty:
            long_scores = score

        total = long_scores.abs().sum()
        if total == 0:
            return {}

        return (long_scores / total).to_dict()

    def generate_trade_decision(self, execute_result=None) -> TradeDecisionWO:
        """Cache state before delegating to parent."""
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, _ = self.trade_calendar.get_step_time(trade_step)

        self._current_state, self._current_trans_prob = self._lookup_state(trade_start_time)

        logger.debug(
            "Date=%s  state=%d  trans_prob=%.3f  risk_degree=%.3f",
            trade_start_time,
            self._current_state,
            self._current_trans_prob,
            self.get_risk_degree(trade_step),
        )

        return super().generate_trade_decision(execute_result)

    def state_summary(self) -> pd.DataFrame:
        """Return count + configured risk degree for each state."""
        if "state" not in self._regime_signal.columns:
            return pd.DataFrame()
        counts = self._regime_signal["state"].value_counts().sort_index()
        risk = counts.index.map(
            lambda s: self._state_risk_map.get(int(s), self._base_risk_degree)
        )
        return pd.DataFrame(
            {"count": counts.values, "risk_degree": risk},
            index=counts.index,
        )
