# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
StateStrategySelector: empirically assign the best options strategy to each HMM state.

Workflow
--------
1. Run ``HMMRegimeModel.predict()`` → per-date state integers.
2. For each candidate options strategy, compute a daily PnL / return Series
   (from vectorbt, a custom backtester, or proxy payoff estimates).
3. Call ``selector.fit(states, strategy_returns)``.
4. Read ``selector.state_strategy_map`` → ``{state_int: strategy_name}``.
5. Optionally use ``selector.state_risk_map()`` as input to
   ``RegimeGatedStrategy``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ...log import get_module_logger

logger = get_module_logger("StateStrategySelector")

_VALID_METRICS = {"sharpe", "mean", "calmar", "win_rate"}


class StateStrategySelector:
    """Select the best options strategy per HMM state from historical PnL.

    Parameters
    ----------
    metric : str
        Ranking metric: ``"sharpe"``, ``"mean"``, ``"calmar"``, or
        ``"win_rate"``.
    min_obs : int
        Minimum observations a state must have in a strategy's return series
        to be eligible.  States with fewer obs get the fallback strategy.
    annualization : int
        Trading periods per year used for Sharpe and Calmar scaling.

    Example
    -------
    ::

        regime_df = model.predict(dataset, segment="train")
        states = regime_df["state"].groupby(level="datetime").first()

        strategy_returns = {
            "IronCondor":    iron_condor_daily_pnl,
            "ShortStraddle": straddle_daily_pnl,
            "BullPutSpread": bull_spread_pnl,
            "Flat":          pd.Series(0.0, index=states.index),
        }

        selector = StateStrategySelector(metric="sharpe", min_obs=30)
        selector.fit(states, strategy_returns, fallback="Flat")

        print(selector.state_strategy_map)   # {0: "IronCondor", 1: "Flat", 2: "ShortStraddle"}
        print(selector.report())             # full per-(state, strategy) table
        risk_map = selector.state_risk_map() # {0: 0.80, 1: 0.10, 2: 0.55}
    """

    def __init__(
        self,
        metric: str = "sharpe",
        min_obs: int = 20,
        annualization: int = 252,
    ):
        if metric not in _VALID_METRICS:
            raise ValueError(f"metric must be one of {_VALID_METRICS}, got '{metric}'")
        self.metric = metric
        self.min_obs = min_obs
        self.annualization = annualization

        self._state_strategy_map: Dict[int, str] = {}
        self._report: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        states: pd.Series,
        strategy_returns: Dict[str, pd.Series],
        fallback: str = "Flat",
    ) -> "StateStrategySelector":
        """Fit the selector.

        Parameters
        ----------
        states : pd.Series[int]
            Per-date HMM state indices (datetime index).  Typically:
            ``regime_df["state"].groupby(level="datetime").first()``.
        strategy_returns : dict[str, pd.Series[float]]
            ``{strategy_name: daily_return_series}``.  Each Series should
            have a datetime index.  Missing dates are filled with NaN and
            filtered out before computing metrics.
        fallback : str
            Strategy used when a state has < ``min_obs`` valid observations.
            If not already in ``strategy_returns``, it is added as zeros.
        """
        strategy_returns = dict(strategy_returns)
        if fallback not in strategy_returns:
            strategy_returns[fallback] = pd.Series(0.0, index=states.index)

        aligned = pd.DataFrame(strategy_returns).reindex(states.index)

        state_arr = states.values.astype(float)
        unique_states = sorted(int(s) for s in np.unique(state_arr[~np.isnan(state_arr)]))

        rows = []
        for state in unique_states:
            mask = state_arr == state
            for strat_name in aligned.columns:
                rets = aligned[strat_name].values[mask]
                rets = rets[~np.isnan(rets)]
                m = len(rets)

                if m < 2:
                    rows.append({
                        "state": state, "strategy": strat_name,
                        "count": m, "mean": np.nan, "std": np.nan,
                        "sharpe": np.nan, "calmar": np.nan,
                        "win_rate": np.nan, "score": -np.inf,
                    })
                    continue

                mean = rets.mean()
                std = rets.std() + 1e-12
                sharpe = mean / std * np.sqrt(self.annualization)

                cum = (1 + rets).cumprod()
                roll_max = np.maximum.accumulate(cum)
                max_dd = abs(((cum - roll_max) / (roll_max + 1e-12)).min()) + 1e-12
                calmar = (mean * self.annualization) / max_dd

                win_rate = float((rets > 0).mean())

                score_val = {
                    "sharpe": sharpe,
                    "mean": mean,
                    "calmar": calmar,
                    "win_rate": win_rate,
                }[self.metric]
                score = score_val if m >= self.min_obs else -np.inf

                rows.append({
                    "state": state, "strategy": strat_name,
                    "count": int(m), "mean": mean, "std": std - 1e-12,
                    "sharpe": sharpe, "calmar": calmar,
                    "win_rate": win_rate, "score": score,
                })

        self._report = pd.DataFrame(rows)

        # Best strategy per state
        self._state_strategy_map = {}
        for state in unique_states:
            state_rows = self._report[self._report["state"] == state]
            best_idx = state_rows["score"].idxmax()
            best_score = state_rows.loc[best_idx, "score"]
            best_strat = (
                state_rows.loc[best_idx, "strategy"]
                if best_score > -np.inf
                else fallback
            )
            self._state_strategy_map[state] = best_strat
            logger.info(
                "State %d → %s  (n=%d, %s=%.3f)",
                state, best_strat,
                int(state_rows.loc[state_rows["strategy"] == best_strat, "count"].values[0]),
                self.metric,
                best_score if best_score > -np.inf else 0.0,
            )

        return self

    # ------------------------------------------------------------------
    # Properties / outputs
    # ------------------------------------------------------------------

    @property
    def state_strategy_map(self) -> Dict[int, str]:
        """``{state_int: best_strategy_name}`` — populated after fit()."""
        if not self._state_strategy_map:
            raise ValueError("Call fit() first.")
        return dict(self._state_strategy_map)

    def state_risk_map(
        self,
        base: float = 0.80,
        floor: float = 0.10,
    ) -> Dict[int, float]:
        """Derive a risk degree per state scaled by the best strategy's Sharpe.

        States whose best strategy has the highest Sharpe get ``base``.
        States with zero or negative Sharpe get ``floor``.
        All others are interpolated linearly between floor and base.

        Parameters
        ----------
        base : float
            Risk degree for the best-performing state.
        floor : float
            Minimum risk degree (applied to states with Sharpe <= 0).

        Returns
        -------
        dict[int, float]
            Suitable as input to ``RegimeGatedStrategy(state_risk_map=...)``.
        """
        if self._report is None:
            raise ValueError("Call fit() first.")

        sharpes: Dict[int, float] = {}
        for state, strat in self._state_strategy_map.items():
            row = self._report[
                (self._report["state"] == state) & (self._report["strategy"] == strat)
            ]
            sh = float(row["sharpe"].values[0]) if len(row) else 0.0
            sharpes[state] = sh if not np.isnan(sh) else 0.0

        clipped = {s: max(0.0, sh) for s, sh in sharpes.items()}
        max_sh = max(clipped.values()) if clipped else 1.0
        max_sh = max_sh or 1.0

        return {
            s: round(floor + (base - floor) * sh / max_sh, 4)
            for s, sh in clipped.items()
        }

    def report(self) -> pd.DataFrame:
        """Full per-(state, strategy) performance table.

        Columns: state, strategy, count, mean, std, sharpe, calmar, win_rate, score.
        """
        if self._report is None:
            raise ValueError("Call fit() first.")
        return self._report.copy()
