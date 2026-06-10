# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
StateStrategySelector: empirically assign the best options strategy to each HMM state.

Workflow
--------
1. Run ``HMMRegimeModel.predict()`` → per-date state integers.
2. For each candidate strategy, compute a daily PnL / return Series.
3. Call ``selector.fit(states, strategy_returns)``.
4. Read ``selector.state_strategy_map`` → ``{state_int: strategy_name}``.
5. Use ``selector.state_risk_map()`` as input to ``RegimeGatedStrategy``.

Statistical regularisation
--------------------------
Per-state strategy selection from daily PnL is inherently noisy.  The SE of
an annualised Sharpe from *n* daily obs is ≈ √(365/n), so with n=80 obs the
SE is still ±2.1.  Three guards are applied:

- ``min_obs``: states with fewer days than this get the fallback (default 80).
- ``margin``: the winner must beat the runner-up by at least this ΔSharpe
  (default 0.30); otherwise fall back.
- ``bootstrap_hit_rate``: block-bootstrap the selection window (block ≈
  ``bootstrap_block`` days) and require the winner to be chosen in at least
  this fraction of resamples (default 0.60); otherwise fall back.

These defaults are deliberately conservative.  Crypto regimes often have
fewer than 150 days per state in a 6-month selection window, making any
ranking statistically weak.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ...log import get_module_logger

logger = get_module_logger("StateStrategySelector")

_VALID_METRICS = {"sharpe", "mean", "calmar", "win_rate"}


# ---------------------------------------------------------------------------
# Per-strategy metric computation (shared by fit and bootstrap)
# ---------------------------------------------------------------------------

def _compute_metrics(rets: np.ndarray, annualization: int) -> dict:
    """Return dict of scalar metrics for a 1-D return array."""
    m = len(rets)
    if m < 2:
        return {"count": m, "mean": np.nan, "std": np.nan,
                "sharpe": np.nan, "calmar": np.nan, "win_rate": np.nan}

    mean = rets.mean()
    std = rets.std() + 1e-12
    sharpe = mean / std * np.sqrt(annualization)

    cum = (1 + rets).cumprod()
    roll_max = np.maximum.accumulate(cum)
    max_dd = abs(((cum - roll_max) / (roll_max + 1e-12)).min()) + 1e-12
    calmar = (mean * annualization) / max_dd

    win_rate = float((rets > 0).mean())

    return {
        "count": int(m), "mean": mean, "std": std - 1e-12,
        "sharpe": sharpe, "calmar": calmar, "win_rate": win_rate,
    }


def _block_bootstrap_winner(
    state_mask: np.ndarray,
    aligned: pd.DataFrame,
    metric: str,
    annualization: int,
    min_obs: int,
    n_resamples: int,
    block_size: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Return {strategy_name: win_fraction} from block bootstrap."""
    T = int(state_mask.sum())
    if T < block_size:
        return {}

    state_idx = np.where(state_mask)[0]
    n_blocks = max(1, T // block_size)
    win_counts: Dict[str, int] = {s: 0 for s in aligned.columns}

    for _ in range(n_resamples):
        # Sample block start indices within the state observations
        starts = rng.integers(0, max(1, T - block_size + 1), size=n_blocks)
        idx = np.concatenate([state_idx[s: s + block_size] for s in starts])[:T]

        scores = {}
        for strat in aligned.columns:
            rets = aligned[strat].values[idx]
            rets = rets[~np.isnan(rets)]
            if len(rets) < min_obs:
                scores[strat] = -np.inf
                continue
            m = _compute_metrics(rets, annualization)
            scores[strat] = m[metric] if not np.isnan(m[metric]) else -np.inf

        winner = max(scores, key=scores.__getitem__)
        if scores[winner] > -np.inf:
            win_counts[winner] += 1

    total = sum(win_counts.values()) or 1
    return {k: v / total for k, v in win_counts.items()}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StateStrategySelector:
    """Select the best options strategy per HMM state from historical PnL.

    Parameters
    ----------
    metric : str
        Ranking metric: ``"sharpe"``, ``"mean"``, ``"calmar"``, or
        ``"win_rate"``.
    min_obs : int
        Minimum observations a state must have to be eligible.
        States with fewer obs get the fallback strategy.
        Default 80 (SE of Sharpe ≈ ±2.1 at n=80).
    annualization : int
        Trading periods per year.  Default 365 (crypto, 24/7).
        Use 252 for equity.
    margin : float
        Minimum ΔSharpe (or Δmetric) the winner must beat the runner-up
        by.  If the gap is smaller, fall back to ``fallback``.
        Default 0.30.
    bootstrap_n : int
        Number of block-bootstrap resamples for stability check.
        Set to 0 to disable bootstrap (faster but less conservative).
    bootstrap_block : int
        Block size for block bootstrap, in bars.
        Should approximate mean state dwell time (~15–20 days).
    bootstrap_hit_rate : float
        Minimum fraction of bootstrap resamples in which the candidate
        winner must be selected.  Default 0.60.
    random_seed : int
        Seed for bootstrap RNG.

    Example
    -------
    ::

        regime_df = model.predict(dataset, segment="train")
        states = regime_df["state"].groupby(level="datetime").first()

        strategy_returns = {
            "IronCondor":    iron_condor_daily_pnl,
            "ShortStraddle": straddle_daily_pnl,
            "FundingCarry":  carry_daily_pnl,
            "Flat":          pd.Series(0.0, index=states.index),
        }

        selector = StateStrategySelector(metric="sharpe", min_obs=80)
        selector.fit(states, strategy_returns, fallback="FundingCarry")

        print(selector.state_strategy_map)
        # {0: "IronCondor", 1: "FundingCarry", 2: "ShortStraddle"}
        risk_map = selector.state_risk_map()
    """

    def __init__(
        self,
        metric: str = "sharpe",
        min_obs: int = 80,
        annualization: int = 365,
        margin: float = 0.30,
        bootstrap_n: int = 500,
        bootstrap_block: int = 15,
        bootstrap_hit_rate: float = 0.60,
        random_seed: int = 42,
    ):
        if metric not in _VALID_METRICS:
            raise ValueError(f"metric must be one of {_VALID_METRICS}, got '{metric}'")
        self.metric = metric
        self.min_obs = min_obs
        self.annualization = annualization
        self.margin = margin
        self.bootstrap_n = bootstrap_n
        self.bootstrap_block = bootstrap_block
        self.bootstrap_hit_rate = bootstrap_hit_rate
        self.random_seed = random_seed

        self._state_strategy_map: Dict[int, str] = {}
        self._report: Optional[pd.DataFrame] = None
        self._bootstrap_report: Optional[pd.DataFrame] = None

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
            have a datetime index.
        fallback : str
            Strategy used when all selection criteria fail.
            Added as zeros if not in ``strategy_returns``.
        """
        if not strategy_returns:
            raise ValueError("strategy_returns must not be empty.")
        strategy_returns = dict(strategy_returns)
        if fallback not in strategy_returns:
            strategy_returns[fallback] = pd.Series(0.0, index=states.index)

        aligned = pd.DataFrame(strategy_returns).reindex(states.index)
        state_arr = states.values.astype(float)
        unique_states = sorted(int(s) for s in np.unique(state_arr[~np.isnan(state_arr)]))

        rng = np.random.default_rng(self.random_seed)

        # -------------------------------------------------------------------
        # Build per-(state, strategy) metrics table
        # -------------------------------------------------------------------
        rows = []
        for state in unique_states:
            mask = state_arr == state
            for strat_name in aligned.columns:
                rets = aligned[strat_name].values[mask]
                rets = rets[~np.isnan(rets)]
                m_dict = _compute_metrics(rets, self.annualization)
                m_dict["state"] = state
                m_dict["strategy"] = strat_name
                m_dict["score"] = (
                    m_dict[self.metric]
                    if (m_dict["count"] >= self.min_obs and not np.isnan(m_dict[self.metric]))
                    else -np.inf
                )
                rows.append(m_dict)

        self._report = pd.DataFrame(rows)

        # -------------------------------------------------------------------
        # Bootstrap stability report
        # -------------------------------------------------------------------
        bootstrap_rows = []
        if self.bootstrap_n > 0:
            for state in unique_states:
                mask = state_arr == state
                hit_fracs = _block_bootstrap_winner(
                    mask, aligned,
                    self.metric, self.annualization, self.min_obs,
                    self.bootstrap_n, self.bootstrap_block, rng,
                )
                for strat, frac in hit_fracs.items():
                    bootstrap_rows.append({"state": state, "strategy": strat, "bootstrap_win_rate": frac})
        self._bootstrap_report = pd.DataFrame(bootstrap_rows) if bootstrap_rows else pd.DataFrame(
            columns=["state", "strategy", "bootstrap_win_rate"]
        )

        # -------------------------------------------------------------------
        # Select best strategy per state with guards
        # -------------------------------------------------------------------
        self._state_strategy_map = {}
        for state in unique_states:
            best_strat = self._select_for_state(state, fallback)
            self._state_strategy_map[state] = best_strat

        return self

    def _select_for_state(self, state: int, fallback: str) -> str:
        """Apply guard rules and return the selected strategy for one state."""
        state_rows = self._report[self._report["state"] == state].copy()
        state_rows = state_rows.sort_values("score", ascending=False)

        best_row = state_rows.iloc[0]
        best_score = best_row["score"]
        best_strat = best_row["strategy"]

        # Guard 1: min_obs
        if best_score == -np.inf:
            logger.info("State %d → %s (fallback: insufficient obs)", state, fallback)
            return fallback

        # Guard 2: margin over runner-up
        if len(state_rows) >= 2:
            runner_score = state_rows.iloc[1]["score"]
            gap = best_score - (runner_score if runner_score > -np.inf else best_score - self.margin - 1)
            if gap < self.margin:
                logger.info(
                    "State %d → %s (fallback: winner margin %.3f < threshold %.3f)",
                    state, fallback, gap, self.margin,
                )
                return fallback

        # Guard 3: bootstrap stability
        if self.bootstrap_n > 0 and len(self._bootstrap_report) > 0:
            bs = self._bootstrap_report[
                (self._bootstrap_report["state"] == state) &
                (self._bootstrap_report["strategy"] == best_strat)
            ]
            hit_rate = float(bs["bootstrap_win_rate"].values[0]) if len(bs) else 0.0
            if hit_rate < self.bootstrap_hit_rate:
                logger.info(
                    "State %d → %s (fallback: bootstrap hit_rate %.2f < %.2f)",
                    state, fallback, hit_rate, self.bootstrap_hit_rate,
                )
                return fallback

        n_obs = int(best_row["count"])
        logger.info(
            "State %d → %s  (n=%d, %s=%.3f)",
            state, best_strat, n_obs, self.metric, best_score,
        )
        return best_strat

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
        Others are interpolated linearly between floor and base.

        Parameters
        ----------
        base : float
            Risk degree for the best-performing state.
        floor : float
            Minimum risk degree.

        Returns
        -------
        dict[int, float]
            Suitable for ``RegimeGatedStrategy(state_risk_map=...)``.
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
        """Full per-(state, strategy) performance table."""
        if self._report is None:
            raise ValueError("Call fit() first.")
        return self._report.copy()

    def bootstrap_report(self) -> pd.DataFrame:
        """Bootstrap win-rate table per (state, strategy)."""
        if self._bootstrap_report is None:
            raise ValueError("Call fit() first.")
        return self._bootstrap_report.copy()
