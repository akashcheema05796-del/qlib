# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Daily PnL simulators for crypto derivatives backtesting.

Each simulator returns a ``pd.Series`` with a ``DatetimeIndex`` and ``float``
values representing **daily PnL as a fraction of notional**.  The series is
directly consumable by ``StateStrategySelector.fit(states, strategy_returns)``.

Design principles
-----------------
* All inputs are strictly historical at time *t* — no look-ahead bias.
* Trading fees and slippage are baked into every PnL calculation so callers
  need not pass a separate fee parameter.
* All inputs must be pre-aligned ``pd.Series`` on a common ``DatetimeIndex``.

Perp strategies (:class:`PerpSimulator`)
-----------------------------------------
Use real funding-rate history.  Valid for any date range where price and
funding data are available.

Option strategies (:class:`OptionSimulator`)
---------------------------------------------
Use Black-76 (futures-options) pricing with Deribit DVOL as the implied
volatility input.  **Option strategies are only valid from 2021-07 onward
when DVOL data is available.**  Pass ``dvol=None`` to obtain a NaN series
for unsupported periods.

Expiry convention: weekly Deribit-style Friday settlements.  Strategies
enter on the first trading day of each week (Monday where possible) and
expire on the following Friday.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Module-level Black-76 helpers
# ---------------------------------------------------------------------------


def _d1_d2(
    F: float,
    K: float,
    T: float,
    sigma: float,
) -> Tuple[float, float]:
    """Compute (d1, d2) for Black-76.

    Parameters
    ----------
    F : float
        Forward / futures price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    sigma : float
        Annualised implied volatility (as a decimal, e.g. 0.80 for 80%).

    Returns
    -------
    (d1, d2) : tuple[float, float]
        Black-76 d1 and d2 values.  Returns (0.0, 0.0) for degenerate inputs
        (T ≤ 0 or sigma ≤ 0).
    """
    if T <= 0.0 or sigma <= 0.0 or F <= 0.0 or K <= 0.0:
        return 0.0, 0.0
    vol_sqrt_T = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    return d1, d2


def _black76_call(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-76 European call price.

    Parameters
    ----------
    F : float
        Forward / futures price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    sigma : float
        Annualised implied volatility (decimal).
    r : float, default 0.0
        Risk-free rate (effectively 0 for crypto).

    Returns
    -------
    float
        Call price.  Returns 0.0 for degenerate inputs (T ≤ 0, sigma ≤ 0).
    """
    if T <= 0.0 or sigma <= 0.0 or F <= 0.0 or K <= 0.0:
        # Intrinsic value only
        return max(F - K, 0.0) * math.exp(-r * max(T, 0.0))
    d1, d2 = _d1_d2(F, K, T, sigma)
    discount = math.exp(-r * T)
    return discount * (F * norm.cdf(d1) - K * norm.cdf(d2))


def _black76_put(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-76 European put price via put-call parity.

    Parameters
    ----------
    F : float
        Forward / futures price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    sigma : float
        Annualised implied volatility (decimal).
    r : float, default 0.0
        Risk-free rate.

    Returns
    -------
    float
        Put price.  Returns 0.0 for degenerate inputs.
    """
    if T <= 0.0 or sigma <= 0.0 or F <= 0.0 or K <= 0.0:
        return max(K - F, 0.0) * math.exp(-r * max(T, 0.0))
    call = _black76_call(F, K, T, sigma, r)
    discount = math.exp(-r * T)
    return call - discount * (F - K)


def _black76_delta_call(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-76 delta of a European call with respect to the forward price F.

    Parameters
    ----------
    F : float
        Forward / futures price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    sigma : float
        Annualised implied volatility (decimal).
    r : float, default 0.0
        Risk-free rate.

    Returns
    -------
    float
        Call delta.  Returns 0.5 as a neutral fallback for degenerate inputs.
    """
    if T <= 0.0 or sigma <= 0.0 or F <= 0.0 or K <= 0.0:
        return 1.0 if F > K else 0.0
    d1, _ = _d1_d2(F, K, T, sigma)
    return math.exp(-r * T) * norm.cdf(d1)


def _black76_delta_put(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-76 delta of a European put with respect to the forward price F.

    Parameters
    ----------
    F : float
        Forward / futures price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    sigma : float
        Annualised implied volatility (decimal).
    r : float, default 0.0
        Risk-free rate.

    Returns
    -------
    float
        Put delta.  Negative for standard puts (range: [-1, 0]).
    """
    return _black76_delta_call(F, K, T, sigma, r) - math.exp(-r * T)


# ---------------------------------------------------------------------------
# PerpSimulator
# ---------------------------------------------------------------------------


class PerpSimulator:
    """Daily PnL simulator for BTC/ETH perpetual futures strategies.

    Uses real funding rate history.  PnL is expressed as a fraction of
    notional so that it integrates directly with
    ``StateStrategySelector.fit(states, strategy_returns)``.

    Parameters
    ----------
    taker_fee : float, default 0.0005
        Taker fee per side (5 bps).
    slippage : float, default 0.00005
        One-way slippage per side (0.5 bps).

    Examples
    --------
    >>> sim = PerpSimulator()
    >>> pnl = sim.long_perp(prices, funding_daily)
    >>> pnl = sim.short_perp(prices, funding_daily)
    >>> pnl = sim.funding_carry(prices, funding_daily, min_funding_ann=0.10)
    >>> pnl = sim.flat(prices.index)
    """

    def __init__(
        self,
        taker_fee: float = 0.0005,
        slippage: float = 0.00005,
    ) -> None:
        self.taker_fee = taker_fee
        self.slippage = slippage

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _round_trip_cost(self) -> float:
        """Total round-trip transaction cost (entry + exit, both sides).

        Returns
        -------
        float
            2 * (taker_fee + slippage).
        """
        return 2.0 * (self.taker_fee + self.slippage)

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def long_perp(
        self,
        prices: pd.Series,
        funding_daily: pd.Series,
    ) -> pd.Series:
        """Daily PnL for a rolling long perpetual position.

        Opens at close price on day *t*, closes at close price on day *t+1*.
        The position is re-opened each day (fully rolled), so round-trip fees
        are charged every day.

        Parameters
        ----------
        prices : pd.Series
            Daily close prices with a ``DatetimeIndex``.
        funding_daily : pd.Series
            Daily funding rate paid by longs (positive → long pays short).
            Must be aligned to ``prices.index``.

        Returns
        -------
        pd.Series
            Daily PnL as fraction of notional.  The final date in the index
            is always ``NaN`` because there is no *t+1* price available.

        Notes
        -----
        PnL_t = (price[t+1] / price[t] - 1) - funding_daily[t]
                - 2*(taker_fee + slippage)
        """
        prices = prices.copy()
        funding_daily = funding_daily.reindex(prices.index)

        price_return = prices.shift(-1) / prices - 1.0
        cost = self._round_trip_cost()
        pnl = price_return - funding_daily - cost

        # Last day has no t+1 price → NaN (shift already produces NaN there)
        pnl.name = "long_perp"
        return pnl

    def short_perp(
        self,
        prices: pd.Series,
        funding_daily: pd.Series,
    ) -> pd.Series:
        """Daily PnL for a rolling short perpetual position.

        Mirror of :meth:`long_perp`.  A short position *receives* funding when
        ``funding_daily > 0`` (longs pay, shorts receive).

        Parameters
        ----------
        prices : pd.Series
            Daily close prices with a ``DatetimeIndex``.
        funding_daily : pd.Series
            Daily funding rate paid by longs (positive → short receives).
            Must be aligned to ``prices.index``.

        Returns
        -------
        pd.Series
            Daily PnL as fraction of notional.  Final date is ``NaN``.

        Notes
        -----
        PnL_t = -(price[t+1] / price[t] - 1) + funding_daily[t]
                - 2*(taker_fee + slippage)
        """
        prices = prices.copy()
        funding_daily = funding_daily.reindex(prices.index)

        price_return = prices.shift(-1) / prices - 1.0
        cost = self._round_trip_cost()
        pnl = -price_return + funding_daily - cost

        pnl.name = "short_perp"
        return pnl

    def funding_carry(
        self,
        prices: pd.Series,
        funding_daily: pd.Series,
        min_funding_ann: float = 0.10,
    ) -> pd.Series:
        """Daily PnL for a delta-hedged funding carry strategy.

        Harvests funding regardless of direction: goes long when
        ``funding_daily > 0`` (collect) and short when ``funding_daily < 0``
        (also collect), hedged with an offsetting spot/perp position.

        Only trades on days when the annualised funding exceeds
        ``min_funding_ann`` in absolute value; otherwise returns 0 (flat,
        no fees charged).

        Parameters
        ----------
        prices : pd.Series
            Daily close prices with a ``DatetimeIndex``.  Used only for index
            alignment; the strategy is delta-neutral so price direction does
            not affect PnL directly.
        funding_daily : pd.Series
            Daily funding rate.  Must be aligned to ``prices.index``.
        min_funding_ann : float, default 0.10
            Minimum annualised funding rate (10%) required to enter a position.
            Below this threshold the day is marked as flat (PnL = 0).

        Returns
        -------
        pd.Series
            Daily PnL as fraction of notional.

        Notes
        -----
        ann_funding_t = funding_daily[t] * 365

        PnL_t = |funding_daily[t]| - 2*(taker_fee + slippage)
                if |ann_funding_t| >= min_funding_ann, else 0
        """
        funding_daily = funding_daily.reindex(prices.index)

        ann_funding = funding_daily.abs() * 365.0
        threshold = min_funding_ann
        cost = self._round_trip_cost()

        pnl = funding_daily.abs() - cost
        # Zero out days where carry is below threshold
        pnl = pnl.where(ann_funding >= threshold, other=0.0)

        pnl.name = "funding_carry"
        return pnl

    def flat(self, index: pd.DatetimeIndex) -> pd.Series:
        """Return a zero PnL series (no position held).

        Parameters
        ----------
        index : pd.DatetimeIndex
            Date range for the series.

        Returns
        -------
        pd.Series
            All-zero series with the provided index.
        """
        return pd.Series(0.0, index=index, name="flat_perp", dtype=float)


# ---------------------------------------------------------------------------
# OptionSimulator
# ---------------------------------------------------------------------------


class OptionSimulator:
    """Daily PnL simulator for crypto options using the Black-76 model.

    Uses Deribit DVOL as the implied volatility input.  Expiries follow the
    Deribit weekly-Friday convention.  Strategies enter on the first available
    trading day each week (Monday where possible) and expire on the Friday
    of the same week.

    PnL values are expressed as a fraction of notional (1 BTC equivalent).

    .. note::
        **Option strategies are only valid from 2021-07 onward when DVOL data
        is available.**  Pass ``dvol=None`` to obtain a NaN series for
        unsupported periods.

    Parameters
    ----------
    taker_fee_per_leg : float, default 0.0003
        Taker fee charged per option leg (3 bps).
    spread_iv_points : float, default 0.15
        Half-spread on ATM IV in vol-point units (e.g. 0.15 = 0.15 vol points).
        Applied symmetrically — buyers pay ask, sellers receive bid.
    risk_free_rate : float, default 0.0
        Continuously compounded risk-free rate.  Effectively 0 for crypto.

    Examples
    --------
    >>> sim = OptionSimulator()
    >>> pnl = sim.short_straddle(prices, dvol)
    >>> pnl = sim.long_straddle(prices, dvol)
    >>> pnl = sim.iron_condor(prices, dvol)
    >>> pnl = sim.bull_put_spread(prices, dvol)
    >>> pnl = sim.flat(prices.index)
    """

    # Hedge rebalance cost per round trip (limit orders on perp for delta hedge)
    _HEDGE_SLIPPAGE: float = 0.00005  # 0.5 bps one-way; no taker fee (limit order)

    def __init__(
        self,
        taker_fee_per_leg: float = 0.0003,
        spread_iv_points: float = 0.15,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.taker_fee_per_leg = taker_fee_per_leg
        self.spread_iv_points = spread_iv_points
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _next_friday(date: pd.Timestamp) -> pd.Timestamp:
        """Return the next Friday on or after *date*.

        Parameters
        ----------
        date : pd.Timestamp
            Reference date.

        Returns
        -------
        pd.Timestamp
            The first Friday that is >= *date*.
        """
        # weekday(): Monday=0, Friday=4
        days_ahead = (4 - date.weekday()) % 7
        return date + pd.Timedelta(days=days_ahead)

    def _get_weekly_windows(
        self,
        index: pd.DatetimeIndex,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Return (entry_date, expiry_date) pairs for each Monday–Friday window.

        Entry is the first trading day of each calendar week present in
        *index* (Monday if available, otherwise the next earlier trading day
        in that week).  Expiry is the Friday of the same calendar week, or
        the latest trading day on/before that Friday that is in *index*.

        Parameters
        ----------
        index : pd.DatetimeIndex
            Trading calendar (dates on which prices are available).

        Returns
        -------
        list of (entry_date, expiry_date) tuples
            Sorted chronologically.  Weeks where no valid (entry, expiry)
            pair can be formed are omitted.
        """
        if len(index) == 0:
            return []

        index_set = set(index)
        index_sorted = sorted(index)

        # Group dates by ISO week
        # iso_calendar returns (year, week, weekday)
        def _iso_week_key(ts: pd.Timestamp) -> Tuple[int, int]:
            ic = ts.isocalendar()
            return (ic[0], ic[1])  # (year, week_number)

        from itertools import groupby

        windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

        for _key, group_iter in groupby(index_sorted, key=_iso_week_key):
            week_dates = sorted(group_iter)

            # Entry: earliest date in the week
            entry = week_dates[0]

            # Expiry: Friday of that calendar week (ISO weekday 5 = Friday)
            # Derive Friday from entry's ISO week
            # ISO week starts on Monday; Friday is +4 days from Monday
            iso = entry.isocalendar()
            # Monday of this ISO week
            monday = entry - pd.Timedelta(days=iso[2] - 1)
            friday = monday + pd.Timedelta(days=4)

            # Find the trading day on/before Friday that is in the index
            if friday in index_set:
                expiry = friday
            else:
                # Walk back up to 3 days to find last trading day <= Friday
                expiry = None
                for offset in range(1, 4):
                    candidate = friday - pd.Timedelta(days=offset)
                    if candidate in index_set:
                        expiry = candidate
                        break

            if expiry is None or expiry <= entry:
                # Cannot form a valid window
                continue

            windows.append((entry, expiry))

        return windows

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_dvol(
        self,
        prices: pd.Series,
        dvol: Optional[pd.Series],
    ) -> Optional[pd.Series]:
        """Reindex *dvol* to *prices* index; return None if dvol is None."""
        if dvol is None:
            return None
        return dvol.reindex(prices.index)

    def _nan_series(self, index: pd.DatetimeIndex, name: str) -> pd.Series:
        """Return an all-NaN series (used when dvol is None)."""
        return pd.Series(np.nan, index=index, name=name, dtype=float)

    def _hedge_cost_per_rebalance(self) -> float:
        """One-rebalance cost for delta hedging (limit orders, no taker fee)."""
        return 2.0 * self._HEDGE_SLIPPAGE

    # ------------------------------------------------------------------
    # Option strategy methods
    # ------------------------------------------------------------------

    def short_straddle(
        self,
        prices: pd.Series,
        dvol: Optional[pd.Series],
    ) -> pd.Series:
        """Weekly short ATM straddle with daily delta hedging.

        Every Monday (or first trading day of the week), sells one ATM call
        and one ATM put struck at K = current price.  The position is delta-
        hedged daily using limit-order perp rebalances and settled at the
        Friday close.

        Parameters
        ----------
        prices : pd.Series
            Daily close prices (used as forward price F in Black-76).
        dvol : pd.Series or None
            Deribit DVOL index in percentage points (e.g. 80.0 → 80% vol).
            Pass ``None`` to receive an all-NaN series.

        Returns
        -------
        pd.Series
            Daily PnL as fraction of notional.  Only settlement days carry
            the terminal option PnL; all interim days carry only hedge costs.
            NaN on days where DVOL data is unavailable.
        """
        dvol_aligned = self._validate_dvol(prices, dvol)
        if dvol_aligned is None:
            return self._nan_series(prices.index, "short_straddle")

        pnl = pd.Series(0.0, index=prices.index, dtype=float)
        windows = self._get_weekly_windows(prices.index)

        for entry_date, expiry_date in windows:
            # Check DVOL availability on entry
            entry_iv_pct = dvol_aligned.get(entry_date)
            if entry_iv_pct is None or pd.isna(entry_iv_pct):
                # Mark all days in window as NaN
                window_dates = prices.index[
                    (prices.index >= entry_date) & (prices.index <= expiry_date)
                ]
                pnl.loc[window_dates] = np.nan
                continue

            F0 = prices.loc[entry_date]
            K = F0  # ATM strike
            sigma_entry = entry_iv_pct / 100.0 + self.spread_iv_points / 100.0
            T_entry = (expiry_date - entry_date).days / 365.0

            if T_entry <= 0.0:
                continue

            # Premium received at entry (sell call + sell put)
            premium_call = _black76_call(F0, K, T_entry, sigma_entry, self.risk_free_rate)
            premium_put = _black76_put(F0, K, T_entry, sigma_entry, self.risk_free_rate)
            total_premium = premium_call + premium_put  # as fraction of F0
            total_premium_frac = total_premium / F0

            # Entry fees: 2 legs × taker_fee_per_leg
            entry_fees = 2.0 * self.taker_fee_per_leg

            # Collect all trading days in [entry_date, expiry_date]
            window_dates = prices.index[
                (prices.index >= entry_date) & (prices.index <= expiry_date)
            ]

            # Delta hedge: track cumulative delta position and its PnL
            total_hedge_cost = 0.0
            prev_delta = 0.0  # net straddle delta (short call + short put = -delta_c + delta_p)
            prev_hedge = 0.0  # hedge position in perp (opposite of straddle delta)

            for i, day in enumerate(window_dates):
                F_day = prices.loc[day]
                iv_day = dvol_aligned.get(day)
                if pd.isna(iv_day):
                    pnl.loc[day] = np.nan
                    continue

                sigma_day = iv_day / 100.0
                T_remaining = (expiry_date - day).days / 365.0

                if day == expiry_date:
                    # Settlement: close hedge, compute intrinsic payoff
                    # Payoff of short straddle = premium - max(F-K, 0) - max(K-F, 0)
                    intrinsic = abs(F_day - K) / F0
                    # Close hedge position (round trip)
                    close_hedge_cost = abs(prev_hedge) * self._hedge_cost_per_rebalance()
                    total_hedge_cost += close_hedge_cost
                    # Terminal PnL on settlement day
                    pnl.loc[day] = total_premium_frac - intrinsic - entry_fees - total_hedge_cost
                    # Reset for next window
                    total_hedge_cost = 0.0
                    prev_delta = 0.0
                    prev_hedge = 0.0
                else:
                    # Interim: rebalance delta hedge
                    if T_remaining > 0.0 and sigma_day > 0.0:
                        d_call = _black76_delta_call(F_day, K, T_remaining, sigma_day, self.risk_free_rate)
                        d_put = _black76_delta_put(F_day, K, T_remaining, sigma_day, self.risk_free_rate)
                        # Short straddle delta = -(delta_call + delta_put) in units of F
                        straddle_delta = -(d_call + d_put)
                    else:
                        straddle_delta = 0.0

                    # Required hedge = -straddle_delta (to neutralise net delta)
                    new_hedge = -straddle_delta
                    delta_change = abs(new_hedge - prev_hedge)
                    rebalance_cost = delta_change * self._hedge_cost_per_rebalance()
                    total_hedge_cost += rebalance_cost
                    pnl.loc[day] = -rebalance_cost  # interim day: only cost
                    prev_hedge = new_hedge

        pnl.name = "short_straddle"
        return pnl

    def long_straddle(
        self,
        prices: pd.Series,
        dvol: Optional[pd.Series],
    ) -> pd.Series:
        """Weekly long ATM straddle with daily delta hedging.

        Mirror of :meth:`short_straddle`.  The buyer pays the ask spread
        (IV used = dvol[t]/100 - spread_iv_points/100) and pays premium
        upfront.

        Parameters
        ----------
        prices : pd.Series
            Daily close prices.
        dvol : pd.Series or None
            Deribit DVOL in percentage points.  Pass ``None`` for NaN series.

        Returns
        -------
        pd.Series
            Daily PnL as fraction of notional.
        """
        dvol_aligned = self._validate_dvol(prices, dvol)
        if dvol_aligned is None:
            return self._nan_series(prices.index, "long_straddle")

        pnl = pd.Series(0.0, index=prices.index, dtype=float)
        windows = self._get_weekly_windows(prices.index)

        for entry_date, expiry_date in windows:
            entry_iv_pct = dvol_aligned.get(entry_date)
            if entry_iv_pct is None or pd.isna(entry_iv_pct):
                window_dates = prices.index[
                    (prices.index >= entry_date) & (prices.index <= expiry_date)
                ]
                pnl.loc[window_dates] = np.nan
                continue

            F0 = prices.loc[entry_date]
            K = F0
            # Buyer pays ask: IV slightly lower (buys at wider spread from mid)
            sigma_entry = max(
                entry_iv_pct / 100.0 - self.spread_iv_points / 100.0,
                1e-4,
            )
            T_entry = (expiry_date - entry_date).days / 365.0

            if T_entry <= 0.0:
                continue

            premium_call = _black76_call(F0, K, T_entry, sigma_entry, self.risk_free_rate)
            premium_put = _black76_put(F0, K, T_entry, sigma_entry, self.risk_free_rate)
            total_premium = premium_call + premium_put
            total_premium_frac = total_premium / F0

            entry_fees = 2.0 * self.taker_fee_per_leg

            window_dates = prices.index[
                (prices.index >= entry_date) & (prices.index <= expiry_date)
            ]

            total_hedge_cost = 0.0
            prev_hedge = 0.0

            for day in window_dates:
                F_day = prices.loc[day]
                iv_day = dvol_aligned.get(day)
                if pd.isna(iv_day):
                    pnl.loc[day] = np.nan
                    continue

                sigma_day = iv_day / 100.0
                T_remaining = (expiry_date - day).days / 365.0

                if day == expiry_date:
                    intrinsic = abs(F_day - K) / F0
                    close_hedge_cost = abs(prev_hedge) * self._hedge_cost_per_rebalance()
                    total_hedge_cost += close_hedge_cost
                    # Long straddle: receive intrinsic, paid premium
                    pnl.loc[day] = intrinsic - total_premium_frac - entry_fees - total_hedge_cost
                    total_hedge_cost = 0.0
                    prev_hedge = 0.0
                else:
                    if T_remaining > 0.0 and sigma_day > 0.0:
                        d_call = _black76_delta_call(F_day, K, T_remaining, sigma_day, self.risk_free_rate)
                        d_put = _black76_delta_put(F_day, K, T_remaining, sigma_day, self.risk_free_rate)
                        # Long straddle delta = delta_call + delta_put
                        straddle_delta = d_call + d_put
                    else:
                        straddle_delta = 0.0

                    new_hedge = -straddle_delta
                    delta_change = abs(new_hedge - prev_hedge)
                    rebalance_cost = delta_change * self._hedge_cost_per_rebalance()
                    total_hedge_cost += rebalance_cost
                    pnl.loc[day] = -rebalance_cost
                    prev_hedge = new_hedge

        pnl.name = "long_straddle"
        return pnl

    def iron_condor(
        self,
        prices: pd.Series,
        dvol: Optional[pd.Series],
    ) -> pd.Series:
        """Weekly iron condor (1-sigma short strangle + 2-sigma wing protection).

        Sells a 1-sigma OTM strangle and buys a 2-sigma OTM strangle (wings)
        for defined-risk.  No delta hedging is needed because the condor is
        inherently directional-risk-bounded.

        Wing strikes (log-normal parameterisation):

        * ``K_call_short = F * exp(+1σ√T)``
        * ``K_put_short  = F * exp(-1σ√T)``
        * ``K_call_long  = F * exp(+2σ√T)``
        * ``K_put_long   = F * exp(-2σ√T)``

        Parameters
        ----------
        prices : pd.Series
            Daily close prices.
        dvol : pd.Series or None
            Deribit DVOL in percentage points.  Pass ``None`` for NaN series.

        Returns
        -------
        pd.Series
            Daily PnL as fraction of notional.  Only entry and expiry dates
            carry non-zero values; interim days are zero.
        """
        dvol_aligned = self._validate_dvol(prices, dvol)
        if dvol_aligned is None:
            return self._nan_series(prices.index, "iron_condor")

        pnl = pd.Series(0.0, index=prices.index, dtype=float)
        windows = self._get_weekly_windows(prices.index)

        for entry_date, expiry_date in windows:
            entry_iv_pct = dvol_aligned.get(entry_date)
            if entry_iv_pct is None or pd.isna(entry_iv_pct):
                window_dates = prices.index[
                    (prices.index >= entry_date) & (prices.index <= expiry_date)
                ]
                pnl.loc[window_dates] = np.nan
                continue

            F0 = prices.loc[entry_date]
            sigma = entry_iv_pct / 100.0
            T = (expiry_date - entry_date).days / 365.0

            if T <= 0.0:
                continue

            vol_sqrt_T = sigma * math.sqrt(T)

            # Short strangle legs (1-sigma)
            K_cs = F0 * math.exp(vol_sqrt_T)
            K_ps = F0 * math.exp(-vol_sqrt_T)
            # Wing legs (2-sigma)
            K_cl = F0 * math.exp(2.0 * vol_sqrt_T)
            K_pl = F0 * math.exp(-2.0 * vol_sqrt_T)

            # Seller pays the spread (ask IV = sigma + spread) for short legs
            sigma_sell = sigma + self.spread_iv_points / 100.0
            # Buyer of wings also pays spread
            sigma_buy = max(sigma - self.spread_iv_points / 100.0, 1e-4)

            r = self.risk_free_rate
            call_short = _black76_call(F0, K_cs, T, sigma_sell, r)
            put_short = _black76_put(F0, K_ps, T, sigma_sell, r)
            call_long = _black76_call(F0, K_cl, T, sigma_buy, r)
            put_long = _black76_put(F0, K_pl, T, sigma_buy, r)

            # Net premium received (all as fraction of F0)
            net_premium_frac = (call_short + put_short - call_long - put_long) / F0
            # 4 legs × taker_fee_per_leg
            total_fees = 4.0 * self.taker_fee_per_leg

            # Settlement payoff
            F_exp = prices.loc[expiry_date]

            # Payoff paid (short strangle payoff - long wing payoff)
            short_call_payout = max(F_exp - K_cs, 0.0)
            short_put_payout = max(K_ps - F_exp, 0.0)
            long_call_payout = max(F_exp - K_cl, 0.0)
            long_put_payout = max(K_pl - F_exp, 0.0)

            net_payout_frac = (
                short_call_payout + short_put_payout - long_call_payout - long_put_payout
            ) / F0

            pnl.loc[expiry_date] = net_premium_frac - net_payout_frac - total_fees

        pnl.name = "iron_condor"
        return pnl

    def bull_put_spread(
        self,
        prices: pd.Series,
        dvol: Optional[pd.Series],
    ) -> pd.Series:
        """Weekly bull put spread (sell ATM put, buy 10-delta put).

        Approximates the 10-delta put strike as
        ``K_10d ≈ F * exp(-0.7 * sigma * sqrt(T))``, which is a close
        approximation for typical crypto vol surfaces.

        Parameters
        ----------
        prices : pd.Series
            Daily close prices.
        dvol : pd.Series or None
            Deribit DVOL in percentage points.  Pass ``None`` for NaN series.

        Returns
        -------
        pd.Series
            Daily PnL as fraction of notional.

        Notes
        -----
        The 10-delta approximation ``K ≈ F * exp(-0.7σ√T)`` follows from
        setting N(d2) ≈ 0.10 and solving approximately for K, which gives a
        coefficient near 0.7 for d2 at 10-delta.
        """
        dvol_aligned = self._validate_dvol(prices, dvol)
        if dvol_aligned is None:
            return self._nan_series(prices.index, "bull_put_spread")

        pnl = pd.Series(0.0, index=prices.index, dtype=float)
        windows = self._get_weekly_windows(prices.index)

        for entry_date, expiry_date in windows:
            entry_iv_pct = dvol_aligned.get(entry_date)
            if entry_iv_pct is None or pd.isna(entry_iv_pct):
                window_dates = prices.index[
                    (prices.index >= entry_date) & (prices.index <= expiry_date)
                ]
                pnl.loc[window_dates] = np.nan
                continue

            F0 = prices.loc[entry_date]
            sigma = entry_iv_pct / 100.0
            T = (expiry_date - entry_date).days / 365.0

            if T <= 0.0:
                continue

            vol_sqrt_T = sigma * math.sqrt(T)

            # ATM put strike = current price
            K_atm = F0
            # 10-delta OTM put strike approximation
            K_10d = F0 * math.exp(-0.7 * vol_sqrt_T)

            r = self.risk_free_rate
            # Sell ATM put at ask (pay spread)
            sigma_sell = sigma + self.spread_iv_points / 100.0
            # Buy 10-delta put at ask (pay spread)
            sigma_buy = max(sigma + self.spread_iv_points / 100.0, 1e-4)

            atm_put = _black76_put(F0, K_atm, T, sigma_sell, r)
            otm_put = _black76_put(F0, K_10d, T, sigma_buy, r)

            net_premium_frac = (atm_put - otm_put) / F0
            total_fees = 2.0 * self.taker_fee_per_leg

            # Settlement
            F_exp = prices.loc[expiry_date]

            # Short ATM put payout, long OTM put payout
            atm_put_payout = max(K_atm - F_exp, 0.0)
            otm_put_payout = max(K_10d - F_exp, 0.0)

            net_payout_frac = (atm_put_payout - otm_put_payout) / F0

            pnl.loc[expiry_date] = net_premium_frac - net_payout_frac - total_fees

        pnl.name = "bull_put_spread"
        return pnl

    def flat(self, index: pd.DatetimeIndex) -> pd.Series:
        """Return a zero PnL series (no position held).

        Parameters
        ----------
        index : pd.DatetimeIndex
            Date range for the series.

        Returns
        -------
        pd.Series
            All-zero series with the provided index.
        """
        return pd.Series(0.0, index=index, name="flat_option", dtype=float)
