# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Data handler for market-regime classification.

Provides a stationary OHLCV-based feature set for HMM regime detection.

Design principle: **all volatility-level features are replaced by their
rolling percentile rank** (Qlib's ``Rank(X, N)`` operator, returning a
value in [0, 1]).  Raw vol levels are non-stationary on crypto — BTC
realised vol compressed from ~100% (2018–2021) to ~40–50% (2024+), which
causes naive HMMs to map all recent data into the "low-vol" state
permanently.  Percentile ranks remove this drift.

Feature groups
--------------
- Returns: RET1/5/10/20 (log returns, already stationary)
- Vol rank: RVOL5/10/20 percentile rank, GK_VOL rank, ATR rank, BB_WIDTH rank
- Shape: BB_POS, SKEW20, KURT20 (bounded / symmetric — OK as-is)
- Volume: VOL_RATIO (already relative)
- Momentum: ROC10/20, PRICE_POS20

Crypto usage
------------
Use the Binance collector at ``scripts/data_collector/crypto_binance/collector.py``
to download OHLCV data, then point Qlib's ``provider_uri`` at the output.

Funding-rate and DVOL (implied-vol) features are **not** loaded here because
they come from separate data sources.  Merge them in your workflow after
calling ``handler.fetch_df_by_col("feature")`` if needed.

Annualisation
-------------
``StateStrategySelector`` defaults to ``annualization=365`` (crypto 24/7).
Do not override to 252.
"""

from ...data.dataset.handler import DataHandlerLP
from ...contrib.data.handler import check_transform_proc


_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]

_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf"},
    {"class": "Fillna"},
]

# Lookback window for percentile-rank features.
# 252 ≈ 1 trading year for equity; use 365 for crypto (24/7).
_RANK_WINDOW = 365


def _regime_feature_config(rank_window: int = _RANK_WINDOW):
    """Return (fields, names) for regime classification features.

    All expressions use only $open, $high, $low, $close, $volume which are
    available in any standard Qlib data source (equity or crypto).

    Vol-level features are expressed as rolling percentile ranks to ensure
    stationarity across structural vol-compression regimes.

    Parameters
    ----------
    rank_window : int
        Rolling window for percentile rank, default 365 (1 yr of daily data).
    """
    fields, names = [], []

    # -----------------------------------------------------------------------
    # 1. Log returns (stationary as-is)
    # -----------------------------------------------------------------------
    for h, tag in [(1, "RET1"), (5, "RET5"), (10, "RET10"), (20, "RET20")]:
        fields.append(f"Log($close/Ref($close,{h}))")
        names.append(tag)

    # -----------------------------------------------------------------------
    # 2. Realised volatility — as percentile rank (stationarity fix)
    # -----------------------------------------------------------------------
    # Rolling std of 1-bar log returns, ranked within past rank_window bars.
    for w, tag in [(5, "RVOL5_RANK"), (10, "RVOL10_RANK"), (20, "RVOL20_RANK")]:
        fields.append(f"Rank(Std(Log($close/Ref($close,1)),{w}),{rank_window})")
        names.append(tag)

    # Garman-Klass vol estimator — rank
    # GK = 0.5*(log(H/L))^2 - (2*ln2-1)*(log(C/O))^2
    fields.append(
        f"Rank(Power(Log($high/$low),2)*0.5"
        f"-Power(Log($close/$open),2)*0.3069,{rank_window})"
    )
    names.append("GK_VOL_RANK")

    # ATR normalized by close (14-bar) — rank
    fields.append(
        f"Rank(Mean(Greater(Greater($high-$low,"
        f"Abs($high-Ref($close,1))),"
        f"Abs($low-Ref($close,1))),14)/$close,{rank_window})"
    )
    names.append("ATR_RANK")

    # Bollinger Band width (20-bar, 2-sigma) — rank
    fields.append(
        f"Rank(4*Std($close,20)/(Mean($close,20)+1e-12),{rank_window})"
    )
    names.append("BB_WIDTH_RANK")

    # -----------------------------------------------------------------------
    # 3. Shape features (bounded / mean-reverting — OK without ranking)
    # -----------------------------------------------------------------------
    # Bollinger Band position (z-score of close within band)
    fields.append("($close-Mean($close,20))/(Std($close,20)+1e-12)")
    names.append("BB_POS")

    # Rolling skewness and kurtosis of 1-bar log returns
    fields.append("Skew(Log($close/Ref($close,1)),20)")
    names.append("SKEW20")

    fields.append("Kurt(Log($close/Ref($close,1)),20)")
    names.append("KURT20")

    # -----------------------------------------------------------------------
    # 4. Volume (already relative to own history)
    # -----------------------------------------------------------------------
    fields.append("Log(($volume+1)/(Mean($volume,20)+1))")
    names.append("VOL_RATIO")

    # -----------------------------------------------------------------------
    # 5. Momentum signals (price-level agnostic)
    # -----------------------------------------------------------------------
    fields.append("$close/Ref($close,10)-1")
    names.append("ROC10")

    fields.append("$close/Ref($close,20)-1")
    names.append("ROC20")

    # Trend strength: close position within recent high-low range
    fields.append(
        "($close-Min($low,20))/(Max($high,20)-Min($low,20)+1e-12)"
    )
    names.append("PRICE_POS20")

    return fields, names


class RegimeDataHandler(DataHandlerLP):
    """Data handler for market regime classification.

    Provides 18 stationary features derived from OHLCV data.  All
    volatility-level features are expressed as percentile ranks within a
    rolling window so the feature distribution remains stable across
    structural vol-compression regimes.

    Compatible with any standard Qlib data source (equity or crypto).

    Feature summary
    ---------------
    - Log returns at 4 horizons: RET1/5/10/20
    - Realised-vol percentile rank at 3 horizons: RVOL5/10/20_RANK
    - Garman-Klass vol percentile rank: GK_VOL_RANK
    - ATR percentile rank (14-bar): ATR_RANK
    - Bollinger Band width percentile rank: BB_WIDTH_RANK
    - Bollinger Band position (z-score): BB_POS
    - Rolling skewness and kurtosis (20-bar): SKEW20, KURT20
    - Log volume ratio vs 20-bar mean: VOL_RATIO
    - Rate of change: ROC10, ROC20
    - Price position within 20-bar range: PRICE_POS20

    Parameters
    ----------
    instruments : str or list
        Qlib instrument pool.  Must be specified explicitly — no default.
        Equity example: ``"csi500"``.
        Crypto example: ``["btcusdt", "ethusdt"]`` (Binance collector output).
    start_time : str
        Start of the data window.
    end_time : str
        End of the data window.
    freq : str
        Bar frequency.  Use ``"day"`` for daily data (equity and crypto).
    fit_start_time : str
        Start of the period used to fit processors (normalisation).
    fit_end_time : str
        End of the period used to fit processors.
    rank_window : int
        Rolling window (bars) for percentile-rank features.
        Default 365 (one year of daily crypto data).
        Use 252 for equity (trading days per year).

    Notes
    -----
    Funding-rate and DVOL features come from separate collectors
    (``scripts/data_collector/crypto_binance/funding_collector.py`` and
    ``scripts/data_collector/crypto_deribit/dvol_collector.py``).  Merge
    them in your workflow after loading this handler's feature DataFrame.

    Crypto trades 24/7 — ``StateStrategySelector`` defaults to
    ``annualization=365``; do not override to 252.
    """

    def __init__(
        self,
        instruments,
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        filter_pipe=None,
        inst_processors=None,
        rank_window: int = _RANK_WINDOW,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": _regime_feature_config(rank_window=rank_window),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            learn_processors=learn_processors,
            infer_processors=infer_processors,
            **kwargs,
        )

    @staticmethod
    def get_label_config():
        """Default label: next-bar log return."""
        return ["Ref($close,-2)/Ref($close,-1)-1"], ["LABEL0"]

    @staticmethod
    def get_feature_config(rank_window: int = _RANK_WINDOW):
        return _regime_feature_config(rank_window=rank_window)
