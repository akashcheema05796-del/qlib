# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Data handler for market-regime classification.

Provides a standard OHLCV-based feature set suitable for HMM regime detection,
covering log returns, realized volatility (Garman-Klass), Bollinger Band width,
ATR, and rolling higher moments. All features are expressed using Qlib's
expression engine — no crypto-specific data sources required.
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


def _regime_feature_config():
    """Return (fields, names) for regime classification features.

    All expressions use only $open, $high, $low, $close, $volume which are
    available in any standard Qlib data source.
    """
    fields, names = [], []

    # --- Log returns (multiple horizons) ---
    for h, tag in [(1, "RET1"), (5, "RET5"), (10, "RET10"), (20, "RET20")]:
        fields.append(f"Log($close/Ref($close,{h}))")
        names.append(tag)

    # --- Realized volatility (rolling std of 1-bar log returns) ---
    for w, tag in [(5, "RVOL5"), (10, "RVOL10"), (20, "RVOL20")]:
        fields.append(f"Std(Log($close/Ref($close,1)),{w})")
        names.append(tag)

    # --- Garman-Klass realized volatility estimator ---
    # GK = 0.5*(log(H/L))^2 - (2*ln2-1)*(log(C/O))^2
    # More efficient than simple std when OHLC is available.
    fields.append(
        "Power(Log($high/$low),2)*0.5-Power(Log($close/$open),2)*0.3069"
    )
    names.append("GK_VOL")

    # --- ATR normalized by close price (14-bar) ---
    # True range = max(H-L, |H-prev_C|, |L-prev_C|)
    fields.append(
        "Mean(Greater(Greater($high-$low,"
        "Abs($high-Ref($close,1))),"
        "Abs($low-Ref($close,1))),14)/$close"
    )
    names.append("ATR_NORM")

    # --- Bollinger Band width (20-bar, 2-sigma) ---
    # BB_width = 4 * std / mean  (= upper - lower) / mid
    fields.append("4*Std($close,20)/(Mean($close,20)+1e-12)")
    names.append("BB_WIDTH")

    # --- Bollinger Band position (z-score of price within band) ---
    fields.append("($close-Mean($close,20))/(Std($close,20)+1e-12)")
    names.append("BB_POS")

    # --- Rolling higher moments of 1-bar log returns ---
    fields.append("Skew(Log($close/Ref($close,1)),20)")
    names.append("SKEW20")

    fields.append("Kurt(Log($close/Ref($close,1)),20)")
    names.append("KURT20")

    # --- Volume ratio: current vs 20-bar mean ---
    fields.append("Log(($volume+1)/(Mean($volume,20)+1))")
    names.append("VOL_RATIO")

    # --- Price momentum signals ---
    # Rate of change
    fields.append("$close/Ref($close,10)-1")
    names.append("ROC10")

    fields.append("$close/Ref($close,20)-1")
    names.append("ROC20")

    # Trend strength proxy: close position within recent high-low range
    fields.append(
        "($close-Min($low,20))/(Max($high,20)-Min($low,20)+1e-12)"
    )
    names.append("PRICE_POS20")

    return fields, names


class RegimeDataHandler(DataHandlerLP):
    """Data handler for market regime classification.

    Provides 18 features derived from OHLCV data — compatible with any
    standard Qlib data source including equity and crypto.

    - Log returns at 4 horizons (1, 5, 10, 20 bars)
    - Realized volatility at 3 horizons (5, 10, 20 bars)
    - Garman-Klass volatility estimator
    - ATR normalized by close
    - Bollinger Band width and position
    - Rolling skew and kurtosis (20-bar)
    - Volume ratio vs 20-bar mean
    - Rate of change (10, 20 bars)
    - Price position within 20-bar range

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
        Bar frequency.  Use ``"day"`` for daily data (both equity and crypto).
    fit_start_time : str
        Start of the period used to fit processors (normalisation).
    fit_end_time : str
        End of the period used to fit processors.

    Notes
    -----
    For crypto: use the Binance collector at
    ``scripts/data_collector/crypto_binance/collector.py`` to download data,
    then initialise Qlib with ``provider_uri`` pointing at the output directory.
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
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": _regime_feature_config(),
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
        """Default label: next-bar log return (forward 2-bar over 1-bar)."""
        return ["Ref($close,-2)/Ref($close,-1)-1"], ["LABEL0"]

    @staticmethod
    def get_feature_config():
        return _regime_feature_config()
