# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Unit tests for the HMM regime classifier integration.

Tests are self-contained: they generate synthetic OHLCV data so that no
real Qlib data download is required.  A minimal mock of DatasetH is used
to exercise the model's fit/predict interface.
"""

import unittest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n_dates: int = 300, n_stocks: int = 3, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with MultiIndex (datetime, instrument) and OHLCV columns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    stocks = [f"SH{i:06d}" for i in range(n_stocks)]

    rows = []
    for stock in stocks:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n_dates)))
        for i, d in enumerate(dates):
            c = close[i]
            rows.append({
                "datetime": d,
                "instrument": stock,
                "close": c,
                "open": c * (1 + rng.normal(0, 0.003)),
                "high": c * (1 + abs(rng.normal(0, 0.005))),
                "low": c * (1 - abs(rng.normal(0, 0.005))),
                "volume": rng.integers(100_000, 1_000_000),
            })

    df = pd.DataFrame(rows).set_index(["datetime", "instrument"])
    return df


def _make_feature_df(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute a small set of regime features from raw OHLCV."""
    result = {}
    for instrument, grp in ohlcv.groupby(level="instrument"):
        c = grp["close"]
        h = grp["high"]
        l_ = grp["low"]
        o = grp["open"]
        v = grp["volume"]

        ret1 = np.log(c / c.shift(1))
        rvol5 = ret1.rolling(5).std()
        rvol20 = ret1.rolling(20).std()
        gk_vol = 0.5 * np.log(h / l_) ** 2 - 0.3069 * np.log(c / o) ** 2
        bb_width = 4 * c.rolling(20).std() / (c.rolling(20).mean() + 1e-12)
        skew20 = ret1.rolling(20).skew()
        kurt20 = ret1.rolling(20).kurt()

        feat = pd.DataFrame({
            "RET1": ret1, "RVOL5": rvol5, "RVOL20": rvol20,
            "GK_VOL": gk_vol, "BB_WIDTH": bb_width,
            "SKEW20": skew20, "KURT20": kurt20,
        }, index=grp.index)
        result[instrument] = feat

    return pd.concat(result.values()).sort_index()


class _MockDataset:
    """Minimal DatasetH mock that wraps pre-built DataFrames."""

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self._data = {"train": train_df, "test": test_df}

    def prepare(self, segment, col_set=None, data_key=None):
        df = self._data[segment]
        if col_set == ["feature"] or col_set == "feature":
            # Wrap in MultiIndex columns to match DataHandlerLP output
            return pd.DataFrame(
                df.values,
                index=df.index,
                columns=pd.MultiIndex.from_tuples(
                    [("feature", c) for c in df.columns]
                ),
            )
        return df


# ---------------------------------------------------------------------------
# Feature-engineering helpers (handler_regime)
# ---------------------------------------------------------------------------

class TestRegimeFeatureConfig(unittest.TestCase):
    def test_feature_config_returns_equal_length_lists(self):
        from qlib.contrib.data.handler_regime import _regime_feature_config
        fields, names = _regime_feature_config()
        self.assertEqual(len(fields), len(names))
        self.assertGreater(len(fields), 0)

    def test_no_duplicate_names(self):
        from qlib.contrib.data.handler_regime import _regime_feature_config
        _, names = _regime_feature_config()
        self.assertEqual(len(names), len(set(names)), "Duplicate feature names found")


# ---------------------------------------------------------------------------
# HMM model
# ---------------------------------------------------------------------------

class TestHMMRegimeModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import hmmlearn  # noqa
        except ImportError:
            raise unittest.SkipTest("hmmlearn not installed")

        from qlib.contrib.model.hmm_regime import HMMRegimeModel

        ohlcv = _make_ohlcv(n_dates=400)
        feat = _make_feature_df(ohlcv)

        n = len(feat.index.get_level_values("datetime").unique())
        split = int(n * 0.75)
        all_dates = feat.index.get_level_values("datetime").unique().sort_values()
        train_dates = all_dates[:split]
        test_dates = all_dates[split:]

        train_df = feat[feat.index.get_level_values("datetime").isin(train_dates)]
        test_df = feat[feat.index.get_level_values("datetime").isin(test_dates)]

        cls.dataset = _MockDataset(train_df, test_df)
        cls.model = HMMRegimeModel(n_states=3, n_seeds=3, n_iter=100, transition_horizon=3)
        cls.model.fit(cls.dataset)

    def test_fit_sets_hmm_model(self):
        self.assertIsNotNone(self.model.hmm_model)

    def test_fit_builds_regime_map(self):
        self.assertIsInstance(self.model.regime_map, dict)
        self.assertGreater(len(self.model.regime_map), 0)

    def test_predict_returns_dataframe(self):
        result = self.model.predict(self.dataset, segment="test")
        self.assertIsInstance(result, pd.DataFrame)

    def test_predict_has_required_columns(self):
        result = self.model.predict(self.dataset, segment="test")
        for col in ("state", "regime", "entropy", "top_prob", "trans_prob"):
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_predict_index_matches_input(self):
        result = self.model.predict(self.dataset, segment="test")
        # Every row in the test DataFrame should have a regime
        test_raw = self.dataset._data["test"]
        self.assertEqual(len(result), len(test_raw))

    def test_entropy_non_negative(self):
        result = self.model.predict(self.dataset, segment="test")
        self.assertTrue((result["entropy"] >= 0).all())

    def test_top_prob_in_unit_interval(self):
        result = self.model.predict(self.dataset, segment="test")
        self.assertTrue(((result["top_prob"] >= 0) & (result["top_prob"] <= 1)).all())

    def test_auto_n_states_bic_sweep(self):
        from qlib.contrib.model.hmm_regime import HMMRegimeModel
        model_auto = HMMRegimeModel(n_states="auto", max_states=3, n_seeds=2, n_iter=50)
        model_auto.fit(self.dataset)
        self.assertIsNotNone(model_auto.hmm_model)
        self.assertGreaterEqual(model_auto.hmm_model.n_components, 2)


# ---------------------------------------------------------------------------
# Regime-gated strategy (unit-level, no live backtest)
# ---------------------------------------------------------------------------

class TestRegimeGatedStrategy(unittest.TestCase):

    def _make_regime_df(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        regimes = ["Low_Vol_Trend", "Choppy", "Vol_Spike"]
        return pd.DataFrame({
            "state": range(len(dates)),
            "regime": [regimes[i % 3] for i in range(len(dates))],
            "entropy": np.random.uniform(0, 1, len(dates)),
            "top_prob": np.random.uniform(0.5, 1.0, len(dates)),
            "trans_prob": np.random.uniform(0, 0.6, len(dates)),
        }, index=dates)

    def test_risk_degree_varies_by_regime(self):
        from qlib.contrib.strategy.regime_gated import RegimeGatedStrategy, DEFAULT_REGIME_RISK_MAP

        dates = pd.date_range("2021-01-01", periods=10, freq="B")
        regime_df = self._make_regime_df(dates)

        # Manually set current regime and check risk degree
        strat = RegimeGatedStrategy.__new__(RegimeGatedStrategy)
        strat._regime_risk_map = dict(DEFAULT_REGIME_RISK_MAP)
        strat._base_risk_degree = 0.80
        strat._trans_prob_thresh = 0.40
        strat._regime_signal = regime_df

        strat._current_regime = "Low_Vol_Trend"
        strat._current_trans_prob = 0.1
        risk_trend = strat.get_risk_degree()

        strat._current_regime = "Vol_Spike"
        strat._current_trans_prob = 0.1
        risk_spike = strat.get_risk_degree()

        self.assertGreater(risk_trend, risk_spike)

    def test_high_trans_prob_reduces_risk(self):
        from qlib.contrib.strategy.regime_gated import RegimeGatedStrategy

        strat = RegimeGatedStrategy.__new__(RegimeGatedStrategy)
        strat._regime_risk_map = {}
        strat._base_risk_degree = 0.90
        strat._trans_prob_thresh = 0.40

        strat._current_regime = "Unknown"
        strat._current_trans_prob = 0.0
        risk_low = strat.get_risk_degree()

        strat._current_trans_prob = 0.80
        risk_high = strat.get_risk_degree()

        self.assertGreater(risk_low, risk_high)

    def test_regime_summary(self):
        from qlib.contrib.strategy.regime_gated import RegimeGatedStrategy

        dates = pd.date_range("2021-01-01", periods=30, freq="B")
        regime_df = self._make_regime_df(dates)

        strat = RegimeGatedStrategy.__new__(RegimeGatedStrategy)
        strat._regime_signal = RegimeGatedStrategy._normalise_regime_signal(regime_df)
        from qlib.contrib.strategy.regime_gated import DEFAULT_REGIME_RISK_MAP
        strat._regime_risk_map = dict(DEFAULT_REGIME_RISK_MAP)
        strat._base_risk_degree = 0.80

        summary = strat.regime_summary()
        self.assertIn("count", summary.columns)
        self.assertIn("risk_degree", summary.columns)


# ---------------------------------------------------------------------------
# Integration smoke test: fit + predict end-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd(unittest.TestCase):

    def test_fit_predict_pipeline(self):
        try:
            import hmmlearn  # noqa
        except ImportError:
            self.skipTest("hmmlearn not installed")

        from qlib.contrib.model.hmm_regime import HMMRegimeModel

        ohlcv = _make_ohlcv(n_dates=250)
        feat = _make_feature_df(ohlcv)

        all_dates = feat.index.get_level_values("datetime").unique().sort_values()
        split = int(len(all_dates) * 0.8)
        train_dates = all_dates[:split]
        test_dates = all_dates[split:]

        train_df = feat[feat.index.get_level_values("datetime").isin(train_dates)]
        test_df = feat[feat.index.get_level_values("datetime").isin(test_dates)]
        dataset = _MockDataset(train_df, test_df)

        model = HMMRegimeModel(n_states=2, n_seeds=2, n_iter=50)
        model.fit(dataset)
        result = model.predict(dataset, segment="test")

        self.assertEqual(len(result), len(test_df))
        self.assertFalse(result["regime"].isna().any())
        self.assertTrue(result["state"].isin(list(model.regime_map.keys())).all())


if __name__ == "__main__":
    unittest.main()
