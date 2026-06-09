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

    def test_predict_returns_dataframe(self):
        result = self.model.predict(self.dataset, segment="test")
        self.assertIsInstance(result, pd.DataFrame)

    def test_predict_has_required_columns(self):
        result = self.model.predict(self.dataset, segment="test")
        for col in ("state", "entropy", "top_prob", "trans_prob"):
            self.assertIn(col, result.columns, f"Missing column: {col}")
        self.assertNotIn("regime", result.columns, "regime column should not be present")

    def test_predict_state_is_integer(self):
        result = self.model.predict(self.dataset, segment="test")
        self.assertTrue(pd.api.types.is_integer_dtype(result["state"]) or
                        result["state"].apply(lambda x: isinstance(x, (int, np.integer))).all())

    def test_predict_index_matches_input(self):
        result = self.model.predict(self.dataset, segment="test")
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

    def test_predict_decode_filtered_no_lookahead(self):
        """filtered decode is the default and produces valid output."""
        result = self.model.predict(self.dataset, segment="test", decode="filtered")
        self.assertIsInstance(result, pd.DataFrame)
        for col in ("state", "entropy", "top_prob", "trans_prob"):
            self.assertIn(col, result.columns)
        self.assertTrue((result["top_prob"] >= 0).all())
        self.assertTrue((result["top_prob"] <= 1).all())

    def test_predict_decode_viterbi(self):
        result = self.model.predict(self.dataset, segment="test", decode="viterbi")
        self.assertIn("state", result.columns)

    def test_predict_decode_smooth(self):
        result = self.model.predict(self.dataset, segment="test", decode="smooth")
        self.assertIn("state", result.columns)

    def test_predict_invalid_decode_raises(self):
        from qlib.contrib.model.hmm_regime import HMMRegimeModel
        with self.assertRaises(ValueError):
            self.model.predict(self.dataset, segment="test", decode="bad_mode")


# ---------------------------------------------------------------------------
# StateStrategySelector
# ---------------------------------------------------------------------------

class TestStateStrategySelector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        dates = pd.date_range("2021-01-01", periods=200, freq="B")
        # Simulated states: 0, 1, 2 cycling
        cls.states = pd.Series(
            np.tile([0, 1, 2], 200)[:200],
            index=dates,
            name="state",
        )
        # Simulated strategy returns (state 0 favours IronCondor, state 1 favours Straddle)
        ic_ret = pd.Series(rng.normal(0.002, 0.01, 200), index=dates)  # generally good
        st_ret = pd.Series(rng.normal(-0.001, 0.02, 200), index=dates)  # noisy
        cls.strategy_returns = {
            "IronCondor": ic_ret,
            "Straddle": st_ret,
            "Flat": pd.Series(0.0, index=dates),
        }

    def test_fit_runs(self):
        from qlib.contrib.strategy.state_strategy_selector import StateStrategySelector
        sel = StateStrategySelector(metric="sharpe", min_obs=10)
        sel.fit(self.states, self.strategy_returns)
        self.assertIsNotNone(sel._state_strategy_map)

    def test_state_strategy_map_covers_all_states(self):
        from qlib.contrib.strategy.state_strategy_selector import StateStrategySelector
        sel = StateStrategySelector(metric="sharpe", min_obs=10)
        sel.fit(self.states, self.strategy_returns)
        mapping = sel.state_strategy_map
        for s in [0, 1, 2]:
            self.assertIn(s, mapping)

    def test_state_strategy_map_values_are_known_strategies(self):
        from qlib.contrib.strategy.state_strategy_selector import StateStrategySelector
        sel = StateStrategySelector(metric="sharpe", min_obs=10)
        sel.fit(self.states, self.strategy_returns)
        known = set(self.strategy_returns.keys())
        for strat in sel.state_strategy_map.values():
            self.assertIn(strat, known)

    def test_report_has_expected_columns(self):
        from qlib.contrib.strategy.state_strategy_selector import StateStrategySelector
        sel = StateStrategySelector(metric="sharpe", min_obs=10)
        sel.fit(self.states, self.strategy_returns)
        report = sel.report()
        for col in ("state", "strategy", "count", "mean", "sharpe", "score"):
            self.assertIn(col, report.columns)

    def test_state_risk_map_in_range(self):
        from qlib.contrib.strategy.state_strategy_selector import StateStrategySelector
        sel = StateStrategySelector(metric="sharpe", min_obs=10)
        sel.fit(self.states, self.strategy_returns)
        risk_map = sel.state_risk_map(base=0.80, floor=0.10)
        for state, risk in risk_map.items():
            self.assertGreaterEqual(risk, 0.10 - 1e-9)
            self.assertLessEqual(risk, 0.80 + 1e-9)

    def test_fallback_for_insufficient_obs(self):
        from qlib.contrib.strategy.state_strategy_selector import StateStrategySelector
        sel = StateStrategySelector(metric="sharpe", min_obs=1000)  # very high threshold
        sel.fit(self.states, self.strategy_returns, fallback="Flat")
        # All states should fall back since no state has 1000 obs
        for strat in sel.state_strategy_map.values():
            self.assertEqual(strat, "Flat")

    def test_invalid_metric_raises(self):
        from qlib.contrib.strategy.state_strategy_selector import StateStrategySelector
        with self.assertRaises(ValueError):
            StateStrategySelector(metric="bad_metric")

    def test_annualization_default_is_365(self):
        from qlib.contrib.strategy.state_strategy_selector import StateStrategySelector
        sel = StateStrategySelector()
        self.assertEqual(sel.annualization, 365)

    def test_empty_strategy_returns_raises(self):
        from qlib.contrib.strategy.state_strategy_selector import StateStrategySelector
        sel = StateStrategySelector()
        with self.assertRaises(ValueError, msg="empty strategy_returns should raise ValueError"):
            sel.fit(self.states, {})


# ---------------------------------------------------------------------------
# Regime-gated strategy (unit-level, no live backtest)
# ---------------------------------------------------------------------------

class TestRegimeGatedStrategy(unittest.TestCase):

    def _make_state_df(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Build a minimal regime_df with state (int) and trans_prob columns."""
        return pd.DataFrame({
            "state": [i % 3 for i in range(len(dates))],
            "entropy": np.random.uniform(0, 1, len(dates)),
            "top_prob": np.random.uniform(0.5, 1.0, len(dates)),
            "trans_prob": np.random.uniform(0, 0.6, len(dates)),
        }, index=dates)

    def test_risk_degree_varies_by_state(self):
        from qlib.contrib.strategy.regime_gated import RegimeGatedStrategy

        dates = pd.date_range("2021-01-01", periods=10, freq="B")
        state_df = self._make_state_df(dates)

        strat = RegimeGatedStrategy.__new__(RegimeGatedStrategy)
        strat._state_risk_map = {0: 0.90, 1: 0.50, 2: 0.20}
        strat._base_risk_degree = 0.80
        strat._trans_prob_thresh = 0.40
        strat._regime_signal = state_df

        strat._current_state = 0
        strat._current_trans_prob = 0.1
        risk_high = strat.get_risk_degree()

        strat._current_state = 2
        strat._current_trans_prob = 0.1
        risk_low = strat.get_risk_degree()

        self.assertGreater(risk_high, risk_low)

    def test_high_trans_prob_reduces_risk(self):
        from qlib.contrib.strategy.regime_gated import RegimeGatedStrategy

        strat = RegimeGatedStrategy.__new__(RegimeGatedStrategy)
        strat._state_risk_map = {}
        strat._base_risk_degree = 0.90
        strat._trans_prob_thresh = 0.40

        strat._current_state = 0
        strat._current_trans_prob = 0.0
        risk_low = strat.get_risk_degree()

        strat._current_trans_prob = 0.80
        risk_high = strat.get_risk_degree()

        self.assertGreater(risk_low, risk_high)

    def test_state_summary(self):
        from qlib.contrib.strategy.regime_gated import RegimeGatedStrategy

        dates = pd.date_range("2021-01-01", periods=30, freq="B")
        state_df = self._make_state_df(dates)

        strat = RegimeGatedStrategy.__new__(RegimeGatedStrategy)
        strat._regime_signal = RegimeGatedStrategy._normalise_regime_signal(state_df)
        strat._state_risk_map = {0: 0.90, 1: 0.50, 2: 0.20}
        strat._base_risk_degree = 0.80

        summary = strat.state_summary()
        self.assertIn("count", summary.columns)
        self.assertIn("risk_degree", summary.columns)

    def test_unknown_state_uses_base_risk(self):
        from qlib.contrib.strategy.regime_gated import RegimeGatedStrategy

        strat = RegimeGatedStrategy.__new__(RegimeGatedStrategy)
        strat._state_risk_map = {0: 0.90}
        strat._base_risk_degree = 0.60
        strat._trans_prob_thresh = 1.0

        strat._current_state = -1  # unknown
        strat._current_trans_prob = 0.0
        self.assertAlmostEqual(strat.get_risk_degree(), 0.60)


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
        self.assertFalse(result["state"].isna().any())
        self.assertNotIn("regime", result.columns)
        # All predicted states must be valid HMM state indices
        valid_states = set(range(model.hmm_model.n_components))
        self.assertTrue(result["state"].isin(valid_states).all())

    def test_state_strategy_selector_end_to_end(self):
        try:
            import hmmlearn  # noqa
        except ImportError:
            self.skipTest("hmmlearn not installed")

        from qlib.contrib.model.hmm_regime import HMMRegimeModel
        from qlib.contrib.strategy.state_strategy_selector import StateStrategySelector

        ohlcv = _make_ohlcv(n_dates=300)
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
        regime_df = model.predict(dataset, segment="train")

        # Get per-date states
        states = regime_df["state"].groupby(level="datetime").first()

        # Synthetic strategy returns (same dates as training states)
        rng = np.random.default_rng(99)
        strategy_returns = {
            "IronCondor": pd.Series(rng.normal(0.001, 0.01, len(states)), index=states.index),
            "Straddle": pd.Series(rng.normal(0.0, 0.02, len(states)), index=states.index),
            "Flat": pd.Series(0.0, index=states.index),
        }

        sel = StateStrategySelector(metric="sharpe", min_obs=5)
        sel.fit(states, strategy_returns, fallback="Flat")

        mapping = sel.state_strategy_map
        self.assertEqual(set(mapping.keys()), set(range(model.hmm_model.n_components)))

        risk_map = sel.state_risk_map()
        self.assertEqual(set(risk_map.keys()), set(mapping.keys()))


if __name__ == "__main__":
    unittest.main()
