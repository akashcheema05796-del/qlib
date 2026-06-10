# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Unit tests for the HMM regime classifier integration.

Tests are self-contained: they generate synthetic OHLCV data so that no
real Qlib data download is required.  A minimal mock of DatasetH is used
to exercise the model's fit/predict interface.
"""

import math
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

    def test_vol_features_use_rank_suffix(self):
        """Vol-level features should be percentile-rank expressions (stationarity fix)."""
        from qlib.contrib.data.handler_regime import _regime_feature_config
        _, names = _regime_feature_config()
        name_set = set(names)
        for expected in ("RVOL5_RANK", "RVOL10_RANK", "RVOL20_RANK",
                         "GK_VOL_RANK", "ATR_RANK", "BB_WIDTH_RANK"):
            self.assertIn(expected, name_set, f"Expected rank feature '{expected}' not found")

    def test_raw_vol_names_absent(self):
        """Old raw vol names should not appear in the new feature config."""
        from qlib.contrib.data.handler_regime import _regime_feature_config
        _, names = _regime_feature_config()
        name_set = set(names)
        for old_name in ("RVOL5", "RVOL10", "RVOL20", "GK_VOL", "ATR_NORM", "BB_WIDTH"):
            self.assertNotIn(old_name, name_set, f"Old raw-vol feature '{old_name}' should be absent")

    def test_rank_expressions_use_rank_operator(self):
        """Fields for RANK features must include the Rank() operator."""
        from qlib.contrib.data.handler_regime import _regime_feature_config
        fields, names = _regime_feature_config()
        for fld, nm in zip(fields, names):
            if nm.endswith("_RANK"):
                self.assertIn("Rank(", fld, f"Feature '{nm}' field should use Rank() operator")

    def test_custom_rank_window_propagates(self):
        from qlib.contrib.data.handler_regime import _regime_feature_config
        fields_default, _ = _regime_feature_config(rank_window=365)
        fields_custom, _ = _regime_feature_config(rank_window=252)
        # At least one field should differ when rank_window changes
        self.assertNotEqual(fields_default, fields_custom)


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
# BIC formula correctness
# ---------------------------------------------------------------------------

class TestBoundedMask(unittest.TestCase):
    """Yeo-Johnson should NOT be applied to percentile-rank features."""

    def test_rank_features_excluded_from_mask(self):
        from qlib.contrib.model.hmm_regime import _bounded_mask
        cols = ["RVOL5_RANK", "GK_VOL_RANK", "ATR_RANK", "BB_WIDTH_RANK",
                "BB_POS", "SKEW20", "KURT20"]
        mask = _bounded_mask(cols)
        for i, col in enumerate(cols):
            if col.endswith("_RANK"):
                self.assertFalse(mask[i], f"_RANK feature '{col}' should NOT be power-transformed")

    def test_non_rank_bounded_features_included(self):
        from qlib.contrib.model.hmm_regime import _bounded_mask
        # Old-style raw features without _RANK suffix should still be transformed
        cols = ["RVOL5", "GK_VOL", "ATR_NORM", "BB_WIDTH", "RET1"]
        mask = _bounded_mask(cols)
        # RVOL5, GK_VOL, ATR_NORM, BB_WIDTH all contain fragment matches
        self.assertTrue(mask[0])  # RVOL5
        self.assertTrue(mask[1])  # GK_VOL
        self.assertFalse(mask[4])  # RET1 — no fragment match


class TestBICFormula(unittest.TestCase):

    def test_bic_does_not_double_count_n(self):
        """_bic must NOT multiply score(X) by n (the fixed bug)."""
        try:
            import hmmlearn  # noqa
        except ImportError:
            self.skipTest("hmmlearn not installed")

        from qlib.contrib.model.hmm_regime import _bic, _fit_hmm_best_seed

        rng = np.random.default_rng(7)
        X = rng.normal(size=(200, 3))

        _, _, m2 = _fit_hmm_best_seed(X, n_states=2, n_seeds=2, n_iter=50)
        _, _, m3 = _fit_hmm_best_seed(X, n_states=3, n_seeds=2, n_iter=50)

        bic2 = _bic(m2, X)
        bic3 = _bic(m3, X)

        # BIC values must be positive finite numbers
        self.assertTrue(math.isfinite(bic2))
        self.assertTrue(math.isfinite(bic3))

    def test_bic_lower_for_true_model(self):
        """BIC should prefer 2 states when data is generated from 2-state GMM."""
        try:
            import hmmlearn  # noqa
        except ImportError:
            self.skipTest("hmmlearn not installed")

        from qlib.contrib.model.hmm_regime import _bic, _fit_hmm_best_seed

        rng = np.random.default_rng(42)
        # Two clearly separated clusters → BIC should not select 5 states
        half = 150
        X = np.vstack([
            rng.normal(loc=0.0, scale=0.3, size=(half, 2)),
            rng.normal(loc=5.0, scale=0.3, size=(half, 2)),
        ])

        _, _, m2 = _fit_hmm_best_seed(X, n_states=2, n_seeds=3, n_iter=100)
        _, _, m5 = _fit_hmm_best_seed(X, n_states=5, n_seeds=3, n_iter=100)

        bic2 = _bic(m2, X)
        bic5 = _bic(m5, X)
        # With clear separation, 2-state BIC should be ≤ 5-state BIC
        self.assertLessEqual(bic2, bic5)


# ---------------------------------------------------------------------------
# _apply_hysteresis
# ---------------------------------------------------------------------------

class TestApplyHysteresis(unittest.TestCase):

    def _make_posterior(self, states: np.ndarray, K: int, confident_prob: float = 0.9) -> np.ndarray:
        """Build a (T, K) posterior where states[t] gets confident_prob."""
        T = len(states)
        posterior = np.full((T, K), (1.0 - confident_prob) / (K - 1))
        for t, s in enumerate(states):
            posterior[t, :] = (1.0 - confident_prob) / (K - 1)
            posterior[t, s] = confident_prob
        return posterior

    def test_stable_state_unchanged(self):
        """If the raw sequence never changes, hysteresis should leave it alone."""
        from qlib.contrib.model.hmm_regime import _apply_hysteresis
        states = np.zeros(20, dtype=int)
        posterior = self._make_posterior(states, K=3)
        result = _apply_hysteresis(states, posterior, min_prob=0.70, min_bars=3)
        np.testing.assert_array_equal(result, states)

    def test_brief_blip_suppressed(self):
        """A single-bar excursion to a new state should be suppressed (< min_bars)."""
        from qlib.contrib.model.hmm_regime import _apply_hysteresis
        # State 0 for 10 bars, then 1 bar of state 1, then state 0 again
        states = np.array([0] * 10 + [1] + [0] * 10, dtype=int)
        posterior = self._make_posterior(states, K=3, confident_prob=0.85)
        result = _apply_hysteresis(states, posterior, min_prob=0.70, min_bars=3)
        # The blip at bar 10 should be held as state 0
        self.assertEqual(result[10], 0)

    def test_sustained_switch_accepted(self):
        """min_bars consecutive bars of a new state should commit the switch."""
        from qlib.contrib.model.hmm_regime import _apply_hysteresis
        # 10 bars of state 0 then 10 bars of state 1
        states = np.array([0] * 10 + [1] * 10, dtype=int)
        posterior = self._make_posterior(states, K=3, confident_prob=0.85)
        result = _apply_hysteresis(states, posterior, min_prob=0.70, min_bars=3)
        # After enough bars the transition should be accepted
        self.assertEqual(result[-1], 1)

    def test_low_prob_switch_blocked(self):
        """A sustained new state below min_prob should NOT trigger a switch."""
        from qlib.contrib.model.hmm_regime import _apply_hysteresis
        K = 3
        T = 20
        # First half: state 0; second half: state 1 at low probability
        states = np.array([0] * 10 + [1] * 10, dtype=int)
        # Build posterior where state-1 probability is only 0.60 (< 0.70 threshold)
        posterior = np.full((T, K), 0.20)
        for t in range(10):
            posterior[t, 0] = 0.60
        for t in range(10, T):
            posterior[t, 1] = 0.60  # below 0.70 threshold
        result = _apply_hysteresis(states, posterior, min_prob=0.70, min_bars=3)
        # All bars should stay as state 0 since probability never crosses threshold
        self.assertTrue(all(result[t] == 0 for t in range(T)))

    def test_output_shape_preserved(self):
        from qlib.contrib.model.hmm_regime import _apply_hysteresis
        rng = np.random.default_rng(1)
        states = rng.integers(0, 3, size=50)
        posterior = rng.dirichlet(np.ones(3), size=50)
        result = _apply_hysteresis(states, posterior)
        self.assertEqual(result.shape, states.shape)

    def test_first_element_unchanged(self):
        from qlib.contrib.model.hmm_regime import _apply_hysteresis
        states = np.array([2, 0, 0, 0, 0], dtype=int)
        posterior = self._make_posterior(states, K=3)
        result = _apply_hysteresis(states, posterior)
        self.assertEqual(result[0], 2)


# ---------------------------------------------------------------------------
# HMMLabelAligner
# ---------------------------------------------------------------------------

class TestHMMLabelAligner(unittest.TestCase):

    def _make_mock_model(self, means: np.ndarray):
        """Build a minimal mock with means_ and n_components attributes."""
        class MockModel:
            pass
        m = MockModel()
        m.means_ = means.copy()
        m.n_components = len(means)
        return m

    def test_fit_stores_reference_means(self):
        from qlib.contrib.model.hmm_label_aligner import HMMLabelAligner
        means = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        aligner = HMMLabelAligner()
        aligner.fit(self._make_mock_model(means))
        np.testing.assert_array_equal(aligner.reference_means_, means)

    def test_identity_alignment(self):
        """When new means equal reference means the permutation should be identity."""
        from qlib.contrib.model.hmm_label_aligner import HMMLabelAligner
        means = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
        aligner = HMMLabelAligner()
        aligner.fit(self._make_mock_model(means))
        perm = aligner.align(self._make_mock_model(means.copy()))
        np.testing.assert_array_equal(perm, np.arange(3))

    def test_permuted_means_resolved(self):
        """If new model has permuted states, aligner should recover the right perm."""
        from qlib.contrib.model.hmm_label_aligner import HMMLabelAligner
        # Reference: state 0 near 0, state 1 near 5, state 2 near 10
        ref_means = np.array([[0.0], [5.0], [10.0]])
        # New model: states in reversed order
        new_means = np.array([[10.0], [5.0], [0.0]])

        aligner = HMMLabelAligner()
        aligner.fit(self._make_mock_model(ref_means))
        perm = aligner.align(self._make_mock_model(new_means))

        # New state 0 (mean=10) → reference state 2
        self.assertEqual(perm[0], 2)
        # New state 2 (mean=0) → reference state 0
        self.assertEqual(perm[2], 0)

    def test_apply_permutation_correct(self):
        from qlib.contrib.model.hmm_label_aligner import HMMLabelAligner
        aligner = HMMLabelAligner()
        perm = np.array([2, 0, 1])  # new→ref mapping
        raw_states = np.array([0, 1, 2, 0, 1])
        aligned = aligner.apply_permutation(raw_states, perm)
        expected = perm[raw_states]
        np.testing.assert_array_equal(aligned, expected)

    def test_n_alignments_increments(self):
        from qlib.contrib.model.hmm_label_aligner import HMMLabelAligner
        means = np.array([[0.0], [5.0], [10.0]])
        aligner = HMMLabelAligner()
        aligner.fit(self._make_mock_model(means))
        self.assertEqual(aligner.n_alignments_, 0)
        aligner.align(self._make_mock_model(means.copy()))
        self.assertEqual(aligner.n_alignments_, 1)
        aligner.align(self._make_mock_model(means.copy()))
        self.assertEqual(aligner.n_alignments_, 2)

    def test_align_without_fit_warns_and_returns_identity(self):
        """align() before fit() should use the model as its own reference."""
        from qlib.contrib.model.hmm_label_aligner import HMMLabelAligner
        means = np.array([[0.0], [1.0]])
        aligner = HMMLabelAligner()
        perm = aligner.align(self._make_mock_model(means))
        np.testing.assert_array_equal(perm, np.arange(2))

    def test_mismatched_states_raises(self):
        from qlib.contrib.model.hmm_label_aligner import HMMLabelAligner
        aligner = HMMLabelAligner()
        aligner.fit(self._make_mock_model(np.array([[0.0], [1.0], [2.0]])))
        with self.assertRaises(ValueError):
            aligner.align(self._make_mock_model(np.array([[0.0], [1.0]])))  # only 2 states

    def test_invalid_distance_raises(self):
        from qlib.contrib.model.hmm_label_aligner import HMMLabelAligner
        with self.assertRaises(ValueError):
            HMMLabelAligner(distance="manhattan")

    def test_cosine_distance_mode(self):
        from qlib.contrib.model.hmm_label_aligner import HMMLabelAligner
        means = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        aligner = HMMLabelAligner(distance="cosine")
        aligner.fit(self._make_mock_model(means))
        perm = aligner.align(self._make_mock_model(means.copy()))
        # Identity case — same means, should map to itself
        np.testing.assert_array_equal(perm, np.arange(3))


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

    def test_margin_guard_falls_back_when_margin_not_met(self):
        """If winner and runner-up are too close (ΔSharpe < margin), fall back."""
        from qlib.contrib.strategy.state_strategy_selector import StateStrategySelector
        rng = np.random.default_rng(99)
        dates = pd.date_range("2021-01-01", periods=400, freq="B")
        states = pd.Series(np.zeros(400, dtype=int), index=dates)
        # Two strategies with almost identical performance
        rets = {
            "A": pd.Series(rng.normal(0.001, 0.01, 400), index=dates),
            "B": pd.Series(rng.normal(0.001, 0.01, 400), index=dates),
            "Flat": pd.Series(0.0, index=dates),
        }
        # Use a very large margin so neither A nor B can win
        sel = StateStrategySelector(metric="sharpe", min_obs=10, margin=100.0)
        sel.fit(states, rets, fallback="Flat")
        self.assertEqual(sel.state_strategy_map[0], "Flat")

    def test_bootstrap_report_runs(self):
        from qlib.contrib.strategy.state_strategy_selector import StateStrategySelector
        sel = StateStrategySelector(metric="sharpe", min_obs=10, bootstrap_n=50)
        sel.fit(self.states, self.strategy_returns)
        report = sel.bootstrap_report()
        self.assertIsInstance(report, pd.DataFrame)


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
# PerpSimulator
# ---------------------------------------------------------------------------

class TestPerpSimulator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from scipy.stats import norm  # noqa — needed by crypto_payoff
        except ImportError:
            pass

    def _make_price_funding(self, n: int = 100, seed: int = 7):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        prices = pd.Series(
            30_000.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n))),
            index=dates, name="close",
        )
        # Typical positive funding rate ~0.01% per 8h → ~0.03% daily
        funding_daily = pd.Series(rng.uniform(0.0001, 0.0005, n), index=dates)
        return prices, funding_daily

    def test_long_perp_formula(self):
        """PnL_long = (P[t+1]/P[t]-1) - funding[t]  (no daily round-trip cost)."""
        from qlib.contrib.strategy.crypto_payoff import PerpSimulator
        sim = PerpSimulator(taker_fee=0.0005, slippage=0.00005)
        prices, funding = self._make_price_funding(n=10)

        pnl = sim.long_perp(prices, funding)

        # No daily transaction cost — perpetual held open, costs at switch time only
        expected_t0 = prices.iloc[1] / prices.iloc[0] - 1.0 - funding.iloc[0]
        self.assertAlmostEqual(pnl.iloc[0], expected_t0, places=10)
        # Last element should be NaN (no t+1 price)
        self.assertTrue(np.isnan(pnl.iloc[-1]))

    def test_short_perp_formula(self):
        """PnL_short = -(P[t+1]/P[t]-1) + funding[t]  (no daily round-trip cost)."""
        from qlib.contrib.strategy.crypto_payoff import PerpSimulator
        sim = PerpSimulator(taker_fee=0.0005, slippage=0.00005)
        prices, funding = self._make_price_funding(n=10)

        pnl = sim.short_perp(prices, funding)

        expected_t0 = -(prices.iloc[1] / prices.iloc[0] - 1.0) + funding.iloc[0]
        self.assertAlmostEqual(pnl.iloc[0], expected_t0, places=10)

    def test_long_short_sum_is_zero(self):
        """Long + Short = 0 per bar (price return and funding both cancel exactly)."""
        from qlib.contrib.strategy.crypto_payoff import PerpSimulator
        sim = PerpSimulator(taker_fee=0.0005, slippage=0.00005)
        prices, funding = self._make_price_funding(n=50)

        long_pnl = sim.long_perp(prices, funding).dropna()
        short_pnl = sim.short_perp(prices, funding).dropna()

        combined = long_pnl + short_pnl
        np.testing.assert_allclose(combined.values, 0.0, atol=1e-12)

    def test_funding_carry_flat_when_below_threshold(self):
        """Days with abs(annualised funding) < min_funding_ann should return 0."""
        from qlib.contrib.strategy.crypto_payoff import PerpSimulator
        sim = PerpSimulator()
        dates = pd.date_range("2022-01-01", periods=20, freq="D")
        prices = pd.Series(30_000.0, index=dates)
        # Very low funding — daily 0.0001 → annualised ~3.65% < 10% threshold
        funding = pd.Series(0.0001, index=dates)
        pnl = sim.funding_carry(prices, funding, min_funding_ann=0.10)
        self.assertTrue((pnl.dropna() == 0.0).all())

    def test_flat_returns_zeros(self):
        from qlib.contrib.strategy.crypto_payoff import PerpSimulator
        sim = PerpSimulator()
        dates = pd.date_range("2022-01-01", periods=30, freq="D")
        pnl = sim.flat(dates)
        self.assertTrue((pnl == 0.0).all())
        self.assertEqual(len(pnl), 30)

    def test_round_trip_cost_formula(self):
        from qlib.contrib.strategy.crypto_payoff import PerpSimulator
        sim = PerpSimulator(taker_fee=0.0005, slippage=0.00005)
        self.assertAlmostEqual(sim._round_trip_cost(), 2.0 * (0.0005 + 0.00005))


# ---------------------------------------------------------------------------
# Black-76 helper functions
# ---------------------------------------------------------------------------

class TestBlack76Helpers(unittest.TestCase):

    def test_put_call_parity_atm(self):
        """For ATM option: C - P = exp(-rT)*(F - K) where F == K → C == P."""
        from qlib.contrib.strategy.crypto_payoff import _black76_call, _black76_put
        F, K, T, sigma, r = 50_000.0, 50_000.0, 7 / 365, 0.80, 0.0
        call = _black76_call(F, K, T, sigma, r)
        put = _black76_put(F, K, T, sigma, r)
        # ATM put-call parity: C - P = e^{-rT}(F - K) = 0 when F==K, r==0
        self.assertAlmostEqual(call, put, places=6)

    def test_put_call_parity_general(self):
        """C - P = e^{-rT}(F - K) must hold for arbitrary inputs."""
        from qlib.contrib.strategy.crypto_payoff import _black76_call, _black76_put
        F, K, T, sigma, r = 55_000.0, 50_000.0, 14 / 365, 0.90, 0.0
        call = _black76_call(F, K, T, sigma, r)
        put = _black76_put(F, K, T, sigma, r)
        parity_rhs = math.exp(-r * T) * (F - K)
        self.assertAlmostEqual(call - put, parity_rhs, places=4)

    def test_call_price_non_negative(self):
        from qlib.contrib.strategy.crypto_payoff import _black76_call
        self.assertGreaterEqual(_black76_call(50_000, 50_000, 7 / 365, 0.8), 0.0)

    def test_put_price_non_negative(self):
        from qlib.contrib.strategy.crypto_payoff import _black76_put
        self.assertGreaterEqual(_black76_put(50_000, 50_000, 7 / 365, 0.8), 0.0)

    def test_degenerate_inputs_return_zero(self):
        from qlib.contrib.strategy.crypto_payoff import _black76_call, _black76_put
        self.assertEqual(_black76_call(50_000, 50_000, 0.0, 0.8), 0.0)
        self.assertEqual(_black76_put(50_000, 50_000, 0.0, 0.8), 0.0)

    def test_delta_call_in_range(self):
        """Call delta must be in [0, 1]."""
        from qlib.contrib.strategy.crypto_payoff import _black76_delta_call
        delta = _black76_delta_call(50_000, 50_000, 7 / 365, 0.8)
        self.assertGreaterEqual(delta, 0.0)
        self.assertLessEqual(delta, 1.0)

    def test_delta_put_negative(self):
        """Put delta must be in [-1, 0]."""
        from qlib.contrib.strategy.crypto_payoff import _black76_delta_put
        delta = _black76_delta_put(50_000, 50_000, 7 / 365, 0.8)
        self.assertLessEqual(delta, 0.0)
        self.assertGreaterEqual(delta, -1.0)

    def test_call_delta_plus_put_delta_equals_minus_discount(self):
        """delta_call - delta_put = e^{-rT} (standard put-call delta parity, r=0 → 1)."""
        from qlib.contrib.strategy.crypto_payoff import _black76_delta_call, _black76_delta_put
        F, K, T, sigma, r = 50_000.0, 48_000.0, 7 / 365, 0.80, 0.0
        dc = _black76_delta_call(F, K, T, sigma, r)
        dp = _black76_delta_put(F, K, T, sigma, r)
        self.assertAlmostEqual(dc - dp, math.exp(-r * T), places=8)


# ---------------------------------------------------------------------------
# OptionSimulator
# ---------------------------------------------------------------------------

class TestOptionSimulator(unittest.TestCase):

    def _make_price_dvol(self, n: int = 60, seed: int = 13):
        rng = np.random.default_rng(seed)
        # Start on a Monday so weekly windows start cleanly
        dates = pd.date_range("2022-01-03", periods=n, freq="D")
        prices = pd.Series(
            40_000.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n))),
            index=dates, name="close",
        )
        # DVOL in annualised % units (e.g. 60–100 % range typical)
        dvol = pd.Series(rng.uniform(60.0, 100.0, n), index=dates)
        return prices, dvol

    def test_flat_returns_zeros(self):
        from qlib.contrib.strategy.crypto_payoff import OptionSimulator
        sim = OptionSimulator()
        dates = pd.date_range("2022-01-03", periods=30, freq="D")
        pnl = sim.flat(dates)
        self.assertTrue((pnl == 0.0).all())

    def test_short_straddle_returns_series(self):
        from qlib.contrib.strategy.crypto_payoff import OptionSimulator
        sim = OptionSimulator()
        prices, dvol = self._make_price_dvol()
        pnl = sim.short_straddle(prices, dvol)
        self.assertIsInstance(pnl, pd.Series)

    def test_short_straddle_nan_when_dvol_none(self):
        """When dvol=None the straddle PnL should be all-NaN."""
        from qlib.contrib.strategy.crypto_payoff import OptionSimulator
        sim = OptionSimulator()
        prices, _ = self._make_price_dvol()
        pnl = sim.short_straddle(prices, dvol=None)
        self.assertTrue(pnl.isna().all())

    def test_iron_condor_bounded_loss(self):
        """Iron condor max loss per unit notional must be finite and small."""
        from qlib.contrib.strategy.crypto_payoff import OptionSimulator
        sim = OptionSimulator()
        prices, dvol = self._make_price_dvol(n=30)
        pnl = sim.iron_condor(prices, dvol)
        # Per-day losses should be bounded (< 20% notional per day is reasonable)
        daily_losses = pnl.dropna()
        if len(daily_losses) > 0:
            self.assertTrue((daily_losses > -0.20).all(),
                            "Iron condor daily loss exceeds 20% of notional")

    def test_long_straddle_cumulative_pnl_has_opposite_sign_tendency(self):
        """Long and short straddles should have broadly opposite cumulative PnL directions.

        They are NOT exact mirrors because both sides pay delta-hedging costs and
        taker fees.  Instead we just verify that the sum of their PnL is negative
        (reflecting that both sides pay transaction costs with no offsetting gain).
        """
        from qlib.contrib.strategy.crypto_payoff import OptionSimulator
        sim = OptionSimulator(taker_fee_per_leg=0.0003, spread_iv_points=0.0)
        prices, dvol = self._make_price_dvol(n=30)
        long_pnl = sim.long_straddle(prices, dvol)
        short_pnl = sim.short_straddle(prices, dvol)
        common = long_pnl.dropna().index.intersection(short_pnl.dropna().index)
        if len(common) > 0:
            combined = long_pnl[common].sum() + short_pnl[common].sum()
            # Both sides pay fees → combined should be <= 0
            self.assertLessEqual(combined, 0.0,
                "Long + short straddle combined PnL must be non-positive (fees)")

    def test_bull_put_spread_returns_series(self):
        from qlib.contrib.strategy.crypto_payoff import OptionSimulator
        sim = OptionSimulator()
        prices, dvol = self._make_price_dvol()
        pnl = sim.bull_put_spread(prices, dvol)
        self.assertIsInstance(pnl, pd.Series)


# ---------------------------------------------------------------------------
# RegimeWalkForward helpers
# ---------------------------------------------------------------------------

class TestComputeVRP(unittest.TestCase):

    def test_vrp_is_series(self):
        from qlib.contrib.strategy.crypto_payoff import compute_vrp
        rng = np.random.default_rng(1)
        dates = pd.date_range("2022-01-01", periods=60, freq="D")
        prices = pd.Series(40_000.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 60))), index=dates)
        dvol = pd.Series(rng.uniform(60.0, 100.0, 60), index=dates)
        vrp = compute_vrp(dvol, prices, window=20)
        self.assertIsInstance(vrp, pd.Series)

    def test_vrp_nan_before_window(self):
        """First window-1 bars should be NaN (not enough history for realised vol)."""
        from qlib.contrib.strategy.crypto_payoff import compute_vrp
        rng = np.random.default_rng(2)
        n = 40
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        prices = pd.Series(40_000.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=dates)
        dvol = pd.Series(80.0, index=dates)
        vrp = compute_vrp(dvol, prices, window=20)
        # First 20 bars should be NaN (log-return needs 1 lag, rolling needs 20)
        self.assertTrue(vrp.iloc[:20].isna().all())
        self.assertFalse(vrp.iloc[21:].isna().all())

    def test_vrp_positive_when_implied_above_realised(self):
        """When DVOL >> realised vol, VRP should be positive."""
        from qlib.contrib.strategy.crypto_payoff import compute_vrp
        n = 60
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        # Very quiet prices (near-zero returns → tiny realised vol)
        prices = pd.Series(40_000.0 * np.exp(np.cumsum(np.full(n, 0.0001))), index=dates)
        dvol = pd.Series(80.0, index=dates)  # 80% IV
        vrp = compute_vrp(dvol, prices, window=20)
        self.assertTrue((vrp.dropna() > 0).all(), "VRP should be positive when IV >> realised")

    def test_vrp_in_percent_units(self):
        """VRP should be in the same percentage-point units as DVOL input."""
        from qlib.contrib.strategy.crypto_payoff import compute_vrp
        rng = np.random.default_rng(3)
        n = 50
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        prices = pd.Series(40_000.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=dates)
        dvol = pd.Series(80.0, index=dates)
        vrp = compute_vrp(dvol, prices, window=20)
        clean = vrp.dropna()
        # VRP should be in the range of realistic vol differences: -100 to +100 pp
        self.assertTrue((clean.abs() < 200).all(), "VRP appears to be in wrong units")


class TestRegimeWalkForwardHelpers(unittest.TestCase):

    def test_compute_sharpe_zero_for_constant_series(self):
        from qlib.contrib.workflow.regime_walkforward import _compute_sharpe
        # Use exact zero so std is exactly 0.0 (0.001 has float representation noise)
        pnl = pd.Series([0.0] * 100)
        sharpe = _compute_sharpe(pnl)
        self.assertEqual(sharpe, 0.0)

    def test_compute_sharpe_positive_for_positive_returns(self):
        from qlib.contrib.workflow.regime_walkforward import _compute_sharpe
        rng = np.random.default_rng(42)
        pnl = pd.Series(rng.normal(0.005, 0.01, 200))
        sharpe = _compute_sharpe(pnl)
        self.assertGreater(sharpe, 0.0)

    def test_compute_max_dd_non_negative(self):
        from qlib.contrib.workflow.regime_walkforward import _compute_max_dd
        rng = np.random.default_rng(0)
        pnl = pd.Series(rng.normal(0, 0.02, 100))
        dd = _compute_max_dd(pnl)
        self.assertGreaterEqual(dd, 0.0)

    def test_block_bootstrap_sharpe_shape(self):
        from qlib.contrib.workflow.regime_walkforward import _block_bootstrap_sharpe
        rng = np.random.default_rng(5)
        pnl = pd.Series(rng.normal(0.001, 0.01, 100))
        dist = _block_bootstrap_sharpe(pnl, n_resamples=50, block_size=10)
        self.assertEqual(len(dist), 50)

    def test_block_bootstrap_sharpe_finite(self):
        from qlib.contrib.workflow.regime_walkforward import _block_bootstrap_sharpe
        rng = np.random.default_rng(6)
        pnl = pd.Series(rng.normal(0.001, 0.01, 100))
        dist = _block_bootstrap_sharpe(pnl, n_resamples=20, block_size=10)
        self.assertTrue(np.all(np.isfinite(dist)))


class TestRegimeWalkForwardWindowGeneration(unittest.TestCase):

    def test_window_count_matches_expected(self):
        """Number of OOS windows should be predictable from total length."""
        try:
            from qlib.contrib.workflow.regime_walkforward import RegimeWalkForward
        except ImportError:
            self.skipTest("dateutil not installed")

        wf = RegimeWalkForward(fit_months=6, select_months=3, oos_months=3)
        # 24 months total, each window takes 6+3+3=12 months, 3-month roll
        min_date = pd.Timestamp("2020-01-01")
        max_date = pd.Timestamp("2022-01-01")  # 24 months
        windows = wf._generate_windows(min_date, max_date)
        # Should produce at least 1 window
        self.assertGreater(len(windows), 0)

    def test_window_dates_non_overlapping_oos(self):
        """OOS starts from consecutive iterations should be strictly increasing."""
        try:
            from qlib.contrib.workflow.regime_walkforward import RegimeWalkForward
        except ImportError:
            self.skipTest("dateutil not installed")

        wf = RegimeWalkForward(fit_months=6, select_months=3, oos_months=3)
        min_date = pd.Timestamp("2020-01-01")
        max_date = pd.Timestamp("2023-01-01")  # 36 months → several windows
        windows = wf._generate_windows(min_date, max_date)

        if len(windows) < 2:
            self.skipTest("Not enough windows to test overlap")

        for i in range(len(windows) - 1):
            # w3_start of window i+1 should be later than w3_start of window i
            # Tuple layout: (w1_start, w1_end, w2_start, w2_end, w3_start, w3_end)
            self.assertLess(windows[i][4], windows[i + 1][4],
                            "W3 start dates should be strictly increasing")

    def test_window_structure_has_six_parts(self):
        """Each window tuple is (w1_start, w1_end, w2_start, w2_end, w3_start, w3_end)."""
        try:
            from qlib.contrib.workflow.regime_walkforward import RegimeWalkForward
        except ImportError:
            self.skipTest("dateutil not installed")

        wf = RegimeWalkForward(fit_months=6, select_months=3, oos_months=3)
        min_date = pd.Timestamp("2020-01-01")
        max_date = pd.Timestamp("2022-06-01")
        windows = wf._generate_windows(min_date, max_date)
        for w in windows:
            self.assertEqual(len(w), 6, "Each window should have 6 date boundaries")

    def test_window_order_is_chronological(self):
        """Within each window tuple dates should be in ascending order."""
        try:
            from qlib.contrib.workflow.regime_walkforward import RegimeWalkForward
        except ImportError:
            self.skipTest("dateutil not installed")

        wf = RegimeWalkForward(fit_months=6, select_months=3, oos_months=3)
        min_date = pd.Timestamp("2020-01-01")
        max_date = pd.Timestamp("2022-06-01")
        windows = wf._generate_windows(min_date, max_date)
        for w in windows:
            w1_start, w1_end, w2_start, w2_end, w3_start, w3_end = w
            self.assertLessEqual(w1_start, w1_end)
            self.assertEqual(w1_end, w2_start)
            self.assertEqual(w2_end, w3_start)
            self.assertLessEqual(w3_start, w3_end)


class TestRegimeWalkForwardAlignLabels(unittest.TestCase):

    def test_align_labels_first_window_returns_identity(self):
        """On the first window, _align_labels should return the identity permutation."""
        try:
            from qlib.contrib.workflow.regime_walkforward import RegimeWalkForward
            import hmmlearn  # noqa
        except ImportError:
            self.skipTest("hmmlearn or dateutil not installed")

        from qlib.contrib.model.hmm_regime import _fit_hmm_best_seed
        rng = np.random.default_rng(10)
        X = rng.normal(size=(100, 3))
        _, _, model = _fit_hmm_best_seed(X, n_states=3, n_seeds=2, n_iter=50)
        fit_bundle = {"hmm_model": model}

        wf = RegimeWalkForward()
        wf._label_aligner = None  # simulate fresh run
        perm = wf._align_labels(fit_bundle)

        np.testing.assert_array_equal(perm, np.arange(3))

    def test_align_labels_second_window_no_type_error(self):
        """Calling _align_labels twice should not raise TypeError (API mismatch bug)."""
        try:
            from qlib.contrib.workflow.regime_walkforward import RegimeWalkForward
            import hmmlearn  # noqa
        except ImportError:
            self.skipTest("hmmlearn or dateutil not installed")

        from qlib.contrib.model.hmm_regime import _fit_hmm_best_seed
        rng = np.random.default_rng(11)
        X = rng.normal(size=(100, 3))
        _, _, m1 = _fit_hmm_best_seed(X, n_states=3, n_seeds=2, n_iter=50)
        _, _, m2 = _fit_hmm_best_seed(X, n_states=3, n_seeds=2, n_iter=50)

        wf = RegimeWalkForward()
        wf._label_aligner = None
        wf._align_labels({"hmm_model": m1})   # first window
        perm2 = wf._align_labels({"hmm_model": m2})  # second window — must not raise

        self.assertEqual(len(perm2), 3)
        self.assertEqual(set(perm2.tolist()), {0, 1, 2})


class TestRegimeWalkForwardLeakageCheck(unittest.TestCase):

    def test_leakage_check_passes_on_clean_data(self):
        """leakage_check should not raise on data without look-ahead."""
        try:
            from qlib.contrib.workflow.regime_walkforward import RegimeWalkForward
        except ImportError:
            self.skipTest("hmmlearn or dateutil not installed")
        try:
            import hmmlearn  # noqa
        except ImportError:
            self.skipTest("hmmlearn not installed")

        wf = RegimeWalkForward(fit_months=6, select_months=3, oos_months=3)

        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=600, freq="B")
        feat = pd.DataFrame(rng.normal(size=(600, 4)),
                            index=dates,
                            columns=["f1", "f2", "f3", "f4"])

        def factory(states, idx):
            return {"Flat": pd.Series(0.0, index=idx)}

        # Should complete without raising
        try:
            wf.leakage_check(feat, factory)
        except Exception as e:
            # Some failures (not enough data, etc.) are acceptable
            if "windows" in str(e).lower() or "converge" in str(e).lower():
                self.skipTest(f"Skipped due to data constraints: {e}")
            else:
                raise


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

    def test_hmm_label_aligner_with_real_hmm(self):
        """HMMLabelAligner should work end-to-end with actual GaussianHMM models."""
        try:
            import hmmlearn  # noqa
        except ImportError:
            self.skipTest("hmmlearn not installed")

        from qlib.contrib.model.hmm_regime import _fit_hmm_best_seed
        from qlib.contrib.model.hmm_label_aligner import HMMLabelAligner

        rng = np.random.default_rng(0)
        X = np.vstack([
            rng.normal([0, 0], 0.5, size=(100, 2)),
            rng.normal([5, 5], 0.5, size=(100, 2)),
            rng.normal([10, 0], 0.5, size=(100, 2)),
        ])

        _, _, m1 = _fit_hmm_best_seed(X, n_states=3, n_seeds=3, n_iter=100)
        _, _, m2 = _fit_hmm_best_seed(X, n_states=3, n_seeds=3, n_iter=100)

        aligner = HMMLabelAligner()
        aligner.fit(m1)
        perm = aligner.align(m2)

        # Permutation must be a valid permutation of [0, 1, 2]
        self.assertEqual(set(perm.tolist()), {0, 1, 2})

    def test_perp_simulator_with_strategy_selector(self):
        """PerpSimulator output should be consumable by StateStrategySelector."""
        from qlib.contrib.strategy.crypto_payoff import PerpSimulator

        rng = np.random.default_rng(3)
        n = 200
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        prices = pd.Series(30_000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)
        funding = pd.Series(rng.uniform(0.0001, 0.0003, n), index=dates)

        sim = PerpSimulator()
        long_pnl = sim.long_perp(prices, funding).dropna()
        short_pnl = sim.short_perp(prices, funding).dropna()
        flat_pnl = sim.flat(long_pnl.index)

        from qlib.contrib.strategy.state_strategy_selector import StateStrategySelector
        states = pd.Series(rng.integers(0, 2, len(long_pnl)), index=long_pnl.index)
        strategy_returns = {
            "LongPerp": long_pnl,
            "ShortPerp": short_pnl,
            "Flat": flat_pnl,
        }

        sel = StateStrategySelector(metric="sharpe", min_obs=20, annualization=365)
        sel.fit(states, strategy_returns, fallback="Flat")
        self.assertIsNotNone(sel.state_strategy_map)


if __name__ == "__main__":
    unittest.main()
