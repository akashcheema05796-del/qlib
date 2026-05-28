# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
HMM-based market regime classifier for Qlib.

Implements a GaussianHMM fitted on the cross-sectional mean of regime features
(see handler_regime.py). The detected regime is broadcast to all instruments on
each date so it can be used as a portfolio-level signal by RegimeGatedStrategy.

Key design choices (mirroring the v6 notebook):
  - BIC sweep over n_states in [2, max_states] on the training slice only
  - Multi-seed fitting (best log-likelihood kept)
  - Yeo-Johnson transform for bounded features before assuming Gaussian emissions
  - State → regime name mapping based on train-set volatility/trend statistics
  - LightGBM transition classifier: P(regime changes within k bars)
  - Posterior entropy and top-probability signals for change-point detection
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

from ...log import get_module_logger
from ...model.base import BaseModel
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP

logger = get_module_logger("HMMRegimeModel")


# ---------------------------------------------------------------------------
# Regime name assignment
# ---------------------------------------------------------------------------

def _assign_regime_names(state_stats: pd.DataFrame) -> Dict[int, str]:
    """Map raw HMM state indices to human-readable regime names.

    Sorting is based on realized volatility (rvol) and trend strength (roc)
    computed on the training segment only, so the mapping is deterministic
    across seeds and re-runs on the same training window.

    The names used are intentionally general so they apply to equities, crypto,
    or any other asset class.
    """
    df = state_stats.copy()
    # Composite score: high vol → crisis/spike; high trend + low vol → trend
    df["vol_rank"] = df["mean_rvol"].rank()
    df["trend_rank"] = df["mean_roc_abs"].rank()
    n = len(df)

    name_map: Dict[int, str] = {}
    for state, row in df.iterrows():
        vr = row["vol_rank"]
        tr = row["trend_rank"]
        if vr >= n * 0.75:
            name = "Vol_Spike"
        elif vr <= n * 0.25 and tr >= n * 0.50:
            name = "Low_Vol_Trend"
        elif vr <= n * 0.25:
            name = "Low_Vol_Range"
        elif tr >= n * 0.75:
            name = "High_Vol_Trend"
        else:
            name = "Choppy"
        name_map[int(state)] = name

    # Deduplicate names by appending state index
    seen: Dict[str, int] = {}
    deduped: Dict[int, str] = {}
    for state, name in name_map.items():
        if name in seen:
            deduped[seen[name]] = f"{name}_{seen[name]}"
            deduped[state] = f"{name}_{state}"
        else:
            seen[name] = state
            deduped[state] = name
    return deduped


# ---------------------------------------------------------------------------
# Core HMM fitting helpers
# ---------------------------------------------------------------------------

def _fit_hmm_best_seed(
    X: np.ndarray,
    n_states: int,
    n_seeds: int,
    n_iter: int,
) -> tuple:
    """Fit GaussianHMM with multiple random seeds; return (log-lik, seed, model)."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as e:
        raise ImportError(
            "hmmlearn is required for HMMRegimeModel. "
            "Install it with: pip install hmmlearn"
        ) from e

    best = None
    for seed in range(n_seeds):
        try:
            m = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=n_iter,
                random_state=seed,
                tol=1e-4,
                verbose=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X)
            if not m.monitor_.converged:
                continue
            ll = m.score(X)
            if best is None or ll > best[0]:
                best = (ll, seed, m)
        except Exception:
            continue

    if best is None:
        raise RuntimeError(
            f"HMM failed to converge for n_states={n_states} across {n_seeds} seeds."
        )
    return best


def _bic(model, X: np.ndarray) -> float:
    """Bayesian Information Criterion for a fitted GaussianHMM."""
    n, d = X.shape
    k = model.n_components
    # Parameters: transition matrix + means + full covariances + initial probs
    n_params = k * (k - 1) + k * d + k * d * (d + 1) / 2 + (k - 1)
    return -2 * model.score(X) * n + n_params * np.log(n)


def _posterior_entropy(posterior: np.ndarray) -> np.ndarray:
    """H(t) = -sum_k p_k * log(p_k). Low when model is certain about state."""
    p = np.clip(posterior, 1e-12, 1.0)
    return np.maximum(0.0, -(p * np.log(p)).sum(axis=1))


# ---------------------------------------------------------------------------
# Yeo-Johnson: only apply to bounded/skewed features
# ---------------------------------------------------------------------------

# Features that benefit from Yeo-Johnson before assuming Gaussian emissions.
# Identified by column name fragments.
_BOUNDED_FEATURE_FRAGMENTS = [
    "ATR", "BB_WIDTH", "RVOL", "GK_VOL", "VOL_RATIO", "KURT", "PRICE_POS",
]


def _bounded_mask(columns: List[str]) -> np.ndarray:
    mask = np.zeros(len(columns), dtype=bool)
    for i, col in enumerate(columns):
        if any(frag in col for frag in _BOUNDED_FEATURE_FRAGMENTS):
            mask[i] = True
    return mask


# ---------------------------------------------------------------------------
# LightGBM transition classifier
# ---------------------------------------------------------------------------

def _build_transition_target(states: np.ndarray, k: int) -> np.ndarray:
    """Binary: did the state change within the next k bars? Last k entries = NaN."""
    out = np.zeros(len(states), dtype=float)
    out[-k:] = np.nan
    for t in range(len(states) - k):
        out[t] = 1.0 if (states[t + 1 : t + k + 1] != states[t]).any() else 0.0
    return out


def _build_lag_features(X: np.ndarray, lags: List[int]) -> np.ndarray:
    """Stack lagged rows as additional columns (simple tabular representation)."""
    rows = []
    max_lag = max(lags)
    for i in range(max_lag, len(X)):
        row = np.concatenate([X[i]] + [X[i - l] for l in lags])
        rows.append(row)
    return np.array(rows)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class HMMRegimeModel(BaseModel):
    """Gaussian HMM market regime classifier.

    Fits on the cross-sectional mean of features across all instruments on each
    date (training segment). Predicts a per-date regime that is then broadcast
    to every instrument so it can be used as a portfolio-level signal.

    Parameters
    ----------
    n_states : int or "auto"
        Number of HMM hidden states. ``"auto"`` selects the best k in
        [2, max_states] by BIC on the training slice.
    max_states : int
        Upper bound for BIC sweep when ``n_states="auto"``.
    n_seeds : int
        Number of random seeds tried per (n_states, BIC) candidate.
    n_iter : int
        Maximum EM iterations per HMM fit.
    transition_horizon : int
        Lookahead bars used to build the LightGBM transition-change target.
    transition_lags : list of int
        Lag offsets used to build tabular features for the transition LightGBM.
    lgb_params : dict
        LightGBM hyper-parameters for the transition classifier.
    """

    def __init__(
        self,
        n_states: Union[int, str] = "auto",
        max_states: int = 6,
        n_seeds: int = 10,
        n_iter: int = 300,
        transition_horizon: int = 5,
        transition_lags: Optional[List[int]] = None,
        lgb_params: Optional[Dict] = None,
    ):
        self.n_states = n_states
        self.max_states = max_states
        self.n_seeds = n_seeds
        self.n_iter = n_iter
        self.transition_horizon = transition_horizon
        self.transition_lags = transition_lags or [1, 2, 3, 5]
        self.lgb_params = lgb_params or {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 20,
            "n_estimators": 200,
            "verbosity": -1,
        }

        # Fitted state
        self.hmm_model = None
        self.power_tfm: PowerTransformer = None
        self.scaler: StandardScaler = None
        self.bounded_mask: np.ndarray = None
        self.feature_cols: List[str] = []
        self.regime_map: Dict[int, str] = {}
        self.lgb_trans = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional mean → one row per date."""
        feat = df["feature"] if isinstance(df.columns, pd.MultiIndex) and "feature" in df.columns.get_level_values(0) else df
        return feat.groupby(level="datetime").mean()

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted Yeo-Johnson + StandardScaler."""
        Xt = X.copy()
        if self.bounded_mask.any():
            Xt[:, self.bounded_mask] = self.power_tfm.transform(Xt[:, self.bounded_mask])
        return self.scaler.transform(Xt)

    def _fit_transforms(self, X: np.ndarray):
        self.power_tfm = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()
        if self.bounded_mask.any():
            self.power_tfm.fit(X[:, self.bounded_mask])
            Xt = X.copy()
            Xt[:, self.bounded_mask] = self.power_tfm.transform(X[:, self.bounded_mask])
        else:
            Xt = X
        self.scaler.fit(Xt)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, dataset: DatasetH, reweighter=None):
        """Fit HMM + transition classifier on the training segment."""
        df_train = dataset.prepare("train", col_set=["feature"], data_key=DataHandlerLP.DK_L)
        x_daily = self._aggregate_features(df_train).dropna()
        self.feature_cols = list(x_daily.columns)
        X_raw = x_daily.values.astype(np.float64)

        self.bounded_mask = _bounded_mask(self.feature_cols)
        self._fit_transforms(X_raw)
        X = self._transform(X_raw)

        # --- BIC sweep or fixed n_states ---
        if self.n_states == "auto":
            logger.info("Running BIC sweep for n_states in [2, %d]...", self.max_states)
            best_bic = np.inf
            best_fit = None
            best_k = 2
            for k in range(2, self.max_states + 1):
                try:
                    fit = _fit_hmm_best_seed(X, k, self.n_seeds, self.n_iter)
                    b = _bic(fit[2], X)
                    logger.info("  n_states=%d  BIC=%.1f  seed=%d  ll=%.1f", k, b, fit[1], fit[0])
                    if b < best_bic:
                        best_bic = b
                        best_fit = fit
                        best_k = k
                except RuntimeError:
                    logger.warning("  n_states=%d  failed — skipping", k)
            logger.info("Selected n_states=%d (BIC=%.1f)", best_k, best_bic)
        else:
            best_k = int(self.n_states)
            best_fit = _fit_hmm_best_seed(X, best_k, self.n_seeds, self.n_iter)
            logger.info("Fitted HMM n_states=%d  ll=%.1f  seed=%d", best_k, best_fit[0], best_fit[1])

        self.hmm_model = best_fit[2]
        states_train = self.hmm_model.predict(X)

        # --- State → regime name mapping (train stats only) ---
        stats_rows = []
        for s in range(best_k):
            mask = states_train == s
            if not mask.any():
                continue
            subset = x_daily.iloc[mask]
            rvol_col = next((c for c in self.feature_cols if "RVOL" in c), None)
            roc_col = next((c for c in self.feature_cols if "RET" in c), None)
            stats_rows.append({
                "state": s,
                "mean_rvol": subset[rvol_col].mean() if rvol_col else 0.0,
                "mean_roc_abs": subset[roc_col].abs().mean() if roc_col else 0.0,
            })
        if stats_rows:
            state_stats = pd.DataFrame(stats_rows).set_index("state")
            self.regime_map = _assign_regime_names(state_stats)
        else:
            self.regime_map = {s: f"State_{s}" for s in range(best_k)}

        logger.info("Regime map: %s", self.regime_map)

        # --- Fit LightGBM transition classifier ---
        self._fit_transition_model(X, states_train)

    def _fit_transition_model(self, X: np.ndarray, states: np.ndarray):
        """Fit LightGBM to predict P(regime changes within next k bars)."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("lightgbm not found; transition model disabled.")
            return

        target = _build_transition_target(states, self.transition_horizon)
        valid = ~np.isnan(target)
        X_lag = _build_lag_features(X, self.transition_lags)
        max_lag = max(self.transition_lags)

        # Align: lag features start at max_lag
        target_aligned = target[max_lag:]
        valid_aligned = valid[max_lag:]
        X_use = X_lag[valid_aligned]
        y_use = target_aligned[valid_aligned]

        if len(y_use) < 50 or y_use.sum() < 10:
            logger.warning("Too few transition samples; transition model skipped.")
            return

        self.lgb_trans = lgb.LGBMClassifier(**self.lgb_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.lgb_trans.fit(X_use, y_use)
        logger.info(
            "Transition classifier fitted on %d samples (%.1f%% positive).",
            len(y_use),
            100 * y_use.mean(),
        )

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, dataset: DatasetH, segment: Union[str, slice] = "test") -> pd.DataFrame:
        """Predict regime labels and associated signals for the given segment.

        Returns a DataFrame indexed by (datetime, instrument) with columns:

        - ``state``      : integer HMM state index
        - ``regime``     : human-readable regime name
        - ``entropy``    : posterior entropy (high = uncertain)
        - ``top_prob``   : highest posterior probability (high = certain)
        - ``trans_prob`` : P(regime changes in next k bars) from LightGBM

        All instruments on a given date receive the same regime since the HMM
        is fitted on the cross-sectional mean.
        """
        if self.hmm_model is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        df = dataset.prepare(segment, col_set=["feature"], data_key=DataHandlerLP.DK_I)
        x_daily = self._aggregate_features(df).dropna()

        missing = [c for c in self.feature_cols if c not in x_daily.columns]
        if missing:
            raise ValueError(f"Features missing in dataset: {missing}")

        X_raw = x_daily[self.feature_cols].values.astype(np.float64)
        X = self._transform(X_raw)

        # Viterbi states
        states = self.hmm_model.predict(X)
        posterior = self.hmm_model.predict_proba(X)
        entropy = _posterior_entropy(posterior)
        top_prob = posterior.max(axis=1)

        # Transition probability
        if self.lgb_trans is not None:
            max_lag = max(self.transition_lags)
            X_lag = _build_lag_features(X, self.transition_lags)
            # pad leading rows where we don't have enough lags
            trans_prob = np.full(len(X), 0.0)
            trans_prob[max_lag:] = self.lgb_trans.predict_proba(X_lag)[:, 1]
        else:
            trans_prob = np.zeros(len(X))

        daily_result = pd.DataFrame(
            {
                "state": states,
                "regime": [self.regime_map.get(int(s), f"State_{s}") for s in states],
                "entropy": entropy,
                "top_prob": top_prob,
                "trans_prob": trans_prob,
            },
            index=x_daily.index,
        )

        # Broadcast daily regime to all (datetime, instrument) pairs in segment
        full_index = df.index  # MultiIndex (datetime, instrument)
        dates = full_index.get_level_values("datetime")
        broadcasted = daily_result.reindex(dates).set_index(full_index)

        return broadcasted
