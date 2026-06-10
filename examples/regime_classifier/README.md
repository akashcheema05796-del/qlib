# Regime-Conditioned Crypto Derivatives Workflow

End-to-end pipeline: HMM market-regime detection → per-state strategy selection →
walk-forward OOS validation, for BTC/ETH perpetual futures and weekly options.

## Prerequisites

```bash
pip install -r ../../requirements/regime_classifier.txt
```

## Pipeline

```
download_data → validate_data → predict_only (optional preview) → run → holdout
```

### 1. `download_data`

```bash
python crypto_workflow.py download_data
```

Fetches three datasets from public APIs:

| Dataset | Source | Available from |
|---|---|---|
| Daily OHLCV | Binance spot | 2018-01 |
| Perpetual funding rates | Binance futures | 2019-09 |
| DVOL implied-vol index | Deribit | 2021-03 |

### 2. `validate_data`

```bash
python crypto_workflow.py validate_data
```

Prints a data-quality report and returns a list of issues. Checks:

- missing/empty files
- calendar gaps > 3 days in OHLCV
- stale prices (7+ identical closes)
- negative/zero prices
- funding NaN rate > 30%
- DVOL NaN rate > 20% or values outside 5–500

Run this before any modelling — silent partial downloads are the most common
source of bogus backtest results.

### 3. `predict_only` (optional)

```bash
python crypto_workflow.py predict_only
```

Quick preview: fits the HMM on a training slice and prints regime predictions,
state distribution, and transition probabilities for a test slice. No
walk-forward, no strategy selection.

### 4. `run` — the research backtest

```bash
python crypto_workflow.py run
```

Executes the full rolling walk-forward protocol over the research period
(2018-01 → 2024-09, keeping the holdout sealed):

- **W1 (18 mo)** — fit GaussianHMM (K=3, 15 seeds) on stationary features
- **W2 (6 mo)** — decode states (forward-filtered + hysteresis), select the
  best strategy per state with min_obs/margin/bootstrap guards
- **W3 (3 mo)** — trade the frozen state→strategy map out-of-sample,
  vol-targeted at 40% annualised, costs charged at strategy switches

The window rolls forward 3 months per iteration. Output: per-window and
aggregate OOS Sharpe (Newey-West HAC t-stat), Calmar, max drawdown, win rate,
and hit rate vs three baselines (buy-and-hold, funding carry, vol-targeting).
Every run is appended to the multiple-testing ledger at
`~/.qlib/regime_trials.csv` with a Bonferroni-corrected p-value.

### 5. `holdout` — run exactly once

```bash
python crypto_workflow.py holdout
```

Evaluates the frozen pipeline on the sealed 2024-10-01+ slice. Prompts for
confirmation; the result stands regardless of outcome. Do not run this until
the research phase is complete and all hyperparameters are final.

## Strategy library

| Name | Description |
|---|---|
| `LongPerp` / `ShortPerp` | Directional perpetual positions; pay/receive funding daily |
| `FundingCarry` | Delta-hedged carry; trades only when \|annualised funding\| ≥ 10% |
| `ShortStraddle` | Weekly ATM short straddle, delta-hedged daily (Black-76, DVOL IV) |
| `IronCondor` | 1σ short strangle + 2σ wings, defined risk |
| `BullPutSpread` | Sell ATM put / buy 10-delta put |
| `VRPShortStraddle` | Short straddle active only when VRP = DVOL − realised vol > 0 |
| `Flat` | Zero position (fallback) |

Option strategies require DVOL and are only valid from **2021-07** onward.

## Configuration

All magic numbers live in the `CFG` dict at the top of `crypto_workflow.py`:
window lengths, HMM parameters, selector guards, holdout boundary. The
`RegimeWalkForward` engine additionally exposes `vol_target` (default 0.40,
`None` disables), `vol_target_window` (20), and `hmm_instrument` (fit the HMM
on a single instrument instead of the cross-sectional mean).

## Files

| File | Purpose |
|---|---|
| `crypto_workflow.py` | This pipeline (crypto derivatives) |
| `workflow.py` | Original equity regime-gating example |

## Tests

```bash
python -m pytest tests/test_regime_classifier.py -v
```

106 tests covering every module; entirely synthetic data, no downloads needed.
