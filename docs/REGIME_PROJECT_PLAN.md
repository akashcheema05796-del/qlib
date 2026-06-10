# Project Plan — Regime-Conditioned Crypto Derivatives Trading System

**Status:** Planning baseline (v1.0)
**Assets:** BTC, ETH (Binance spot/perp; Deribit vol data)
**Decision frequency:** Daily (4h optional extension)
**Branch:** `claude/compassionate-clarke-7pxDD`

---

## 1. Research Hypothesis

This project is an experiment, designed to be falsifiable. Three hypotheses,
each with a gate that can kill the project:

- **H1** — BTC/ETH exhibit persistent latent market regimes detectable
  *causally* (no future data) from price, volatility, and funding features.
- **H2** — The optimal derivative strategy (long/short perp, funding carry,
  short vol, long gamma) differs materially across these regimes.
- **H3** — Conditioning strategy selection on the causal regime estimate
  produces out-of-sample risk-adjusted returns superior to unconditional
  baselines, **net of all costs**.

If H3 fails against a two-line vol-targeting rule, the regime machinery is
not earning its complexity and the project stops before live capital.

---

## 2. Scope

**In scope**
- BTC and ETH, daily bars (4h as stretch goal)
- Perpetual futures with real historical funding rates
- Synthetic European options (Black-76) calibrated to Deribit DVOL
- Free public data only: Binance REST API, Deribit public API
- Walk-forward backtesting inside the Qlib framework

**Out of scope**
- Live execution / exchange connectivity
- Altcoins, orderbook microstructure, intraday HFT
- ML alpha models beyond the regime layer (no return forecasting)
- American/exotic options

---

## 3. Architecture — Five Layers

```
┌─────────────────────────────────────────────────────────────┐
│ L5  VALIDATION    walk-forward harness, baselines,          │
│                   bootstrap, sealed holdout                  │
├─────────────────────────────────────────────────────────────┤
│ L4  SELECTION     per-state strategy ranking from PnL,      │
│ + PORTFOLIO       state_risk_map, regime-gated exposure     │
├─────────────────────────────────────────────────────────────┤
│ L3  PAYOFF        PerpSimulator (real funding + fees),      │
│                   OptionSimulator (Black-76 + DVOL + fees)  │
├─────────────────────────────────────────────────────────────┤
│ L2  REGIME        GaussianHMM K=3, filtered decode,         │
│                   label alignment, hysteresis, trans_prob   │
├─────────────────────────────────────────────────────────────┤
│ L1  DATA+FEATURES OHLCV, funding, DVOL → vol-relative       │
│                   stationary features                        │
└─────────────────────────────────────────────────────────────┘
```

Information flows strictly upward. Nothing in L2–L5 may touch raw future
data; the leakage test suite (Phase 6) enforces this mechanically.

---

## 4. Decision Gates (kill criteria between phases)

| Gate | After phase | Pass condition | If fail |
|------|------------|----------------|---------|
| G0 | Data audit | All 3 data series available with <2% gaps over backtest period | Re-scope period or data source |
| G1 | Regime model | States stable across quarterly refits (≥80% label-alignment agreement); mean dwell time ≥ 8 days | Reduce K, lengthen fit window, or stop |
| G2 | Payoff sims | Simulated short-straddle PnL sign-correlates with realized VRP (IV−RV); carry PnL matches funding arithmetic exactly | Fix calibration before selection |
| G3 | Walk-forward | Beats ≥2 of 3 baselines on aggregate OOS Sharpe; t-stat > 1.5 | **Project stops.** Publish negative result |
| G4 | Holdout | Holdout Sharpe > 0.5, max DD < 30% | Do not deploy; document |

---

## 5. Data Plan

| Series | Source | Coverage | Endpoint |
|--------|--------|----------|----------|
| OHLCV daily/4h | Binance spot | BTC 2017-08→, ETH 2017-08→ | `/api/v3/klines` *(collector built)* |
| Funding rates (8h) | Binance futures | BTC 2019-09→, ETH 2020→ | `/fapi/v1/fundingRate` |
| DVOL index (IV) | Deribit | BTC/ETH 2021-03→ | `/api/v2/public/get_index_price_history` |
| Realized vol | computed | full history | close-to-close, Parkinson, Garman-Klass |

**Period split:**
- Backtest universe: 2018-01-01 → 2024-09-30
- **Sealed holdout: 2024-10-01 → end of data (~15 months). Touched exactly
  once, at Phase 8. No parameter may change after first holdout run.**
- Options strategies enter the menu only from 2021-07 (DVOL + 6mo burn-in);
  perp strategies cover the full period. The strategy menu is time-aware.

---

## 6. Feature Layer (stationarity is the design driver)

Raw volatility levels are non-stationary on crypto (BTC RV compressed from
~100% in 2018–2021 to ~40–50% in 2024+). A HMM fit on levels maps all recent
data into the "low vol" state permanently. Therefore: **all vol features
enter as percentile ranks or z-scores against their own rolling 1–2yr
window.**

| Group | Features | Form |
|-------|----------|------|
| Returns | RET 1/5/10/20 | log returns (stationary as-is) |
| Volatility | RV5/10/20, GK, ATR, BB width | **percentile rank, 1yr rolling** |
| Shape | SKEW20, KURT20, BB position | as-is |
| Momentum | ROC10/20, PRICE_POS20 | as-is |
| Volume | VOL_RATIO | as-is (already relative) |
| Carry | funding 30d MA | new |
| Vol premium | VRP = DVOL − RV30; VRP percentile | new (2021+) |

~22 features. Exit check: rolling-mean drift diagnostic on every feature
over the full backtest period.

---

## 7. Regime Layer

- **K = 3 states, fixed.** Not BIC-selected. Rationale: three regimes are
  the natural prior (trending/quiet, crisis/high-vol, choppy) and the data
  cannot statistically support more (see §9). BIC is run once as a
  diagnostic only — after fixing the known bug (`score()` returns total
  log-likelihood; current code multiplies it by n).
- **Fit:** rolling 18-month window, refit quarterly, 15 seeds, keep best
  converged log-likelihood. Cross-sectional mean of BTC+ETH features.
- **Label alignment:** Hungarian assignment (`linear_sum_assignment`) of
  new emission means onto previous window's means. Without this, "state 1"
  changes meaning every refit and the strategy map is garbage.
- **Decoding:** filtered (forward algorithm) only — already implemented.
  Viterbi/smoothed reserved for post-hoc charts.
- **Hysteresis:** commit a state switch only when the filtered probability
  of the new state exceeds 0.70 for 3 consecutive bars. Prevents chatter
  → fee bleed.
- **Transition classifier (LightGBM trans_prob):** demoted to risk-reducer
  only. When trans_prob > 0.45, exposure scales by (1 − trans_prob). It
  never switches strategies.

---

## 8. Payoff Layer

### Perps (fully realistic — real data end to end)
| Strategy | Daily PnL | Notes |
|----------|-----------|-------|
| LongPerp | ΔP/P − funding − fees | |
| ShortPerp | −ΔP/P + funding − fees | |
| FundingCarry | sign(funding)·funding − fees | delta-hedged carry harvest |
| Flat | 0 | always in menu (fallback) |

Fees: 5bps taker per side, 0.5bps slippage.

### Options (synthetic, calibrated — the weakest link, treated accordingly)
ShortStraddle, LongStraddle, IronCondor, BullPutSpread, BearCallSpread.
Black-76, weekly Friday expiry (Deribit convention), daily delta-hedge for
straddles.

**Calibration rule (critical):** IV(t) = DVOL(t). Never an assumed
constant. The entire edge in short-vol strategies is the variance risk
premium (IV − subsequent RV); a simulator without real IV has no VRP and
will select strategies on fiction. Validation (gate G2): simulated
straddle PnL must sign-correlate with realized (DVOL − RV).

Costs: 3bps/leg taker + 0.15 IV-point half-spread on ATM options.

---

## 9. Selection Layer — statistics first

The standard error of an annualized Sharpe from n daily observations is
≈ √(365/n):

| Days in state | SE of annualized Sharpe |
|---------------|------------------------|
| 20 | ±4.3 |
| 50 | ±2.7 |
| 80 | ±2.1 |
| 150 | ±1.6 |

And the *effective* n is smaller still — regime days come in persistent
blocks (a state with 150 days may be 3 independent episodes). Therefore
per-state selection is treated as a **heavily regularized weak prior**, not
a precise ranking:

1. `min_obs = 80` days (raise from 20)
2. **Margin rule:** the winner must beat the runner-up by ΔSharpe ≥ 0.3,
   otherwise fall back to FundingCarry (perps era) / Flat
3. **Selection stability:** block-bootstrap the W2 selection window
   (block ≈ mean state dwell, ~15–20 days, 1,000 resamples); require the
   winner to be selected in ≥60% of resamples, else fall back
4. `state_risk_map` simplified to coarse buckets (e.g., 0.8 / 0.4 / 0.1)
   rather than a Sharpe-linear interpolation — fewer fitted decimals

---

## 10. Walk-Forward Protocol

```
|—— W1: HMM fit (18 mo) ——|—— W2: select (6 mo) ——|—— W3: OOS (3 mo) ——|
                                                       roll +3 mo →
```

- W1: fit HMM, align labels to previous window
- W2: filtered-decode states, run all simulators, select per-state strategy
  under §9 rules
- W3: trade the frozen map. No refits, no peeking. Record everything.
- Roll quarterly. 2018→2024-09 yields **~19 OOS quarters (~57 months OOS)**.

**Baselines run through the identical harness:**
1. Buy-and-hold BTC (beta)
2. Constant FundingCarry (no regime conditioning)
3. Vol-targeting: 40% target vol, exposure = target/RV20, rebalanced daily

**Leakage test suite (mandatory deliverable):** (a) shift all features
forward one day → performance must not improve; (b) perturb future bars →
predictions at t must be bit-identical; (c) assert holdout dates never
appear in any fit/select window.

---

## 11. Evaluation & Statistical Protocol

**Per OOS window:** Sharpe (365d), Calmar, max DD, win rate, switch count,
fees paid, avg exposure.

**Aggregate:** mean/std/min window Sharpe; one-sided t-stat on mean Sharpe
> 0; hit-rate vs each baseline; block-bootstrap 5th-percentile Sharpe on
concatenated OOS PnL.

**Multiple-testing ledger:** every hyperparameter combination ever
evaluated (K, windows, thresholds, metrics, feature sets) is logged in a
ledger file. The final report deflates significance by the number of trials
(deflated Sharpe ratio). This is the honesty mechanism — without it the
walk-forward numbers are quietly cherry-picked.

**Holdout (one shot):** run the final frozen pipeline on 2024-10 → end
exactly once. Result is reported regardless of outcome.

---

## 12. Work Breakdown — Phases, Deliverables, Exit Criteria

| # | Phase | Deliverables | Exit criterion | Effort |
|---|-------|-------------|----------------|--------|
| P0 | **Data audit** | Coverage/gap report for klines, funding, DVOL; holdout boundary fixed in config | Gate G0 | 1 session |
| P1 | **Data pipeline** | `funding_collector.py`, `dvol_collector.py` (+ OHLCV collector exists); QA checks: gaps, outliers, no forward-fill leakage | All series load via Qlib handler | 2 |
| P2 | **Feature layer** | Vol-relative `handler_regime.py` rework; stationarity diagnostics | All features pass drift check | 1 |
| P3 | **Regime model** | BIC bugfix; `hmm_label_aligner.py` (Hungarian); hysteresis; trans_prob demoted to risk-reducer | Gate G1 | 2 |
| P4 | **Payoff library** | `crypto_payoff.py`: PerpSimulator + OptionSimulator, fees baked in; put-call parity + funding-arithmetic unit tests | Gate G2 | 2–3 |
| P5 | **Selection hardening** | min_obs=80, margin rule, bootstrap stability, coarse risk buckets in `StateStrategySelector` | Selector stable on resamples | 1 |
| P6 | **Walk-forward harness** | `regime_walkforward.py`: W1/W2/W3 roller, baselines, leakage test suite, multiple-testing ledger | Leakage suite green; ≥19 windows produced | 2 |
| P7 | **Research run** | Full 2018–2024-09 run; per-window table; aggregate stats; decision memo | Gate G3 — **go/no-go** | 1 |
| P8 | **Holdout + report** | One-shot holdout; final report with deflated Sharpe; `crypto_workflow.py` example | Gate G4 | 1 |
| P9 | *Stretch* | 4h frequency; vol-gate layer (VRP rich/cheap); paper trading harness | — | open |

**Total: ~13–14 working sessions.** P1–P2 parallelizable; P4 depends on P1;
P6 depends on P3+P4+P5.

---

## 13. Engineering Standards (cross-cutting)

- Every phase lands with unit tests green (`tests/test_regime_classifier.py`
  extended per phase); current suite: 29 passing
- All randomness seeded; config (windows, thresholds, fees) in one YAML —
  no magic numbers in code
- Each walk-forward run recorded via Qlib Recorder (params + per-window
  results) so the multiple-testing ledger is automatic
- Repo layout:

```
qlib/contrib/
  data/handler_regime.py            P2 rework
  model/hmm_regime.py               P3 fixes        [built, needs P3]
  model/hmm_label_aligner.py        P3 new
  strategy/regime_gated.py                          [built]
  strategy/state_strategy_selector.py  P5 hardening [built, needs P5]
  strategy/crypto_payoff.py         P4 new
  workflow/regime_walkforward.py    P6 new
scripts/data_collector/
  crypto_binance/collector.py                       [built]
  crypto_binance/funding_collector.py  P1 new
  crypto_deribit/dvol_collector.py     P1 new
examples/regime_classifier/crypto_workflow.py  P8 new
docs/REGIME_PROJECT_PLAN.md         this file
```

---

## 14. Risk Register

| Risk | P | Impact | Mitigation |
|------|---|--------|------------|
| Per-state selection is noise (small n) | **High** | Selects wrong strategies | §9: min_obs 80, margin, bootstrap stability, fallback |
| Label switching across refits | **High** | Strategy map meaningless OOS | P3 Hungarian aligner, gate G1 |
| Synthetic option PnL ≠ real PnL | Med-High | Selector misled | DVOL calibration, gate G2 VRP check |
| Switching costs eat the edge | High | Net PnL negative | Hysteresis, dwell, fees in sims from day 1, switch-count metric |
| Non-stationarity (vol compression) | Med | Stale state mapping | Percentile-rank features, 18mo rolling refit |
| DVOL gap pre-2021 | Med | Short options history | Time-aware menu; perps cover full period |
| Multiple-testing inflation | Med | False discovery | Ledger + deflated Sharpe + sealed holdout |
| HMM non-convergence on 18mo windows | Low-Med | Missing windows | 15 seeds, K=3 fixed, fall back to previous window's model |

---

## 15. Success Criteria (final, ordered)

1. Walk-forward aggregate OOS Sharpe t-stat > 1.5
2. Beats constant FundingCarry in > 60% of OOS windows
3. Beats vol-targeting on aggregate OOS Sharpe
4. OOS max drawdown < 30%
5. Positive net PnL after all fees and switching costs
6. Sealed holdout Sharpe > 0.5

Criteria 1–3 are the go/no-go at Gate G3. The design principle throughout:
**regime classification has value only if regimes genuinely differ in their
optimal strategy, and that difference survives estimation noise, label
drift, and transaction costs.** This plan is built to test that claim
honestly, not to demonstrate it.
