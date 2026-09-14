# SAFEX Maize — CTA Gate & Outright Signal Research

**Date:** 2026-09-11
**Notebook:** `notebooks/ts_momentum.ipynb`
**Data:** SAFEX YMAZ/WMAZ vs CBOT corn, 2010-06-07 → 2026-09-01
**Objective:** find a CTA-style gate to time the SAFEX/CBOT import-parity arb, and
test whether a standalone systematic outright system exists on yellow or white maize.

---

## 0. Headline

| Hypothesis | Verdict |
|---|---|
| Outright has tradeable AR structure | **No** — Ljung-Box passes at daily and weekly |
| Time-series momentum (TSMOM) on the outright | **No** — all formation windows insignificant, betas mildly negative |
| Vol / volume / OI / MA as directional features | **No** — none significant |
| **CTA gate: outright momentum state conditions the arb** | **No** — all four interaction terms insignificant |
| **SA maize scarcity proxy predicts 5d outright return** | **Marginal lead** — t≈2.3 (USD), Sharpe ~0.25–0.30 after window sensitivity; does *not* survive multiple-testing |

Nothing here is a finding. One thing is a lead worth pre-registering.

> **Revision (§9a):** the headline Sharpe of 0.41 is **window-luck**. Neighbouring
> z-score windows give 0.24–0.29. Discount to ~0.25–0.30, which is barely above the
> 0.19 buy-and-hold benchmark.

---

## 1. Pipeline corrections made first

The original analysis differenced the held-contract price level down the date index,
which spans the roll. Three fixes were applied before any result below was trusted.

**`src/roll.py:134` — missing comma.** `("log_safex_usd","log_safex_zar" "log_cbot", ...)`
silently concatenated to `"log_safex_zarlog_cbot"`, so `log_cbot` never reached the
liquid file. Fixing it also exposed a duplicate `log_safex_zar` in the merge list.

**Within-contract returns added.** `build_liquid_series` now emits
`<leg>_log_return` for each outright leg, differenced within expiry and blanked on
roll dates — the same protection `arb_change` already had.

**`ljung_box` / `print_ljung_box` added to `src/features.py`** and wired into the
notebook's ACF/PACF cells.

### Why it mattered

| | naive (spans roll) | within-contract |
|---|---|---|
| std | 0.0162 | 0.0140 |
| max abs return | 0.2491 | 0.0647 |
| n | 3,946 | 3,902 |

Roll dates are **1.1% of observations but 26.1% of the sum of squares**, and all ten
largest "moves" in the naive series were rolls, not market days. Downstream this had
annualised vol 15.7% too high and buy-and-hold Sharpe at 0.144 instead of **0.187**.

### Liquid series validation

```
duplicate dates          0        backward rolls              0
dates monotonic       True        days held into delivery     0
distinct expiries       45        min days to delivery month 10
max calendar gap    5 days        gaps > 7 days               0
```

Identities hold to machine precision: `USD == ZAR/usdzar` exact,
`USD return == ZAR return − FX return` max residual 2.7e-15, `arb_level == USD − CBOT` exact.
Held contract is the #1 open-interest expiry on **92.4%** of days (#2 on 7.2%, #3 on 0.4%);
the residual is the deliberate one-day OI lag plus the no-backward-roll constraint.

---

## 2. Autocorrelation

Clean within-contract log returns, USD.

| Series | LB(5) | LB(10) | LB(20) | bars outside band |
|---|---|---|---|---|
| daily, level (n=3,902) | 0.459 | 0.869 | 0.644 | 5/60 (expect 3) |
| weekly, level (n=848) | 0.344* | 0.741* | 0.509* | 2/60 (expect 3) |
| daily, **squared** | 2e-40 | 2e-53 | 3e-63 | — |
| weekly, **squared** | 0.0004* | 0.0030* | 0.0085* | — |

\* weekly lags are 4 / 8 / 12.

**Returns are serially uncorrelated in the mean but strongly dependent in the variance.**
Squared-return ACF is 0.109 at lag 1, decaying slowly. Direction is not forecastable;
volatility is. That argues for vol in the *position sizer*, not the feature matrix.

---

## 3. Time-series momentum

An ACF test does not rule out TSMOM — TSMOM works through the sum of many tiny
autocorrelations that are individually invisible. Tested properly: pooled overlapping
regression of next-21d return on past-k return, Newey–West(21).

| k (days) | beta | t-NW | p | R² |
|---|---|---|---|---|
| 21 | −0.0798 | −1.31 | 0.192 | 0.0064 |
| 63 | −0.0203 | −0.64 | 0.525 | 0.0012 |
| 126 | −0.0022 | −0.09 | 0.924 | 0.0000 |
| 252 | −0.0346 | −1.81 | 0.070 | 0.0131 |

No momentum at any window. Every beta is negative — a hint of 12-month *reversal*,
economically plausible for an ag (high price → planting response → lower price), but
not significant across 4 windows × 2 specifications.

---

## 4. Generic CTA features (21d forward, RankIC)

| Feature | RankIC | t-NW | p |
|---|---|---|---|
| carry (term-structure slope) | −0.091 | −2.06 | 0.039 |
| arb z-score | +0.097 | 1.61 | 0.107 |
| realised vol 20d | −0.063 | −0.39 | 0.695 |
| vol ratio 20/120 | −0.057 | −0.73 | 0.463 |
| MA 20/100 crossover | −0.017 | −1.39 | 0.163 |
| OI change 20d | −0.024 | −0.56 | 0.575 |
| volume z 20d | −0.015 | −0.14 | 0.892 |

**Vol, volume, OI and moving averages have no directional predictive power.**
Carry's sign is backwards versus the commodity carry literature, which on a single
seasonal ag is more likely old-crop/new-crop seasonality than a risk premium.

### The 21-day arb z-score does not survive proper inference

The +0.097 above used Spearman p-values on overlapping windows, which assume 3,594
independent observations when there are ~171. Under Newey–West(21):

| Specification | beta | t-NW | p |
|---|---|---|---|
| fwd_arb ~ arb_z | +0.598 | 1.21 | 0.225 |
| + contract age control | +0.582 | 1.17 | 0.241 |
| 2010–2018 | +0.639 | 0.77 | 0.440 |
| 2018–2026 | +0.364 | 0.57 | 0.566 |

Not significant, halves out of sample. **Discard.** This is the canonical false
positive in CTA research and it is worth keeping in the record.

### Effective sample size — the binding constraint

16 years of daily data at a 21-day horizon is **~184 independent blocks**. The standard
error of a correlation at n=184 is ≈0.074, so |IC| > 0.145 is needed for significance.
This is why all subsequent work moved to a 5-day horizon (~780 blocks, bar drops to ≈0.07).

---

## 5. Is the arb even mean-reverting?

Per-contract ADF on the arb level, `regression="c"`, contracts with ≥150 observations:

```
contracts tested            82
reject unit root at 5%       3  (3.7%)
median p-value          0.5496
```

**3.7% is below what chance alone gives.** The arb level does not test as mean-reverting
within contract. ADF has poor power at n≈250 against a slow half-life, so this is
"fails to reject", not proof of a unit root — but it does not support a fixed
parity-band fade, and it motivates a time-varying band (Kalman) rather than a
constant 120-day z-score.

---

## 6. THE GATE — does outright momentum state condition the arb?

This was the PM's question. Specification:
`fwd_arb(5d) ~ arb_z + gate + arb_z × gate`, HAC(5). **The interaction term is the claim.**
Reported alongside a non-overlapping (every 5th row) subsample so the two inference
methods can be cross-checked.

| Gate | beta (interaction) | t-NW | p | non-overlap t | non-overlap p |
|---|---|---|---|---|---|
| × mom60 (signed trend) | −0.750 | −0.59 | 0.552 | −1.28 | 0.200 |
| × \|mom60\| (trend strength) | −1.749 | −0.87 | 0.385 | −1.87 | 0.062 |
| × realised vol 20d | −45.099 | −1.65 | 0.099 | −1.36 | 0.175 |
| × vol ratio 20/120 | −0.463 | −1.08 | 0.279 | −0.93 | 0.351 |

n = 3,766 overlapping / 754 non-overlapping.

**Verdict: not supported.** All four insignificant under HAC. `|mom60|` flickers in the
non-overlapping specification (t=−1.87) but its HAC t is −0.87 — the two methods
disagree, which is the signature of noise rather than signal. All four coefficients
share a negative sign, which is at least internally consistent, but none are
distinguishable from zero.

An earlier tercile-bucket version of this test showed p=0.000 across the board. That
was the same overlapping-windows artifact as §4 and has been discarded.

---

## 7. THE OUTRIGHT SYSTEM — arb and yellow-white spread as features

Target: 5-day forward outright log return. All features computed **within contract**
from the panel, evaluated on the held-contract spine.

| Specification (coefficient on last term) | t-NW | p | non-overlap t |
|---|---|---|---|
| own_z — SAFEX yellow's own level | 0.36 | 0.719 | 0.26 |
| cbot_z | −1.63 | 0.103 | −1.22 |
| **arb_z** | **2.03** | **0.042** | 1.49 |
| **yw_z** | **2.30** | **0.021** | **2.23** |
| own_z + arb_z → arb_z | 1.99 | 0.046 | 1.50 |
| own_z + yw_z → yw_z | 2.26 | 0.024 | 2.23 |
| own_z + arb_z + yw_z → yw_z | 1.29 | 0.196 | 1.59 |

Feature correlations: `corr(own_z, arb_z)=0.377`, `corr(own_z, yw_z)=0.268`,
**`corr(arb_z, yw_z)=0.580`**.

Three things to take from this table:

1. **`own_z` is flat (t=0.36).** The outright's own level has no predictive power, which
   rules out the trivial explanation that these are own-mean-reversion in disguise.
2. **Both `arb_z` and `yw_z` survive the `own_z` control**, so they carry genuinely
   relative information.
3. **They are 0.580 correlated and cannibalise each other in the joint regression.**
   This is one signal, not two. `yw_z` is the more robust — significant under *both*
   inference methods, which `arb_z` is not.

Full grid across both outrights and both currencies:

| Target | arb_z t-NW (p) | yw_z t-NW (p) |
|---|---|---|
| Yellow, USD | 2.03 (0.042) | 2.30 (0.021) |
| Yellow, ZAR | 2.37 (0.018) | 1.83 (0.067) |
| White, USD | 1.89 (0.059) | 1.74 (0.082) |
| White, ZAR | 2.22 (0.026) | 1.48 (0.138) |

---

## 8. The economics — and the dynamic that is *not* what it looks like

### The sign is backwards from the pre-registered direction

The hypothesis was beta < 0: SAFEX rich versus import parity should fall back toward it.
**Every estimate came out positive.** Rich versus parity predicts the outright rising
further over the next 5 days.

The defensible reading is that **import parity is a ceiling that binds slowly.**
South Africa's maize interior is landlocked; when SAFEX trades rich it signals genuine
local scarcity, and that scarcity persists for weeks while imports physically arrive —
vessel booking, Durban/Maputo discharge, rail inland. At a 5-day horizon you are
measuring tightness, not convergence. Parity anchors the level over months; it does not
anchor it over a week.

### White rallying against yellow does NOT drop white and lift yellow

This is the important correction to the intuitive story.

A substitution mechanism — white maize (human consumption) gets expensive, consumers
switch to yellow (feed), yellow demand rises, yellow rallies — predicts **opposite
signs** on the two legs. If `yw_z` were a substitution signal, a high white premium
should lift yellow and depress white.

**That is not what the data shows.** Both legs load *positive* on `yw_z`:

```
yellow outright ~ yw_z    t = +2.30 (USD),  +1.83 (ZAR)
white  outright ~ yw_z    t = +1.74 (USD),  +1.48 (ZAR)
```

Same sign, both legs. A high white premium is followed by **both** maize contracts
rising. So `yw_z` is not measuring relative substitution between the two grades — it is
measuring **tightness of the South African maize complex as a whole.**

The mechanism that fits: the white premium blows out in drought and low-stock years,
because white is the food staple with inelastic demand and a thinner deliverable supply,
so it repriced first and hardest. The same drought tightens yellow. The white premium
is therefore an early, high-sensitivity **barometer of local scarcity** — it reacts
before the yellow board does, which is exactly what makes it useful as a feature for
the yellow outright.

This reconciles cleanly with `arb_z` pointing the same direction and being 0.580
correlated with `yw_z`: **both are proxies for the same underlying state — South African
maize scarcity — and that state persists at short horizons.** Two different windows onto
one latent variable, not two independent signals.

### Why white maize is not a second arb

```
corr(arb level,  YM vs WM) = 0.925
corr(arb change, YM vs WM) = 0.830
pooling moves effective n from ~184 to ~193
```

The WM arb is nearly the same bet as the YM arb — shared FX leg, shared CBOT leg,
shared freight. It adds almost no independent information.

The **yellow-white spread** is the opposite:

```
corr( d(YW spread), d(YM arb) ) = 0.078
```

Essentially orthogonal to the arb. That is what makes it worth having.

---

## 9. Strategy test

Long yellow when the scarcity proxy is high. Signal `yw_z` clipped to ±2, vol-targeted
to 15% annualised, position capped at ±3, non-overlapping 5-day rebalance.

```
periods                750 non-overlapping 5d
gross Sharpe          0.41      (buy-and-hold benchmark 0.19)
ann return            9.21%
ann vol              22.68%
turnover               0.32 contracts per rebalance
BREAKEVEN COST          57 bp round trip
1st half Sharpe       +0.33
2nd half Sharpe       +0.47
```

57bp of breakeven is comfortable against SAFEX maize bid-ask (a few rand on ~R4,000/t
≈ 5–15bp round trip) — **verify against desk fills before relying on it.**

### There are no entry rules

This is a **signal test, not a trading strategy**, and the distinction matters:

- **Always in the market.** No threshold, no flat state. z = 0.1 is a small long;
  z = −1.5 is a short. "Entry" and "exit" do not exist as discrete events — only size
  changes.
- **Calendar rebalance every 5 days**, not event-driven.
- **No stop loss, no take profit, no time stop.**

For a spread the physical market calls a widow-maker, continuous exposure with no stop
is precisely the failure mode. Before this is a strategy it needs: a z-threshold to
create a flat state (engage only beyond |z| > 1?), a stop in volatility units, and a
maximum holding period.

---

## 9a. Window sensitivity — the 0.41 does not survive

`yw_z` uses a fixed 120-day lookback. Testing whether that window is special
(HAC t-stats, forward YELLOW ZAR return):

| Window | H=3 | H=5 | H=10 | H=21 | Sharpe (H=5) |
|---|---|---|---|---|---|
| 40 | 1.01 | 1.11 | 1.46 | 0.90 | 0.25 |
| 60 | 1.19 | 1.37 | 1.75 | 1.37 | 0.29 |
| 90 | 1.55 | 1.60 | 1.84 | 1.48 | 0.24 |
| **120** | 1.75 | 1.78 | **1.89** | 1.54 | **0.41** |
| 180 | 1.25 | 1.17 | 1.09 | 0.84 | 0.25 |
| 250 | 1.20 | 1.05 | 1.04 | 0.57 | 0.27 |

**Two readings, and they disagree.**

The t-stats form a smooth hump peaking at 90–120 and decaying either side. A pure
artifact would be jagged, so the hump is mildly reassuring that *something* is there.

But the **Sharpe of 0.41 at exactly 120 is an outlier** — its immediate neighbours give
0.24 and 0.25. That number was luck in how vol-scaling interacted with that specific
lookback, not a property of the signal. **Discount the headline to ~0.25–0.30.**

Note also that no ZAR t-stat reaches 2.0; the 2.30 quoted in §7 is the USD target. The
result is currency-specification dependent as well as window dependent.

In 120's mild defence: 120 trading days ≈ 5.7 months, chosen a priori as a round
~6-month number rather than optimised. That is a weak defence against the table above.

**Implication:** the fixed window is a free parameter the result leans on. Replacing it
with a Kalman-filtered level is now the main path forward, not a refinement. Caveat —
Kalman does not remove the free parameter, it replaces "window length" with the
process/observation noise ratio Q/R. The gain is that Q/R can be estimated by MLE/EM
from the data rather than chosen, and the state adapts to regime, which is right for a
band that moves with drought cycles and the import tariff.

---

## 10. Multiple testing — read this before quoting any number above

Approximately **38 hypothesis tests** were run across this study:

| Block | Tests |
|---|---|
| TSMOM formation windows × specs | 8 |
| Generic CTA features | 7 |
| Gate interactions | 4 |
| Outright specs (2 assets × 2 currencies × 3) | 12 |
| Decomposition specifications | 7 |

Best p-value anywhere: **0.021**. Bonferroni at 38 trials requires **p < 0.0013**.

**Nothing in this document survives multiple-testing correction.** Deflate the 0.41
Sharpe for 38 trials and it is not distinguishable from zero.

What nonetheless makes `yw_z` worth pursuing:

1. Survives the `own_z` control — the obvious spurious explanation is ruled out
2. Consistent under both overlapping (HAC) and non-overlapping inference
3. Stable across both halves of the sample (+0.33 / +0.47)
4. Comfortable breakeven versus realistic transaction costs
5. A coherent economic mechanism, once amended from substitution to scarcity-proxy

Five-for-five is more than data mining usually delivers — but the discipline is
unchanged: **pre-register, then test out of sample.**

---

## 11. Next steps

1. **Pre-register one hypothesis:** *SA maize scarcity proxy (`yw_z`) predicts positive
   5-day yellow outright returns.* Direction specified in advance. Drop `arb_z` — it is
   the weaker of the correlated pair and testing both is what produced 38 trials.
2. **Test out of sample on untouched data:**
   - **WEAT.** `ingest.py` already loads wheat and filters it at `SAFEX_CONTRACTS` —
     one line to enable. An independent SAFEX market.
   - **DCE soy complex**, currently unused in `data/raw/`.
3. **Kalman the band.** Given §5, a constant 120-day window is the wrong model for a
   parity band that drifts with freight, the variable import tariff and FX regime.
   Do it because it is the correct specification, not to rescue a signal — and note it
   costs parameters in a small effective sample.
4. **Add roll transaction costs.** The roll gap is not P&L (closing A and opening B is
   cash-neutral), but the bid-ask crossed at each roll is a real cost and is not yet
   in any backtest here.
5. **Seasonality is still untested** and is the most obvious remaining lever for a
   southern-hemisphere crop (plant Oct–Dec, harvest Apr–Jul).

---

## Appendix — reproducibility notes

- All rolling/lagging features computed with `groupby("expiry")`; frames sorted by
  `["expiry","dt"]` before grouping, since `groupby().diff()` respects row order.
- Features computed on the **panel** (contract's full ~249-day trading history), then
  joined onto the **held spine** (~98 days held). This is why a 60-day lookback is
  available on 98.5% of held days despite the short holding window — a contract has a
  median of 229 days of its own history by the time it becomes the held contract.
- Spreads crossing zero (`ym_corn_arb` negative 13.5%; YW spread negative 45.8%) use
  **first differences**, never log returns.
- No back-adjusted price series is used or required anywhere in this study.
