# SAFEX maize factor research

Factor research on JSE/SAFEX white and yellow maize, with CBOT corn and USDZAR as
the cross-market legs. The question is whether a set of CTA-style features has any
predictive power over forward returns, either on the outright price or on the
SAFEX/CBOT arb.

The point of the repo is the process as much as the answer: build features, screen
them honestly, throw away what does not survive, and only then fit a model.

## Layout

    src/features.py    feature generation
    src/model.py       XGB + walk-forward validation + performance stats
    src/utilities.py   IC screening, decay, correlation, plots
    src/ingest.py      raw SAFEX/CBOT/FX loaders
    src/roll.py        roll spine and liquid series
    build.py           builds the parquet panels in data/processed

    notebooks/factor_analysis.ipynb    the pipeline, end to end

## The data problem that shapes everything

The panel is a stitched front-contract series: one row per date, 45 contracts,
median contract about 98 rows. Differencing the raw price across a roll measures
the gap between two delivery months, not a market move.

It is not a small effect. On the outright, naive daily log returns hit a maximum
of 26% against 6% once returns are taken within expiry. On the arb, roll days move
the spread a median of 6 USD/t against a typical daily move under 3, with a worst
case of 66.

So `build_log_returns()` takes changes within each contract and sums them back
into a continuous series - `clog_` for prices, `c_` for spreads. Every feature is
built on those, never on the raw column.

Two consequences worth knowing:

- A 200-day window cannot be computed inside a 98-row contract. The continuous
  series is what makes long windows possible at all.
- Rolls follow the calendar, so calendar features will happily predict them.
  Before the fix, `season_cos` and `days_to_harvest` were the strongest features
  in the screen. Both vanished once rolls could no longer leak.

## Features

Moving-average distance (z-scored), MA crossover, MACD, share of days above the
MA, realised vol and vol regime, volume ratio and trend, seasonality, days to
harvest, and the yellow/white spread z-score.

Spreads get differences rather than logs, because they cross zero - the arb is
negative about 17% of the time.

## Process

1. **IC screen.** Rank IC of every feature against every horizon, with
   Newey-West t-stats at lag = horizon. Overlapping forward returns are not
   independent, and the correction matters: at a 10-day horizon a naive p-value
   of 0.0016 becomes 0.21.
2. **Decay.** IC by horizon. A real signal decays smoothly from a peak; a jagged
   line around zero is noise.
3. **Correlation.** Adjacent MA windows correlate around 0.9, so five features
   are really about two. Anything above 0.8 gets pruned, strongest t first.
4. **Selection.** Bonferroni over the whole screen grid. More features means a
   higher bar for all of them, which is the honest cost of searching.
5. **Model.** XGB, shallow trees, walk-forward. Two loops: the outer one walks
   through time, the inner one picks hyperparameters inside the training data
   only.

Screening runs on the train slice only, and the model drops any fold that would
test inside that window. Otherwise the folds score the selection rather than the
signal.

The embargo is set to the forecast horizon. A 10-day target at row *i* reads
prices through row *i+10*, so training up to the test block puts ten days of the
future into the fit. `embargo = horizon` is the exact minimum.

## What came out

**Outright (USD maize).** Nothing survives. MA distance peaks around 5-21 days at
IC ~0.10 in sample and is dead past a month. Out of sample the mean IC is -0.02
with 3 of 6 folds positive, win rate 48%, and a t-stat of -0.83. Shortening to a
1-day horizon gives five times the independent sample and rules out anything
bigger than ~6bp a trade, against a cost of roughly 10bp just to break even.

**Arb (SAFEX minus CBOT).** Better, and the most interesting result here. On the
roll-safe series, `vol_regime` clears Bonferroni at IC -0.27 (t -4.31) against a
15-day forward move, and `yw_z` survives at t 3.01. Out of sample the model gets
mean IC 0.10 across three folds, +10% a year, -27% max drawdown, Sharpe 0.60.

That is still not significance. The t-stat is 1.23 on 105 non-overlapping trades,
and the three folds only cover 2022 onward, which we already know is the
favourable half of the sample.

**A caveat worth repeating.** With a 10-day hold there are ten equally valid ways
to pick a non-overlapping sample, and on this data they give Sharpes between 0.33
and 1.24. `performance_stats` averages over all of them and reports the min and
max, so a single figure cannot be quoted without its spread.

## Running it

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python build.py

Then run `notebooks/factor_analysis.ipynb` top to bottom. The notebook uses
`%autoreload`, so edits to `src/` take effect without restarting the kernel.
